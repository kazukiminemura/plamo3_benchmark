from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path
from typing import Any

from .common import check_hugging_face_access, die, import_auto_tokenizer, is_local_model_path


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        die("openvino is not installed. Run `uv sync` first.")
        raise exc
    return ov


def _import_weight_compression() -> tuple[Any, Any, Any, Any]:
    try:
        from nncf import CompressWeightsMode, GroupSizeFallbackMode, compress_weights
        from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    except ImportError as exc:
        die("NNCF is required for int8/int4 weight compression. Run `uv sync` first.")
        raise exc
    return compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters


@contextmanager
def _patch_gqa_attention(torch: Any) -> Any:
    original = torch.nn.functional.scaled_dot_product_attention

    def patched(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("enable_gqa", None)
        if query.shape[-3] != key.shape[-3]:
            repeat = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return original(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def _is_npu(device: str | None) -> bool:
    return bool(device and any(part.startswith("NPU") for part in device.upper().replace(":", ",").split(",")))


def _weight_format(args: Any) -> str:
    if _is_npu(args.target_device) and args.weight_format == "fp32":
        print("warning: NPU target requested with fp32; saving FP16 weights instead.", file=sys.stderr)
        return "fp16"
    return args.weight_format


def _compression_mode(weight_format: str, *, npu: bool) -> str | None:
    if weight_format == "int8":
        return "INT8_SYM" if npu else "INT8_ASYM"
    if weight_format == "int4":
        return "INT4_SYM" if npu else "INT4_ASYM"
    return None


def _compress_weights(ov_model: Any, weight_format: str, *, npu: bool) -> Any:
    mode_name = _compression_mode(weight_format, npu=npu)
    if mode_name is None:
        return ov_model

    compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters = (
        _import_weight_compression()
    )
    mode = getattr(CompressWeightsMode, mode_name)
    kwargs: dict[str, Any] = {}
    advanced_parameters = None
    if weight_format == "int4":
        advanced_parameters = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
        if npu:
            kwargs = {"ratio": 1.0, "group_size": -1}

    print(f"Compressing OpenVINO model weights to {mode_name} with NNCF...", file=sys.stderr)
    return compress_weights(ov_model, mode=mode, advanced_parameters=advanced_parameters, **kwargs)


def _write_json_if_present(model: str, output_dir: Path, filename: str, *, local_files_only: bool) -> None:
    if is_local_model_path(model):
        source = Path(model) / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    try:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(repo_id=model, filename=filename, local_files_only=local_files_only))
    except Exception:
        return
    (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _save_tokenizer_and_configs(tokenizer: Any, args: Any, output_dir: Path) -> None:
    for name in (
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
        "tokenizer.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    tokenizer.save_pretrained(output_dir)
    _write_json_if_present(args.model, output_dir, "config.json", local_files_only=args.local_files_only)
    _write_json_if_present(args.model, output_dir, "generation_config.json", local_files_only=args.local_files_only)
    if not (output_dir / "generation_config.json").exists():
        (output_dir / "generation_config.json").write_text('{"max_new_tokens": 128}\n', encoding="utf-8")


def _read_info(output_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((output_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_info(output_dir: Path, info: dict[str, Any]) -> None:
    (output_dir / "plamo3_ov_conversion.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _trace_len(ov: Any, xml_path: Path) -> int | None:
    try:
        shape = list(ov.Core().read_model(xml_path).outputs[0].get_partial_shape())
    except Exception:
        return None
    return shape[1].get_length() if len(shape) >= 2 and shape[1].is_static else None


def _save_model(ov: Any, ov_model: Any, xml_path: Path, *, fp16: bool) -> None:
    tmp_xml = xml_path.with_name(f"{xml_path.stem}.tmp{xml_path.suffix}")
    ov.save_model(ov_model, tmp_xml, compress_to_fp16=fp16)
    del ov_model
    gc.collect()
    tmp_xml.with_suffix(".bin").replace(xml_path.with_suffix(".bin"))
    tmp_xml.replace(xml_path)


def convert(args: Any) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_hugging_face_access(args.model, local_files_only=args.local_files_only)

    npu = _is_npu(args.target_device)
    if npu and args.max_seq_len is None:
        args.max_seq_len = 512
        print("NPU target requested; using --max-seq-len 512.", file=sys.stderr)

    ov = _import_openvino()
    tokenizer = _load_tokenizer(args)
    xml_path = output_dir / "openvino_model.xml"
    if xml_path.exists() and not args.force:
        return _update_existing(args, output_dir, xml_path, tokenizer, ov)
    return _convert_new(args, output_dir, tokenizer, ov)


def _load_tokenizer(args: Any) -> Any:
    try:
        return import_auto_tokenizer().from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=False,
        )
    except Exception as exc:
        die(f"Failed to load tokenizer for {args.model!r}: {exc}")


def _update_existing(args: Any, output_dir: Path, xml_path: Path, tokenizer: Any, ov: Any) -> int:
    print(f"Reusing existing OpenVINO model: {xml_path}", file=sys.stderr)
    info = _read_info(output_dir)
    weight_format = _weight_format(args)
    trace_len = _trace_len(ov, xml_path)

    uses_kv_cache = bool(info.get("uses_kv_cache", False))
    if args.kv_cache != uses_kv_cache:
        die("Existing model KV-cache setting differs from --kv-cache; re-run with `--force`.")
    if not uses_kv_cache and args.max_seq_len is not None and trace_len is not None and int(args.max_seq_len) != trace_len:
        die(
            f"openvino_model.xml already exists with traced sequence length {trace_len}, "
            f"but --max-seq-len {args.max_seq_len} was requested. Re-run with `--force`."
        )
    if info.get("weight_format") and info.get("weight_format") != weight_format:
        die(f"Existing model is {info['weight_format']}; re-run with `--force` to save {weight_format}.")
    if _is_npu(args.target_device) and (info.get("static_shapes") is not True or info.get("input_dtype") != "int32"):
        die("Existing model is not NPU static-shape/int32 IR; re-run with `--force`.")

    _save_tokenizer_and_configs(tokenizer, args, output_dir)
    _write_info(output_dir, _conversion_info(args, weight_format, trace_len, uses_kv_cache=uses_kv_cache))
    print(f"Updated existing OpenVINO model directory: {output_dir}", file=sys.stderr)
    return 0


def _convert_new(args: Any, output_dir: Path, tokenizer: Any, ov: Any) -> int:
    try:
        from torch import nn
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        die("transformers and torch are required for conversion. Run `uv sync` first.")
        raise exc

    class LogitsOnly(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.model(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask.to(torch.long),
                use_cache=False,
                return_dict=False,
            )[0]

    class KVLogits(nn.Module):
        def __init__(self, model: nn.Module, layers: int) -> None:
            super().__init__()
            self.model = model
            self.layers = layers

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *past: torch.Tensor) -> tuple[Any, ...]:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            for layer_idx in range(self.layers):
                cache.update(past[layer_idx * 2], past[layer_idx * 2 + 1], layer_idx)
            out = self.model(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask.to(torch.long),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            flat_cache = []
            for layer_idx in range(self.layers):
                key, value = out.past_key_values[layer_idx]
                flat_cache.extend([key, value])
            return (out.logits, *flat_cache)

    weight_format = _weight_format(args)
    npu = _is_npu(args.target_device)
    dtype = torch.float16 if weight_format in {"fp16", "int8", "int4"} else torch.float32

    print("Loading Hugging Face model with trust_remote_code=True...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    model.config.use_cache = bool(args.kv_cache)

    token_dtype = torch.int32 if npu else torch.long
    if args.kv_cache:
        encoded = {
            "input_ids": torch.tensor([[tokenizer.bos_token_id or 1]], dtype=token_dtype),
            "attention_mask": torch.ones((1, 1), dtype=token_dtype),
        }
    else:
        if args.max_seq_len is None:
            encoded = tokenizer("こんにちは", return_tensors="pt")
        else:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            encoded = {
                "input_ids": torch.full((1, args.max_seq_len), pad_id, dtype=token_dtype),
                "attention_mask": torch.ones((1, args.max_seq_len), dtype=token_dtype),
            }
    if npu:
        encoded = {name: tensor.to(torch.int32) for name, tensor in encoded.items()}

    print("Converting PyTorch model with openvino.convert_model...", file=sys.stderr)
    if args.kv_cache:
        layers = int(getattr(model.config, "num_hidden_layers"))
        kv_heads = int(getattr(model.config, "num_key_value_heads"))
        head_dim = int(getattr(model.config, "head_dim"))
        example_past = tuple(torch.zeros((1, kv_heads, 0, head_dim), dtype=dtype) for _ in range(layers * 2))
        wrapped_model = KVLogits(model, layers).eval()
        example_input = (encoded["input_ids"], encoded["attention_mask"], *example_past)
    else:
        wrapped_model = LogitsOnly(model).eval()
        example_input = (encoded["input_ids"], encoded["attention_mask"])
    with _patch_gqa_attention(torch):
        ov_model = ov.convert_model(
            wrapped_model,
            example_input=example_input,
            dynamo=True,
        )

    ov_model.inputs[0].get_tensor().set_names({"input_ids"})
    ov_model.inputs[1].get_tensor().set_names({"attention_mask"})
    ov_model.outputs[0].get_tensor().set_names({"logits"})

    trace_len = int(encoded["input_ids"].shape[1])
    if args.kv_cache:
        shapes = {
            "input_ids": ov.PartialShape([1, -1]),
            "attention_mask": ov.PartialShape([1, -1]),
        }
        for layer_idx in range(layers):
            shapes[f"past.{layer_idx}.key"] = ov.PartialShape([1, kv_heads, -1, head_dim])
            shapes[f"past.{layer_idx}.value"] = ov.PartialShape([1, kv_heads, -1, head_dim])
        for idx, input_node in enumerate(ov_model.inputs[2:]):
            input_node.get_tensor().set_names({f"past.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"})
        for idx, output_node in enumerate(ov_model.outputs[1:]):
            output_node.get_tensor().set_names({f"present.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"})
    else:
        shapes = {
            "input_ids": ov.PartialShape([1, trace_len] if npu else [-1, -1]),
            "attention_mask": ov.PartialShape([1, trace_len] if npu else [-1, -1]),
        }
    try:
        ov_model.reshape(shapes)
    except Exception:
        if npu:
            raise
        print("warning: converted model kept the traced prompt shape.", file=sys.stderr)

    ov_model = _compress_weights(ov_model, weight_format, npu=npu)
    _save_model(ov, ov_model, output_dir / "openvino_model.xml", fp16=weight_format == "fp16")
    _save_tokenizer_and_configs(tokenizer, args, output_dir)
    _write_info(output_dir, _conversion_info(args, weight_format, trace_len, uses_kv_cache=bool(args.kv_cache)))
    print(f"Saved OpenVINO model directory to: {output_dir}", file=sys.stderr)
    return 0


def _conversion_info(args: Any, weight_format: str, trace_len: int | None, *, uses_kv_cache: bool) -> dict[str, Any]:
    npu = _is_npu(args.target_device)
    info = {
        "model": args.model,
        "weight_format": weight_format,
        "target_device": args.target_device,
        "compression_mode": _compression_mode(weight_format, npu=npu),
        "trace_sequence_length": trace_len,
        "static_shapes": npu,
        "input_dtype": "int32" if npu else "int64",
        "uses_kv_cache": uses_kv_cache,
    }
    if uses_kv_cache:
        config_path = Path(args.output_dir) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            info.update(
                {
                    "num_hidden_layers": config["num_hidden_layers"],
                    "num_key_value_heads": config["num_key_value_heads"],
                    "head_dim": config["head_dim"],
                }
            )
        except Exception:
            pass
    return info
