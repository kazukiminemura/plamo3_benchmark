from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path
from typing import Any

from .common import check_hugging_face_access, die, import_transformers, is_local_model_path


def _import_openvino_converter() -> tuple[Any, Any]:
    try:
        import openvino as ov
        import openvino_tokenizers
    except ImportError as exc:
        die("OpenVINO conversion dependencies are not installed. Run `uv sync` first.")
        raise exc
    return ov, openvino_tokenizers


def _import_nncf() -> tuple[Any, Any, Any, Any]:
    try:
        from nncf import CompressWeightsMode, GroupSizeFallbackMode, compress_weights
        from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    except ImportError as exc:
        die("NNCF is required for weight compression. Run `uv sync` first.")
        raise exc
    return compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters


@contextmanager
def _patch_gqa_scaled_dot_product_attention(torch: Any) -> Any:
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def patched_sdpa(query: Any, key: Any, value: Any, *sdpa_args: Any, **sdpa_kwargs: Any) -> Any:
        sdpa_kwargs.pop("enable_gqa", None)
        query_heads = query.shape[-3]
        key_heads = key.shape[-3]
        if query_heads != key_heads:
            if query_heads % key_heads != 0:
                raise ValueError(f"Cannot expand GQA heads: query_heads={query_heads}, key_heads={key_heads}")
            repeat = query_heads // key_heads
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return original_sdpa(query, key, value, *sdpa_args, **sdpa_kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa


def _write_json_if_present(model: str, output_dir: Path, filename: str, *, local_files_only: bool) -> None:
    if is_local_model_path(model):
        source = Path(model) / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    try:
        source_path = Path(hf_hub_download(repo_id=model, filename=filename, local_files_only=local_files_only))
    except Exception:
        return
    (output_dir / filename).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _save_tokenizer_and_configs(tokenizer: Any, model: str, output_dir: Path, *, local_files_only: bool) -> None:
    tokenizer.save_pretrained(output_dir)
    _write_json_if_present(model, output_dir, "config.json", local_files_only=local_files_only)
    _write_json_if_present(model, output_dir, "generation_config.json", local_files_only=local_files_only)
    if not (output_dir / "generation_config.json").exists():
        (output_dir / "generation_config.json").write_text(
            json.dumps({"max_new_tokens": 128}, indent=2),
            encoding="utf-8",
        )


def _write_conversion_info(output_dir: Path, info: dict[str, Any]) -> None:
    (output_dir / "plamo3_ov_conversion.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_conversion_info(output_dir: Path) -> dict[str, Any]:
    info_path = output_dir / "plamo3_ov_conversion.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _try_save_openvino_tokenizer(tokenizer: Any, ov: Any, openvino_tokenizers: Any, output_dir: Path) -> str:
    print("Converting tokenizer/detokenizer for OpenVINO GenAI...", file=sys.stderr)
    try:
        tokenizer_model, detokenizer_model = openvino_tokenizers.convert_tokenizer(
            tokenizer,
            with_detokenizer=True,
            skip_special_tokens=True,
            streaming_detokenizer=True,
        )
        ov.save_model(tokenizer_model, output_dir / "openvino_tokenizer.xml")
        ov.save_model(detokenizer_model, output_dir / "openvino_detokenizer.xml")
        return "openvino"
    except Exception as exc:
        if tokenizer.__class__.__name__ == "Plamo3Tokenizer":
            fast_tokenizer = _build_plamo3_fast_tokenizer(tokenizer)
            if fast_tokenizer is not None:
                try:
                    tokenizer_model, detokenizer_model = openvino_tokenizers.convert_tokenizer(
                        fast_tokenizer,
                        with_detokenizer=True,
                        skip_special_tokens=True,
                        streaming_detokenizer=True,
                    )
                    ov.save_model(tokenizer_model, output_dir / "openvino_tokenizer.xml")
                    ov.save_model(detokenizer_model, output_dir / "openvino_detokenizer.xml")
                    fast_tokenizer.save_pretrained(output_dir)
                    print(
                        "Converted PLaMo 3 tokenizer through a compatible Fast Unigram tokenizer.",
                        file=sys.stderr,
                    )
                    return "openvino"
                except Exception as fast_exc:
                    print(
                        "warning: PLaMo 3 Fast tokenizer compatibility path failed; saved Hugging Face tokenizer "
                        f"and will use the OpenVINO Core fallback generator. Original errors: {exc}; {fast_exc}",
                        file=sys.stderr,
                    )
                    return "huggingface"
        print(
            "warning: OpenVINO tokenizer conversion failed; saved Hugging Face tokenizer "
            f"and will use the OpenVINO Core fallback generator. Original error: {exc}",
            file=sys.stderr,
        )
        return "huggingface"


def _build_plamo3_fast_tokenizer(tokenizer: Any) -> Any | None:
    try:
        from tokenizers import Regex, Tokenizer
        from tokenizers import models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return None

    data = getattr(tokenizer, "data", None)
    if not data:
        return None

    vocab = [(str(row[0]), float(row[1])) for row in data]
    backend = Tokenizer(models.Unigram(vocab, unk_id=tokenizer.unk_token_id or 0, byte_fallback=True))

    splitters: list[Any] = [
        pre_tokenizers.Split(
            Regex(r"(<\|plamo:[^|\s]{,64}\|>)"),
            behavior="isolated",
            invert=False,
        )
    ]
    boundary_patterns: list[str] = []
    repeated_threshold = getattr(tokenizer, "break_around_repeated_chars_threshold", None)
    spaces_threshold = getattr(tokenizer, "break_around_consecutive_spaces_threshold", None)
    if repeated_threshold is not None:
        boundary_patterns.append(f"(.)\\2{{{int(repeated_threshold) - 1},}}")
    if spaces_threshold is not None:
        boundary_patterns.append(f" {{{int(spaces_threshold)},}}")
    if boundary_patterns:
        splitters.append(
            pre_tokenizers.Split(
                Regex(f"({'|'.join(boundary_patterns)})"),
                behavior="isolated",
                invert=False,
            )
        )
    backend.pre_tokenizer = pre_tokenizers.Sequence(splitters)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=tokenizer.unk_token,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
    )
    fast_tokenizer.add_bos_token = tokenizer.add_bos_token
    fast_tokenizer.add_eos_token = tokenizer.add_eos_token
    return fast_tokenizer


def _die_hf_load_error(model: str, exc: Exception) -> None:
    die(
        f"Failed to load Hugging Face model files for {model!r}. "
        "If this is a transient HEAD/download failure, retry the command. "
        "If the files are already cached, add `--local-files-only`; otherwise check your "
        "network connection and Hugging Face authentication. "
        f"Original error: {exc}"
    )


def _load_tokenizer(AutoTokenizer: Any, args: Any) -> Any:
    try:
        return AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        _die_hf_load_error(args.model, exc)


def _compression_mode_name(weight_format: str, *, npu_target: bool) -> str | None:
    if weight_format == "int8":
        return "INT8_SYM" if npu_target else "INT8_ASYM"
    if weight_format == "int4":
        return "INT4_SYM" if npu_target else "INT4_ASYM"
    return None


def _apply_weight_compression(ov_model: Any, weight_format: str, *, npu_target: bool) -> Any:
    if weight_format not in {"int8", "int4"}:
        return ov_model

    compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters = _import_nncf()
    if weight_format == "int8" and npu_target:
        mode = CompressWeightsMode.INT8_SYM
        advanced_parameters = None
        kwargs: dict[str, Any] = {}
    elif weight_format == "int8":
        mode = CompressWeightsMode.INT8_ASYM
        advanced_parameters = None
        kwargs = {}
    elif npu_target:
        mode = CompressWeightsMode.INT4_SYM
        advanced_parameters = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
        kwargs = {"ratio": 1.0, "group_size": -1}
    else:
        mode = CompressWeightsMode.INT4_ASYM
        advanced_parameters = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
        kwargs = {}
    print(f"Compressing OpenVINO model weights to {mode} with NNCF...", file=sys.stderr)
    return compress_weights(ov_model, mode=mode, advanced_parameters=advanced_parameters, **kwargs)


def _device_targets_npu(device: str | None) -> bool:
    if not device:
        return False
    parts = device.upper().replace(":", ",").split(",")
    return any(part == "NPU" or part.startswith("NPU.") for part in parts)


def _conversion_weight_format(args: Any) -> str:
    if _device_targets_npu(getattr(args, "target_device", None)) and args.weight_format == "fp32":
        print(
            "warning: NPU target requested with --weight-format fp32; saving FP16 weights because "
            "NPU execution is optimized for FP16/quantized IR.",
            file=sys.stderr,
        )
        return "fp16"
    return args.weight_format


def _save_openvino_model(ov: Any, ov_model: Any, xml_path: Path, *, compress_to_fp16: bool) -> None:
    tmp_xml = xml_path.with_name(f"{xml_path.stem}.tmp{xml_path.suffix}")
    tmp_bin = tmp_xml.with_suffix(".bin")
    bin_path = xml_path.with_suffix(".bin")

    for path in (tmp_xml, tmp_bin):
        if path.exists():
            path.unlink()

    ov.save_model(ov_model, tmp_xml, compress_to_fp16=compress_to_fp16)
    del ov_model
    gc.collect()
    tmp_bin.replace(bin_path)
    tmp_xml.replace(xml_path)


def convert(args: Any) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_hugging_face_access(args.model, local_files_only=args.local_files_only)

    if args.weight_format not in {"fp32", "fp16", "int8", "int4"}:
        die("This OpenVINO GenAI conversion path supports --weight-format fp32, fp16, int8, or int4.")
    if _device_targets_npu(getattr(args, "target_device", None)) and args.max_seq_len is None:
        args.max_seq_len = 512
        print(
            "NPU target requested; using --max-seq-len 512 to export a static-shape graph. "
            "Pass --max-seq-len to choose a different fixed context length.",
            file=sys.stderr,
        )

    AutoTokenizer, _, _ = import_transformers()
    ov, openvino_tokenizers = _import_openvino_converter()
    tokenizer = _load_tokenizer(AutoTokenizer, args)

    existing_model = output_dir / "openvino_model.xml"
    if existing_model.exists() and not args.force:
        return _update_existing_model_dir(args, output_dir, existing_model, tokenizer, ov, openvino_tokenizers)

    return _convert_model_from_transformers(args, output_dir, tokenizer, ov, openvino_tokenizers)


def _update_existing_model_dir(
    args: Any,
    output_dir: Path,
    existing_model: Path,
    tokenizer: Any,
    ov: Any,
    openvino_tokenizers: Any,
) -> int:
    print(f"Reusing existing OpenVINO model: {existing_model}", file=sys.stderr)
    existing_info = _read_conversion_info(output_dir)
    weight_format = _conversion_weight_format(args)
    npu_target = _device_targets_npu(getattr(args, "target_device", None))
    if _device_targets_npu(getattr(args, "target_device", None)) and (
        existing_info.get("static_shapes") is not True or existing_info.get("input_dtype") != "int32"
    ):
        die(
            "openvino_model.xml already exists but is not marked as NPU-friendly static-shape/int32 IR. "
            "Re-run with `--force` to rebuild it for NPU, or use a new --output-dir."
        )
    if weight_format in {"int8", "int4"} and existing_info.get("weight_format") != weight_format:
        die(
            f"openvino_model.xml already exists and is not marked as {weight_format.upper()}. "
            "On Windows, in-place compression can leave the existing .bin locked. "
            f"Re-run with `--force` to rebuild and save {weight_format.upper()}, or use a new "
            f"--output-dir such as `ov-plamo3-{weight_format}`."
        )
    compression_mode = _compression_mode_name(weight_format, npu_target=npu_target)
    if npu_target and compression_mode and existing_info.get("compression_mode") != compression_mode:
        die(
            f"openvino_model.xml already exists but is not marked as NPU-friendly {compression_mode}. "
            f"Re-run with `--force` to rebuild it for NPU {weight_format}, or use a new --output-dir."
        )

    try:
        ov_model = ov.Core().read_model(existing_model)
        output_shape = ov_model.outputs[0].get_partial_shape()
        trace_sequence_length = (
            output_shape[1].get_length() if len(output_shape) >= 2 and output_shape[1].is_static else None
        )
    except Exception:
        trace_sequence_length = None

    requested_trace_length = getattr(args, "max_seq_len", None)
    if requested_trace_length is not None and trace_sequence_length is not None:
        if int(requested_trace_length) != int(trace_sequence_length):
            die(
                f"openvino_model.xml already exists with traced sequence length {trace_sequence_length}, "
                f"but --max-seq-len {requested_trace_length} was requested. Re-run with `--force` "
                "to rebuild the model body, or use a new --output-dir."
            )

    _save_tokenizer_and_configs(tokenizer, args.model, output_dir, local_files_only=args.local_files_only)
    tokenizer_backend = _try_save_openvino_tokenizer(tokenizer, ov, openvino_tokenizers, output_dir)

    _write_conversion_info(
        output_dir,
        {
            "model": args.model,
            "weight_format": weight_format,
            "target_device": args.target_device,
            "compression_mode": compression_mode,
            "tokenizer_backend": tokenizer_backend,
            "trace_sequence_length": trace_sequence_length,
            "static_shapes": npu_target,
            "input_dtype": "int32" if npu_target else existing_info.get("input_dtype", "int64"),
            "uses_kv_cache": False,
            "generator": "openvino_core_hf_tokenizer",
        },
    )
    print(f"Updated existing OpenVINO model directory: {output_dir}", file=sys.stderr)
    return 0


def _convert_model_from_transformers(
    args: Any,
    output_dir: Path,
    tokenizer: Any,
    ov: Any,
    openvino_tokenizers: Any,
) -> int:
    try:
        from transformers import AutoModelForCausalLM
        import torch
        from torch import nn
    except ImportError as exc:
        die("transformers and torch are required for conversion. Run `uv sync` first.")
        raise exc

    class CausalLMLogitsWrapper(nn.Module):
        def __init__(self, wrapped_model: nn.Module) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, beam_idx: torch.Tensor) -> torch.Tensor:
            del beam_idx
            return self.wrapped_model(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask.to(torch.long),
                use_cache=False,
                return_dict=False,
            )[0]

    weight_format = _conversion_weight_format(args)
    npu_target = _device_targets_npu(getattr(args, "target_device", None))
    token_dtype = torch.int32 if npu_target else torch.long
    dtype = torch.float16 if weight_format in {"fp16", "int8", "int4"} else torch.float32
    print("Loading Hugging Face model with trust_remote_code=True...", file=sys.stderr)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        _die_hf_load_error(args.model, exc)
    model.eval()
    model.config.use_cache = False
    wrapped_model = CausalLMLogitsWrapper(model).eval()

    if args.max_seq_len is not None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        example_inputs = {
            "input_ids": torch.full((1, args.max_seq_len), pad_id, dtype=token_dtype),
            "attention_mask": torch.ones((1, args.max_seq_len), dtype=token_dtype),
        }
    else:
        prompt = args.example_prompt or "こんにちは"
        example_inputs = tokenizer(prompt, return_tensors="pt")
        if npu_target:
            example_inputs["input_ids"] = example_inputs["input_ids"].to(torch.int32)
            example_inputs["attention_mask"] = example_inputs["attention_mask"].to(torch.int32)
    example_beam_idx = torch.arange(example_inputs["input_ids"].shape[0], dtype=torch.int32)

    print("Converting PyTorch model with openvino.convert_model...", file=sys.stderr)
    try:
        with _patch_gqa_scaled_dot_product_attention(torch):
            ov_model = ov.convert_model(
                wrapped_model,
                example_input=(example_inputs["input_ids"], example_inputs["attention_mask"], example_beam_idx),
                dynamo=True,
            )
        if npu_target:
            trace_len = int(example_inputs["input_ids"].shape[1])
            ov_model.reshape(
                {
                    "input_ids": ov.PartialShape([1, trace_len]),
                    "attention_mask": ov.PartialShape([1, trace_len]),
                    "beam_idx": ov.PartialShape([1]),
                }
            )
        else:
            try:
                ov_model.reshape(
                    {
                        "input_ids": ov.PartialShape([-1, -1]),
                        "attention_mask": ov.PartialShape([-1, -1]),
                        "beam_idx": ov.PartialShape([-1]),
                    }
                )
            except Exception:
                print(
                    "warning: converted model kept the traced prompt shape; use --example-prompt "
                    "with a representative prompt length if generation rejects other lengths.",
                    file=sys.stderr,
                )
    except Exception as exc:
        die(
            "OpenVINO conversion failed for the PLaMo 3 custom PyTorch model. "
            "openvino-genai does not provide a Hugging Face model converter; this route uses "
            "openvino.convert_model and then saves a model directory that OpenVINO GenAI can load "
            f"when tokenizer conversion succeeds. Original error: {exc}"
        )

    compression_mode = _compression_mode_name(weight_format, npu_target=npu_target)
    ov_model = _apply_weight_compression(ov_model, weight_format, npu_target=npu_target)
    _save_openvino_model(
        ov,
        ov_model,
        output_dir / "openvino_model.xml",
        compress_to_fp16=weight_format == "fp16",
    )
    _save_tokenizer_and_configs(tokenizer, args.model, output_dir, local_files_only=args.local_files_only)

    tokenizer_backend = _try_save_openvino_tokenizer(tokenizer, ov, openvino_tokenizers, output_dir)
    _write_conversion_info(
        output_dir,
        {
            "model": args.model,
            "weight_format": weight_format,
            "target_device": args.target_device,
            "compression_mode": compression_mode,
            "tokenizer_backend": tokenizer_backend,
            "trace_sequence_length": int(example_inputs["input_ids"].shape[1]),
            "static_shapes": npu_target,
            "input_dtype": "int32" if npu_target else "int64",
            "uses_kv_cache": False,
            "generator": "openvino_core_hf_tokenizer",
        },
    )

    print(f"Saved experimental OpenVINO model directory to: {output_dir}", file=sys.stderr)
    return 0
