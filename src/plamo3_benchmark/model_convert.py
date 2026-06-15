from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

from .common import die
from .model_artifacts import read_info, save_model_atomic, save_tokenizer_and_configs, trace_len, write_info
from .model_download import ensure_model_access, load_causal_lm, load_tokenizer
from .model_export import export_openvino_model
from .quantization import compress_weights_for_target, compression_mode, is_npu, weight_format


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        die("openvino is not installed. Run `uv sync` first.")
        raise exc
    return ov


def convert(args: Any) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_model_access(args.model, local_files_only=args.local_files_only)

    target_is_npu = is_npu(args.target_device)
    if target_is_npu and args.max_seq_len is None:
        args.max_seq_len = 512
        print("NPU target requested; using --max-seq-len 512 for static KV-cache.", file=sys.stderr)

    ov = _import_openvino()
    tokenizer = load_tokenizer(args)
    xml_path = output_dir / "openvino_model.xml"
    if xml_path.exists() and not args.force:
        return _update_existing(args, output_dir, xml_path, tokenizer, ov)
    return _convert_new(args, output_dir, tokenizer, ov, target_is_npu=target_is_npu)


def _update_existing(args: Any, output_dir: Path, xml_path: Path, tokenizer: Any, ov: Any) -> int:
    print(f"Reusing existing OpenVINO model: {xml_path}", file=sys.stderr)
    info = read_info(output_dir)
    format_name = weight_format(args)
    target_is_npu = is_npu(args.target_device)

    _validate_existing_ir(args, ov, xml_path, info, target_is_npu=target_is_npu)
    if info.get("weight_format") and info.get("weight_format") != format_name:
        die(f"Existing model is {info['weight_format']}; re-run with `--force` to save {format_name}.")

    save_tokenizer_and_configs(ov, tokenizer, args, output_dir)
    existing_len = trace_len(ov, xml_path)
    if target_is_npu:
        existing_len = int(args.max_seq_len or info.get("trace_sequence_length") or existing_len or 512)
    write_info(output_dir, _conversion_info(args, format_name, trace_len=existing_len))
    print(f"Updated existing OpenVINO model directory: {output_dir}", file=sys.stderr)
    return 0


def _validate_existing_ir(args: Any, ov: Any, xml_path: Path, info: dict[str, Any], *, target_is_npu: bool) -> None:
    model = ov.Core().read_model(xml_path)
    input_names = {item.get_any_name() for item in model.inputs}
    stateful = "beam_idx" in input_names and bool(getattr(model, "get_variables", list)())

    if target_is_npu:
        existing_len = int(info.get("trace_sequence_length") or 0) or None
        if info.get("static_shapes") is not True or not stateful or info.get("input_dtype") != "int32":
            die("Existing model is not NPU static stateful KV-cache/int32 IR; re-run convert with `--force`.")
        if info.get("kv_cache_format_version") != 4:
            die("Existing NPU stateful model uses an old KV-cache layout; re-run convert with `--force`.")
        if args.max_seq_len is not None and existing_len is not None and int(args.max_seq_len) != existing_len:
            die(
                f"openvino_model.xml already exists with traced KV-cache length {existing_len}, "
                f"but --max-seq-len {args.max_seq_len} was requested. Re-run with `--force`."
            )
        return

    if not stateful:
        die("Existing model is not stateful GenAI IR; re-run convert with `--force`.")
    if info.get("kv_cache_format_version") != 4:
        die("Existing stateful model uses an old KV-cache layout; re-run convert with `--force`.")


def _convert_new(args: Any, output_dir: Path, tokenizer: Any, ov: Any, *, target_is_npu: bool) -> int:
    try:
        import torch
    except ImportError as exc:
        die("torch is required for conversion. Run `uv sync` first.")
        raise exc

    format_name = weight_format(args)
    dtype = torch.float16 if format_name in {"fp16", "int8", "int4"} else torch.float32

    print("Loading Hugging Face model with trust_remote_code=True...", file=sys.stderr)
    model = load_causal_lm(args, dtype=dtype)
    model.config.use_cache = False

    print("Converting PyTorch model with openvino.convert_model...", file=sys.stderr)
    ov_model, traced_len = export_openvino_model(args, tokenizer, model, ov, torch, dtype)

    del model
    gc.collect()

    ov_model = compress_weights_for_target(ov_model, format_name, npu=target_is_npu)
    save_model_atomic(ov, ov_model, output_dir / "openvino_model.xml", fp16=format_name == "fp16")
    save_tokenizer_and_configs(ov, tokenizer, args, output_dir)
    write_info(output_dir, _conversion_info(args, format_name, trace_len=traced_len))
    print(f"Saved OpenVINO model directory to: {output_dir}", file=sys.stderr)
    return 0


def _conversion_info(args: Any, format_name: str, trace_len: int | None) -> dict[str, Any]:
    target_is_npu = is_npu(args.target_device)
    info = {
        "model": args.model,
        "weight_format": format_name,
        "target_device": args.target_device,
        "compression_mode": compression_mode(format_name, npu=target_is_npu),
        "trace_sequence_length": trace_len,
        "static_shapes": target_is_npu,
        "input_dtype": "int32" if target_is_npu else "int64",
        "uses_kv_cache": True,
        "stateful": True,
        "kv_cache_format_version": 4,
    }

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
