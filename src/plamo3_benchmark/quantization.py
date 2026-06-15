from __future__ import annotations

import sys
from typing import Any

from .common import die


def is_npu(device: str | None) -> bool:
    return bool(device and any(part.startswith("NPU") for part in device.upper().replace(":", ",").split(",")))


def weight_format(args: Any) -> str:
    if is_npu(args.target_device) and args.weight_format != "int4":
        print("warning: NPU target requested; using --weight-format int4.", file=sys.stderr)
        return "int4"
    return args.weight_format


def compression_mode(format_name: str, *, npu: bool) -> str | None:
    if format_name == "int8":
        return "INT8_ASYM"
    if format_name == "int4":
        return "INT4_SYM" if npu else "INT4_ASYM"
    return None


def compression_options(format_name: str, *, npu: bool) -> dict[str, Any]:
    if npu and format_name == "int4":
        return {"symmetric": True, "group_size": 128, "ratio": 1.0, "backup_mode": "INT8_SYM"}
    return {}


def compress_weights_for_target(ov_model: Any, format_name: str, *, npu: bool) -> Any:
    mode_name = compression_mode(format_name, npu=npu)
    if mode_name is None:
        return ov_model

    try:
        from nncf import BackupMode, CompressWeightsMode, GroupSizeFallbackMode, compress_weights
        from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    except ImportError as exc:
        die("NNCF is required for int8/int4 weight compression. Run `uv sync` first.")
        raise exc

    kwargs: dict[str, Any] = {}
    advanced_parameters = None
    if format_name == "int4":
        advanced_parameters = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
        if npu:
            kwargs = {"ratio": 1.0, "group_size": 128, "backup_mode": BackupMode.INT8_SYM}

    print(f"Compressing OpenVINO model weights to {mode_name} with NNCF...", file=sys.stderr)
    return compress_weights(
        ov_model,
        mode=getattr(CompressWeightsMode, mode_name),
        advanced_parameters=advanced_parameters,
        **kwargs,
    )
