from __future__ import annotations

from typing import Any

from .common import check_hugging_face_access, die, import_auto_tokenizer


def ensure_model_access(model: str, *, local_files_only: bool) -> None:
    check_hugging_face_access(model, local_files_only=local_files_only)


def load_tokenizer(args: Any) -> Any:
    try:
        return import_auto_tokenizer().from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=False,
        )
    except Exception as exc:
        die(f"Failed to load tokenizer for {args.model!r}: {exc}")


def load_causal_lm(args: Any, *, dtype: Any) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        die("transformers is required for conversion. Run `uv sync` first.")
        raise exc

    return AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
