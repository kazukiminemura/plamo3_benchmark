from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "pfnet/plamo-3-nict-8b-base"


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def die(message: str, exit_code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoTokenizer, TextStreamer, set_seed
    except ImportError as exc:
        die("transformers is not installed. Run `uv sync` first.")
        raise exc
    return AutoTokenizer, TextStreamer, set_seed


def is_local_model_path(model: str) -> bool:
    return Path(model).exists()


def looks_like_openvino_dir(path: str) -> bool:
    model_path = Path(path)
    if not model_path.is_dir():
        return False
    return any(model_path.glob("*.xml")) and (model_path / "config.json").exists()


def check_hugging_face_access(model: str, *, local_files_only: bool = False) -> None:
    if is_local_model_path(model):
        return

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return

    try:
        hf_hub_download(repo_id=model, filename="config.json", local_files_only=local_files_only)
    except GatedRepoError:
        die(
            f"Cannot access gated Hugging Face model {model!r}.\n"
            "Open the model page, request/accept access, then authenticate with one of:\n"
            "  uv run huggingface-cli login\n"
            "  $env:HF_TOKEN='<your-token>'\n"
            f"Model page: https://huggingface.co/{model}"
        )
    except RepositoryNotFoundError:
        die(f"Hugging Face model {model!r} was not found, or your account cannot see it.")
    except Exception as exc:
        if local_files_only:
            die(
                f"Hugging Face model {model!r} was not found in the local cache. "
                "Run once without `--local-files-only`, or pass a local model directory. "
                f"Original error: {exc}"
            )
        raise


def sampling_kwargs(args: Any) -> dict[str, Any]:
    do_sample = args.temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
        kwargs["top_k"] = args.top_k
    return kwargs
