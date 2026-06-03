from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "pfnet/plamo-3-nict-8b-base"


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _die(message: str, exit_code: int = 1) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoTokenizer, TextStreamer, set_seed
    except ImportError as exc:
        _die(
            "transformers is not installed. Run `python -m pip install -e .` first."
        )
        raise exc
    return AutoTokenizer, TextStreamer, set_seed


def _import_openvino_genai() -> Any:
    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        _die(
            "openvino-genai is not installed. Run `uv sync` first."
        )
        raise exc
    return ov_genai


def _import_openvino_converter() -> tuple[Any, Any]:
    try:
        import openvino as ov
        import openvino_tokenizers
    except ImportError as exc:
        _die("OpenVINO conversion dependencies are not installed. Run `uv sync` first.")
        raise exc
    return ov, openvino_tokenizers


def _import_nncf() -> tuple[Any, Any]:
    try:
        from nncf import CompressWeightsMode, compress_weights
    except ImportError as exc:
        _die("NNCF is required for INT8 weight compression. Run `uv sync` first.")
        raise exc
    return compress_weights, CompressWeightsMode


def _read_prompt(args: argparse.Namespace) -> str:
    sources = [args.prompt is not None, args.prompt_file is not None, args.stdin]
    if sum(sources) > 1:
        _die("choose only one prompt source: positional prompt, --prompt-file, or --stdin")

    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")

    if args.stdin:
        return sys.stdin.read()

    if args.prompt is not None:
        return args.prompt

    _die("provide a prompt, --prompt-file, or --stdin")
    return ""


def _looks_like_openvino_dir(path: str) -> bool:
    model_path = Path(path)
    if not model_path.is_dir():
        return False
    return any(model_path.glob("*.xml")) and (model_path / "config.json").exists()


def _is_local_model_path(model: str) -> bool:
    return Path(model).exists()


def _check_hugging_face_access(model: str) -> None:
    if _is_local_model_path(model):
        return

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return

    try:
        hf_hub_download(repo_id=model, filename="config.json")
    except GatedRepoError:
        _die(
            f"Cannot access gated Hugging Face model {model!r}.\n"
            "Open the model page, request/accept access, then authenticate with one of:\n"
            "  uv run huggingface-cli login\n"
            "  $env:HF_TOKEN='<your-token>'\n"
            f"Model page: https://huggingface.co/{model}"
        )
    except RepositoryNotFoundError:
        _die(
            f"Hugging Face model {model!r} was not found, or your account cannot see it."
        )


def _sampling_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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


@contextmanager
def _patch_gqa_scaled_dot_product_attention(torch: Any) -> Any:
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def patched_sdpa(query: Any, key: Any, value: Any, *sdpa_args: Any, **sdpa_kwargs: Any) -> Any:
        sdpa_kwargs.pop("enable_gqa", None)
        query_heads = query.shape[-3]
        key_heads = key.shape[-3]
        if query_heads != key_heads:
            if query_heads % key_heads != 0:
                raise ValueError(
                    f"Cannot expand GQA heads: query_heads={query_heads}, key_heads={key_heads}"
                )
            repeat = query_heads // key_heads
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return original_sdpa(query, key, value, *sdpa_args, **sdpa_kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa


def _write_json_if_present(model: str, output_dir: Path, filename: str) -> None:
    if _is_local_model_path(model):
        source = Path(model) / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return

    try:
        source_path = Path(hf_hub_download(repo_id=model, filename=filename))
    except Exception:
        return
    (output_dir / filename).write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def _save_tokenizer_and_configs(tokenizer: Any, model: str, output_dir: Path) -> None:
    tokenizer.save_pretrained(output_dir)
    _write_json_if_present(model, output_dir, "config.json")
    _write_json_if_present(model, output_dir, "generation_config.json")
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
        print(
            "warning: OpenVINO tokenizer conversion failed; saved Hugging Face tokenizer "
            f"and will use the OpenVINO Core fallback generator. Original error: {exc}",
            file=sys.stderr,
        )
        return "huggingface"


def _apply_weight_compression(ov_model: Any, weight_format: str) -> Any:
    if weight_format != "int8":
        return ov_model

    compress_weights, CompressWeightsMode = _import_nncf()
    print("Compressing OpenVINO model weights to INT8_ASYM with NNCF...", file=sys.stderr)
    return compress_weights(ov_model, mode=CompressWeightsMode.INT8_ASYM)


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


def _read_conversion_info(output_dir: Path) -> dict[str, Any]:
    info_path = output_dir / "plamo3_ov_conversion.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def convert(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_hugging_face_access(args.model)

    if args.weight_format not in {"fp32", "fp16", "int8"}:
        _die(
            "This OpenVINO GenAI conversion path supports --weight-format fp32, fp16, or int8."
        )

    AutoTokenizer, _, _ = _import_transformers()
    ov, openvino_tokenizers = _import_openvino_converter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    existing_model = output_dir / "openvino_model.xml"
    if existing_model.exists() and not args.force:
        print(f"Reusing existing OpenVINO model: {existing_model}", file=sys.stderr)
        existing_info = _read_conversion_info(output_dir)
        if args.weight_format == "int8" and existing_info.get("weight_format") != "int8":
            _die(
                "openvino_model.xml already exists and is not marked as INT8. "
                "On Windows, in-place compression can leave the existing .bin locked. "
                "Re-run with `--force` to rebuild and save INT8, or use a new "
                "--output-dir such as `ov-plamo3-int8`."
            )

        _save_tokenizer_and_configs(tokenizer, args.model, output_dir)
        tokenizer_backend = _try_save_openvino_tokenizer(tokenizer, ov, openvino_tokenizers, output_dir)
        try:
            ov_model = ov.Core().read_model(existing_model)
            output_shape = ov_model.outputs[0].get_partial_shape()
            trace_sequence_length = (
                output_shape[1].get_length()
                if len(output_shape) >= 2 and output_shape[1].is_static
                else None
            )
        except Exception:
            trace_sequence_length = None
        _write_conversion_info(
            output_dir,
            {
                "model": args.model,
                "weight_format": args.weight_format,
                "tokenizer_backend": tokenizer_backend,
                "trace_sequence_length": trace_sequence_length,
                "uses_kv_cache": False,
                "generator": "openvino_genai" if tokenizer_backend == "openvino" else "openvino_core_hf_tokenizer",
            },
        )
        print(f"Updated existing OpenVINO model directory: {output_dir}", file=sys.stderr)
        return 0

    try:
        from transformers import AutoModelForCausalLM
        import torch
        from torch import nn
    except ImportError as exc:
        _die("transformers and torch are required for conversion. Run `uv sync` first.")
        raise exc

    class CausalLMLogitsWrapper(nn.Module):
        def __init__(self, wrapped_model: nn.Module) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            beam_idx: torch.Tensor,
        ) -> torch.Tensor:
            del beam_idx
            return self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )[0]

    dtype = torch.float16 if args.weight_format in {"fp16", "int8"} else torch.float32
    print("Loading Hugging Face model with trust_remote_code=True...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    wrapped_model = CausalLMLogitsWrapper(model).eval()

    if args.max_seq_len is not None:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        example_inputs = {
            "input_ids": torch.full((1, args.max_seq_len), pad_id, dtype=torch.long),
            "attention_mask": torch.ones((1, args.max_seq_len), dtype=torch.long),
        }
    else:
        prompt = args.example_prompt or "こんにちは"
        example_inputs = tokenizer(prompt, return_tensors="pt")
    example_beam_idx = torch.arange(example_inputs["input_ids"].shape[0], dtype=torch.int32)

    print("Converting PyTorch model with openvino.convert_model...", file=sys.stderr)
    try:
        with _patch_gqa_scaled_dot_product_attention(torch):
            ov_model = ov.convert_model(
                wrapped_model,
                example_input=(
                    example_inputs["input_ids"],
                    example_inputs["attention_mask"],
                    example_beam_idx,
                ),
                dynamo=True,
            )
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
        _die(
            "OpenVINO conversion failed for the PLaMo 3 custom PyTorch model. "
            "This route avoids optimum-intel, but PLaMo 3 may still require a custom "
            f"OpenVINO GenAI model adapter/exporter. Original error: {exc}"
        )

    ov_model = _apply_weight_compression(ov_model, args.weight_format)
    _save_openvino_model(
        ov,
        ov_model,
        output_dir / "openvino_model.xml",
        compress_to_fp16=args.weight_format == "fp16",
    )
    _save_tokenizer_and_configs(tokenizer, args.model, output_dir)

    tokenizer_backend = _try_save_openvino_tokenizer(tokenizer, ov, openvino_tokenizers, output_dir)
    _write_conversion_info(
        output_dir,
        {
            "model": args.model,
            "weight_format": args.weight_format,
            "tokenizer_backend": tokenizer_backend,
            "trace_sequence_length": int(example_inputs["input_ids"].shape[1]),
            "uses_kv_cache": False,
            "generator": "openvino_genai" if tokenizer_backend == "openvino" else "openvino_core_hf_tokenizer",
        },
    )

    print(f"Saved experimental OpenVINO GenAI model directory to: {output_dir}", file=sys.stderr)
    return 0


def _sample_next_token(logits: Any, args: argparse.Namespace) -> int:
    import numpy as np

    logits = logits.astype("float64")
    if args.repetition_penalty != 1.0:
        pass
    if args.temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / args.temperature
    if args.top_k > 0 and args.top_k < logits.shape[-1]:
        keep = np.argpartition(logits, -args.top_k)[-args.top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked

    probs = np.exp(logits - np.nanmax(logits))
    probs = probs / probs.sum()
    if args.top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_count = max(1, int(np.searchsorted(cumulative, args.top_p, side="left") + 1))
        keep = order[:keep_count]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()
    return int(np.random.choice(np.arange(probs.shape[-1]), p=probs))


def _generate_with_openvino_core(args: argparse.Namespace, prompt: str) -> int:
    import numpy as np
    import openvino as ov

    AutoTokenizer, _, _ = _import_transformers()

    model_dir = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    core = ov.Core()
    ov_model = core.read_model(model_dir / "openvino_model.xml")
    output = ov_model.output(0)
    input_names = {item.get_any_name(): item for item in ov_model.inputs}

    encoded = tokenizer(prompt, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    prompt_len = int(input_ids.shape[1])
    attention_mask = encoded["attention_mask"].astype(np.int64)

    result_shape = list(output.get_partial_shape())
    trace_len = result_shape[1].get_length() if len(result_shape) >= 2 and result_shape[1].is_static else prompt_len
    if prompt_len + args.max_new_tokens > trace_len:
        _die(
            f"Prompt length ({prompt_len}) + max_new_tokens ({args.max_new_tokens}) exceeds "
            f"the traced sequence length ({trace_len}). Re-run convert with a larger "
            "`--max-seq-len`, for example `--max-seq-len 512`."
        )

    print(f"Using OpenVINO inference device: {args.device}", file=sys.stderr)
    compiled = core.compile_model(ov_model, args.device)
    output = compiled.output(0)

    tokens = np.full((1, trace_len), tokenizer.pad_token_id or tokenizer.eos_token_id or 0, dtype=np.int64)
    mask = np.zeros((1, trace_len), dtype=np.int64)
    tokens[:, :prompt_len] = input_ids
    mask[:, :prompt_len] = attention_mask

    generated: list[int] = []
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()

    if not args.skip_prompt:
        print(prompt, end="" if args.stream else "\n")

    for position in range(prompt_len, min(trace_len, prompt_len + args.max_new_tokens)):
        inputs = {
            "input_ids": tokens,
            "attention_mask": mask,
        }
        if "beam_idx" in input_names:
            inputs["beam_idx"] = np.array([0], dtype=np.int32)
        logits = compiled(inputs)[output][0, position - 1]
        next_id = _sample_next_token(logits, args)
        if next_id in eos_ids:
            break
        tokens[0, position] = next_id
        mask[0, position] = 1
        generated.append(next_id)
        if args.stream:
            print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)

    if args.stream:
        print()
    else:
        print(tokenizer.decode(generated, skip_special_tokens=True))
    return 0


def generate(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args)
    ov_genai = _import_openvino_genai()

    model_source = args.model
    if not _looks_like_openvino_dir(model_source):
        _die(
            f"{model_source!r} does not look like an exported OpenVINO model. "
            "Run `plamo3-ov convert --output-dir ov-plamo3` first."
        )

    generation_kwargs = _sampling_kwargs(args)
    generation_kwargs["echo"] = not args.skip_prompt

    has_openvino_tokenizer = (Path(model_source) / "openvino_tokenizer.xml").exists()
    if not has_openvino_tokenizer:
        return _generate_with_openvino_core(args, prompt)

    try:
        print(f"Using OpenVINO GenAI inference device: {args.device}", file=sys.stderr)
        pipe = ov_genai.LLMPipeline(model_source, args.device)
    except RuntimeError as exc:
        _die(f"OpenVINO GenAI could not load the model directory: {exc}")

    if args.stream:
        def streamer(token: str) -> bool:
            print(token, end="", flush=True)
            return False

        try:
            pipe.generate(prompt, streamer=streamer, **generation_kwargs)
        except RuntimeError as exc:
            _die(
                "OpenVINO GenAI generation failed. If this model was produced by "
                "`plamo3-ov convert`, it may not have the stateful KV-cache LLM graph "
                f"that LLMPipeline expects. Original error: {exc}"
            )
        print()
        return 0

    try:
        print(pipe.generate(prompt, **generation_kwargs))
    except RuntimeError as exc:
        _die(
            "OpenVINO GenAI generation failed. If this model was produced by "
            "`plamo3-ov convert`, it may not have the stateful KV-cache LLM graph "
            f"that LLMPipeline expects. Original error: {exc}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plamo3-ov",
        description="Run pfnet/plamo-3-nict-8b-base with OpenVINO.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Export the Hugging Face model to OpenVINO IR.",
    )
    convert_parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    convert_parser.add_argument("--output-dir", required=True)
    convert_parser.add_argument(
        "--weight-format",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="OpenVINO save precision. int8 applies NNCF INT8_ASYM weight compression.",
    )
    convert_parser.add_argument("--example-prompt", help="Prompt used to trace the PyTorch model.")
    convert_parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Trace with this fixed sequence length. Needed for the HF-tokenizer fallback generator.",
    )
    convert_parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert the OpenVINO model even if openvino_model.xml already exists.",
    )
    convert_parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Required by PLaMo 3 custom model/tokenizer code.",
    )
    convert_parser.set_defaults(func=convert)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text from an exported OpenVINO model.",
    )
    generate_parser.add_argument("prompt", nargs="?")
    generate_parser.add_argument(
        "--model",
        default="ov-plamo3",
        help="Path to an OpenVINO GenAI model directory.",
    )
    generate_parser.add_argument("--prompt-file")
    generate_parser.add_argument("--stdin", action="store_true")
    generate_parser.add_argument(
        "--device",
        default="CPU",
        help=(
            "OpenVINO inference device, such as CPU, GPU, NPU, AUTO, GPU.0, "
            "or AUTO:GPU,CPU. Default: CPU."
        ),
    )
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--top-k", type=int, default=50)
    generate_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    generate_parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    generate_parser.add_argument("--skip-prompt", action=argparse.BooleanOptionalAction, default=True)
    generate_parser.set_defaults(func=generate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
