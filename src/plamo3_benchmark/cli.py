from __future__ import annotations

import argparse

from .common import DEFAULT_MODEL_ID, configure_output_encoding
from .inference import chat, generate
from .model_convert import convert


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="CPU",
        help=(
            "OpenVINO inference device, such as CPU, GPU, NPU, AUTO, GPU.0, "
            "or AUTO:GPU,CPU. Default: CPU."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-prompt", action=argparse.BooleanOptionalAction, default=True)


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
        choices=["fp32", "fp16", "int8", "int4"],
        help=(
            "OpenVINO save precision. int8 applies NNCF INT8_ASYM weight compression; "
            "int4 applies NNCF INT4_ASYM weight compression. With --target-device NPU, "
            "int8/int4 use symmetric compression."
        ),
    )
    convert_parser.add_argument("--example-prompt", help="Prompt used to trace the PyTorch model.")
    convert_parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Trace with this fixed sequence length. Needed for the HF-tokenizer fallback generator.",
    )
    convert_parser.add_argument(
        "--target-device",
        default="CPU",
        help=(
            "Device to optimize the exported IR for. Use NPU to keep static shapes and int32 "
            "token inputs. Default: CPU."
        ),
    )
    convert_parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert the OpenVINO model even if openvino_model.xml already exists.",
    )
    convert_parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the local Hugging Face cache or a local model directory.",
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
    _add_generation_options(generate_parser)
    generate_parser.set_defaults(func=generate)

    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat interactively with an exported OpenVINO model.",
    )
    chat_parser.add_argument(
        "--model",
        default="ov-plamo3",
        help="Path to an OpenVINO GenAI model directory.",
    )
    chat_parser.add_argument("--system", help="Optional system prompt prepended to the chat history.")
    chat_parser.add_argument("--max-turns", type=int, help="Exit after this many user turns.")
    _add_generation_options(chat_parser)
    chat_parser.set_defaults(func=chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
