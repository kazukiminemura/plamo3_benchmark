from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from .common import die, looks_like_openvino_dir
from transformers import AutoTokenizer

try:
    import openvino_genai as ov_genai
except ImportError as exc:
    die("openvino-genai is not installed. Run `uv sync` first.")
    raise exc


def _generation_config(args: Any) -> Any:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    return config


def read_prompt(args: Any) -> str:
    sources = [args.prompt is not None, args.prompt_file is not None, args.stdin]
    if sum(sources) > 1:
        die("choose only one prompt source: positional prompt, --prompt-file, or --stdin")

    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    if args.prompt is not None:
        return args.prompt

    die("provide a prompt, --prompt-file, or --stdin")
    return ""


def _sample_next_token(logits: Any, args: Any) -> int:
    logits = logits.astype("float64")
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


def _format_chat_prompt(messages: list[dict[str, str]], system_prompt: str | None) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt.strip()}")
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        parts.append(f"{role}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _print_generation_metrics(token_count: int, start_time: float, first_token_time: float | None) -> None:
    total_time = max(time.perf_counter() - start_time, 1e-9)
    fttt_text = "n/a" if first_token_time is None else f"{first_token_time - start_time:.3f}s"
    tokens_per_second = token_count / total_time if token_count else 0.0
    print(
        f"metrics: FTTT={fttt_text}, tokens={token_count}, total={total_time:.3f}s, "
        f"tokens/sec={tokens_per_second:.2f}",
        file=sys.stderr,
    )


class OpenVINOGenAIGenerator:
    uses_native_chat = True

    def __init__(self, args: Any) -> None:
        self.args = args
        self.config = _generation_config(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        self._check_genai_model_inputs(Path(args.model))
        print(f"Using OpenVINO GenAI inference device: {args.device}", file=sys.stderr)
        self.pipe = ov_genai.LLMPipeline(str(args.model), args.device)
        self.pipe.set_generation_config(self.config)

    @staticmethod
    def _check_genai_model_inputs(model_dir: Path) -> None:
        input_names = {item.get_any_name() for item in ov.Core().read_model(model_dir / "openvino_model.xml").inputs}
        if "beam_idx" not in input_names:
            try:
                info = json.loads((model_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
            except Exception:
                info = {}
            command = (
                f"plamo3-ov convert --output-dir {model_dir} "
                f"--weight-format {info.get('weight_format', 'fp16')} "
                f"--target-device {info.get('target_device', 'CPU')} --force"
            )
            die(
                "This OpenVINO model was exported before GenAI support and is missing `beam_idx`. "
                f"Re-run `{command}`."
            )

    def _encode(self, prompt: str) -> tuple[Any, int]:
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        return (
            ov_genai.TokenizedInputs(
                ov.Tensor(input_ids),
                ov.Tensor(attention_mask),
            ),
            int(input_ids.shape[1]),
        )

    def _decode_result(self, result: Any, prompt_len: int) -> str:
        if hasattr(result, "texts"):
            return "".join(result.texts)

        tokens = getattr(result, "tokens", result)
        token_array = np.array(tokens.data if hasattr(tokens, "data") else tokens)
        if token_array.ndim > 1:
            token_array = token_array[0]

        token_ids = token_array.astype(np.int64).tolist()
        if len(token_ids) > prompt_len:
            token_ids = token_ids[prompt_len:]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(self, prompt: str, *, print_output: bool) -> str:
        inputs, prompt_len = self._encode(prompt)
        if self.args.stream and print_output:
            chunks: list[str] = []

            def streamer(token: Any) -> bool:
                if isinstance(token, str):
                    text = token
                else:
                    text = self.tokenizer.decode([int(token)], skip_special_tokens=True)
                chunks.append(text)
                print(text, end="", flush=True)
                return False

            result = self.pipe.generate(inputs, self.config, streamer)
            print()
            return "".join(chunks) if chunks else self._decode_result(result, prompt_len)

        text = self._decode_result(self.pipe.generate(inputs, self.config), prompt_len)
        if print_output:
            print(text)
        return text

    def start_chat(self, system_message: str = "") -> None:
        self.pipe.start_chat(system_message)

    def finish_chat(self) -> None:
        self.pipe.finish_chat()


class DirectOpenVINOGenerator:
    uses_native_chat = False

    def __init__(self, args: Any) -> None:
        self.args = args
        self.model_dir = Path(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True, use_fast=False)
        core = ov.Core()
        model = core.read_model(self.model_dir / "openvino_model.xml")
        self.input_names = {item.get_any_name(): item for item in model.inputs}
        self.output = model.output(0)
        output_shape = list(model.output(0).get_partial_shape())
        self.trace_len = output_shape[1].get_length() if len(output_shape) > 1 and output_shape[1].is_static else None
        if self.trace_len is None:
            input_shape = list(self.input_names["input_ids"].get_partial_shape())
            self.trace_len = input_shape[1].get_length() if len(input_shape) > 1 and input_shape[1].is_static else None
        if self.trace_len is None:
            self.trace_len = int(self._read_info().get("trace_sequence_length") or 512)

        print(f"Using direct OpenVINO inference device: {args.device}", file=sys.stderr)
        self.compiled = core.compile_model(model, args.device)
        self.compiled_output = self.compiled.output(0)

    def _read_info(self) -> dict[str, Any]:
        try:
            return json.loads((self.model_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    def generate(self, prompt: str, *, print_output: bool) -> str:
        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        prompt_len = int(input_ids.shape[1])
        if prompt_len >= self.trace_len:
            die(
                f"Prompt length ({prompt_len}) reaches the traced sequence length ({self.trace_len}). "
                "Use /reset, shorten the prompt, or reconvert with a larger --max-seq-len."
            )

        max_new_tokens = min(self.args.max_new_tokens, self.trace_len - prompt_len)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        tokens = np.full((1, self.trace_len), pad_id, dtype=np.int64)
        mask = np.zeros((1, self.trace_len), dtype=np.int64)
        tokens[:, :prompt_len] = input_ids
        mask[:, :prompt_len] = attention_mask

        generated: list[int] = []
        eos_ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        start_time = time.perf_counter()
        first_token_time: float | None = None

        for position in range(prompt_len, prompt_len + max_new_tokens):
            inputs = {"input_ids": tokens, "attention_mask": mask}
            if "beam_idx" in self.input_names:
                inputs["beam_idx"] = np.array([0], dtype=np.int32)
            logits = self.compiled(inputs)[self.compiled_output][0, position - 1]
            next_id = _sample_next_token(logits, self.args)
            if next_id in eos_ids:
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens[0, position] = next_id
            mask[0, position] = 1
            generated.append(next_id)
            if self.args.stream and print_output:
                print(self.tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if self.args.stream and print_output:
            print()
        elif print_output:
            print(text)
        _print_generation_metrics(len(generated), start_time, first_token_time)
        return text

    def start_chat(self, system_message: str = "") -> None:
        return None

    def finish_chat(self) -> None:
        return None


def _is_genai_compatible(model_dir: Path) -> bool:
    tokenizer_exists = (model_dir / "openvino_tokenizer.xml").exists() and (model_dir / "openvino_detokenizer.xml").exists()
    if not tokenizer_exists:
        return False
    model = ov.Core().read_model(model_dir / "openvino_model.xml")
    input_names = {item.get_any_name() for item in model.inputs}
    if "beam_idx" not in input_names:
        return False
    output_shape = list(model.output(0).get_partial_shape())
    return len(output_shape) < 2 or output_shape[1].is_dynamic


def load_generator(args: Any) -> OpenVINOGenAIGenerator:
    model_source = args.model
    if not looks_like_openvino_dir(model_source):
        die(
            f"{model_source!r} does not look like an exported OpenVINO model. "
            "Run `plamo3-ov convert --output-dir ov-plamo3` first."
        )
    model_dir = Path(model_source)
    if _is_genai_compatible(model_dir):
        return OpenVINOGenAIGenerator(args)
    print(
        "warning: model is not fully compatible with OpenVINO GenAI chat; "
        "falling back to direct OpenVINO generation.",
        file=sys.stderr,
    )
    return DirectOpenVINOGenerator(args)


def generate(args: Any) -> int:
    prompt = read_prompt(args)
    generator = load_generator(args)
    generator.generate(prompt, print_output=True)
    return 0


def chat(args: Any) -> int:
    generator = load_generator(args)
    print(
        "PLaMo 3 chat. Model is loaded once for this session. "
        "Type /exit or /quit to leave, /reset to clear history.",
        file=sys.stderr,
    )
    messages: list[dict[str, str]] = []
    generator.start_chat(args.system or "")
    try:
        turns = 0
        while True:
            try:
                user_text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                return 0
            if user_text == "/reset":
                generator.finish_chat()
                messages.clear()
                generator.start_chat(args.system or "")
                turns = 0
                print("history reset", file=sys.stderr)
                continue

            print("assistant> ", end="", flush=True)
            if generator.uses_native_chat:
                assistant_text = generator.generate(user_text, print_output=True).strip()
            else:
                messages.append({"role": "user", "content": user_text})
                prompt = _format_chat_prompt(messages, args.system)
                assistant_text = generator.generate(prompt, print_output=True).strip()
                messages.append({"role": "assistant", "content": assistant_text})
            turns += 1

            if args.max_turns is not None and turns >= args.max_turns:
                return 0
    finally:
        generator.finish_chat()
