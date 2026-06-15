from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from .common import die, looks_like_openvino_dir

try:
    import openvino_genai as ov_genai
except ImportError as exc:
    die("openvino-genai is not installed. Run `uv sync` first.")
    raise exc


DEFAULT_CHAT_SYSTEM = "You are a helpful AI assistant. Answer naturally and directly in the user's language."
CHAT_STOP_STRINGS = ("\nUser:", "\nSystem:", "\nAssistant:", "\nユーザー:", "\nシステム:", "\nアシスタント:")


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


def read_model_info(model_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((model_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_chat_prompt(messages: list[dict[str, str]], system_prompt: str | None) -> str:
    parts = [f"System: {(system_prompt or DEFAULT_CHAT_SYSTEM).strip()}"]
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        parts.append(f"{role}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def sample_next_token(logits: Any, args: Any) -> int:
    logits = logits.astype("float64")
    if args.temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / args.temperature
    if 0 < args.top_k < logits.shape[-1]:
        keep = np.argpartition(logits, -args.top_k)[-args.top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked

    probs = np.exp(logits - np.nanmax(logits))
    probs = probs / probs.sum()
    if args.top_p < 1.0:
        order = np.argsort(probs)[::-1]
        keep_count = max(1, int(np.searchsorted(np.cumsum(probs[order]), args.top_p, side="left") + 1))
        filtered = np.zeros_like(probs)
        filtered[order[:keep_count]] = probs[order[:keep_count]]
        probs = filtered / filtered.sum()
    return int(np.random.choice(np.arange(probs.shape[-1]), p=probs))


def cache_config(args: Any, model_dir: Path) -> dict[Any, str]:
    if not getattr(args, "model_cache", True):
        return {}
    cache_dir = Path(args.model_cache_dir) if getattr(args, "model_cache_dir", None) else default_cache_dir(args, model_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using OpenVINO model cache: {cache_dir}", file=sys.stderr)
    return {ov.properties.cache_dir: str(cache_dir)}


def default_cache_dir(args: Any, model_dir: Path) -> Path:
    safe_device = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(args.device)).strip("_")
    return model_dir / ".openvino_cache" / (safe_device or "device")


def static_dim(port: Any, index: int = 1) -> int | None:
    shape = list(port.get_partial_shape())
    return shape[index].get_length() if len(shape) > index and shape[index].is_static else None


def emit_metrics(
    model_load_seconds: float,
    output_tokens: int,
    started_at: float,
    first_token_at: float | None,
    finished_at: float,
) -> None:
    first = "n/a" if first_token_at is None else f"{first_token_at - started_at:.3f}s"
    decode_seconds = None if first_token_at is None else max(finished_at - first_token_at, 1e-9)
    tokens_per_second = 0.0 if decode_seconds is None else output_tokens / decode_seconds
    print(
        f"[metrics] model_load: {model_load_seconds:.3f}s | "
        f"time_to_first_token: {first} | "
        f"total_inference: {finished_at - started_at:.3f}s | "
        f"output_tokens: {output_tokens} | "
        f"tokens/sec: {tokens_per_second:.2f}",
        file=sys.stderr,
    )


class BaseGenerator:
    uses_native_chat = False

    def __init__(self, args: Any) -> None:
        self.args = args
        self.model_dir = Path(args.model)
        self.model_info = read_model_info(self.model_dir)
        self.in_chat = False
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        self.model_load_seconds = 0.0

    def emit_metrics(
        self,
        output_tokens: int,
        started_at: float,
        first_token_at: float | None,
        finished_at: float | None = None,
    ) -> None:
        emit_metrics(self.model_load_seconds, output_tokens, started_at, first_token_at, finished_at or time.perf_counter())

    def start_chat(self, system_message: str = "") -> None:
        return None

    def finish_chat(self) -> None:
        return None


class OpenVINOGenAIGenerator(BaseGenerator):
    def __init__(self, args: Any) -> None:
        started_at = time.perf_counter()
        super().__init__(args)
        model_id = str(self.model_info.get("model", "")).lower()
        self.uses_native_chat = not (model_id.endswith("-base") or "base" in model_id.rsplit("/", 1)[-1].split("-"))

        print(f"Using OpenVINO GenAI inference device: {args.device}", file=sys.stderr)
        self.pipe = ov_genai.LLMPipeline(str(args.model), args.device, cache_config(args, self.model_dir))
        self.model_load_seconds = time.perf_counter() - started_at

    def _decode(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        if hasattr(result, "texts"):
            return "".join(result.texts)
        tokens = getattr(result, "tokens", result)
        tokens = np.array(tokens.data if hasattr(tokens, "data") else tokens)
        if tokens.ndim > 1:
            tokens = tokens[0]
        return self.tokenizer.decode(tokens.astype(np.int64).tolist(), skip_special_tokens=True)

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return int(self.pipe.get_tokenizer().encode(text).input_ids.shape[-1])
        except Exception:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, *, print_output: bool, stop_strings: tuple[str, ...] = ()) -> str:
        first_token_at: float | None = None

        def streamer(_: str) -> bool:
            nonlocal first_token_at
            first_token_at = first_token_at or time.perf_counter()
            return False

        started_at = time.perf_counter()
        result = self.pipe.generate(
            prompt,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False,
            apply_chat_template=not self.in_chat and bool(getattr(self.args, "apply_chat_template", False)),
            stop_strings=set(stop_strings),
            include_stop_str_in_output=False,
            streamer=streamer,
        )
        finished_at = time.perf_counter()
        text = self._decode(result)
        if print_output:
            print(text)
        self.emit_metrics(self._count_tokens(text), started_at, first_token_at, finished_at)
        return text

    def start_chat(self, system_message: str = "") -> None:
        if self.uses_native_chat:
            self.pipe.start_chat(system_message)
            self.in_chat = True

    def finish_chat(self) -> None:
        if self.uses_native_chat and self.in_chat:
            self.pipe.finish_chat()
            self.in_chat = False


class DirectOpenVINOGenerator(BaseGenerator):
    def __init__(self, args: Any) -> None:
        started_at = time.perf_counter()
        super().__init__(args)

        core = ov.Core()
        config = cache_config(args, self.model_dir)
        if config:
            core.set_property(config)
        model = core.read_model(self.model_dir / "openvino_model.xml")

        self.inputs = {port.get_any_name(): port for port in model.inputs}
        self.input_dtype = np.int32 if self.model_info.get("input_dtype") == "int32" else np.int64
        self.stateful = "position_ids" in self.inputs and bool(getattr(model, "get_variables", list)())
        if not self.stateful:
            die("Direct OpenVINO generation requires stateful KV-cache IR. Re-run convert with `--force`.")
        self.input_len = static_dim(self.inputs["input_ids"])
        self.attention_len = static_dim(self.inputs["attention_mask"]) if "attention_mask" in self.inputs else None

        print(f"Using direct OpenVINO inference device: {args.device}", file=sys.stderr)
        print("Compiling OpenVINO model for direct inference...", file=sys.stderr)
        self.compiled = core.compile_model(model, args.device)
        print("Compiled OpenVINO model for direct inference.", file=sys.stderr)
        self.output = self.compiled.output(0)
        self.request = self.compiled.create_infer_request() if self.stateful else None
        self.model_load_seconds = time.perf_counter() - started_at

    def _infer_stateful(self, token_ids: list[int], position: int) -> Any:
        if self.input_len is not None and len(token_ids) != self.input_len:
            die(f"Stateful model expects {self.input_len} token(s) per inference, got {len(token_ids)}.")

        total_len = position + len(token_ids)
        attention_len = self.attention_len or total_len
        if total_len > attention_len:
            die(
                f"Sequence length ({total_len}) exceeds the traced KV-cache length ({attention_len}). "
                "Use /reset, shorten the prompt, or reconvert with a larger --max-seq-len."
            )

        mask = np.zeros((1, attention_len), dtype=self.input_dtype)
        mask[:, :total_len] = 1
        inputs = {
            "input_ids": np.array([token_ids], dtype=self.input_dtype),
            "attention_mask": mask,
            "position_ids": np.arange(position, total_len, dtype=self.input_dtype)[None],
        }
        if "beam_idx" in self.inputs:
            inputs["beam_idx"] = np.array([0], dtype=np.int32)
        self.request.infer(inputs)
        return self.request.get_output_tensor(0).data[0, -1]

    def _prefill_state(self, prompt_ids: list[int]) -> tuple[Any, int]:
        self.request.reset_state()
        if self.input_len == 1:
            logits = None
            for position, token_id in enumerate(prompt_ids):
                logits = self._infer_stateful([token_id], position)
            if logits is None:
                die("Prompt produced no tokens.")
            return logits, len(prompt_ids)
        return self._infer_stateful(prompt_ids, 0), len(prompt_ids)

    def _run_decode_loop(self, logits: Any, position: int, *, print_output: bool) -> tuple[str, int, float | None]:
        generated: list[int] = []
        eos_ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        first_token_at: float | None = None

        for _ in range(self.args.max_new_tokens):
            next_id = sample_next_token(logits, self.args)
            if next_id in eos_ids:
                break
            first_token_at = first_token_at or time.perf_counter()
            generated.append(next_id)
            if self.args.stream and print_output:
                print(self.tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)
            logits = self._infer_stateful([next_id], position)
            position += 1

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if self.args.stream and print_output:
            print()
        elif print_output:
            print(text)
        return text, len(generated), first_token_at

    def _generate_stateful(self, prompt: str, *, print_output: bool) -> str:
        started_at = time.perf_counter()
        prompt_ids = [int(token) for token in self.tokenizer(prompt, return_tensors="np")["input_ids"][0]]
        logits, position = self._prefill_state(prompt_ids)
        text, token_count, first_token_at = self._run_decode_loop(logits, position, print_output=print_output)
        self.emit_metrics(token_count, started_at, first_token_at)
        return text

    def generate(self, prompt: str, *, print_output: bool, stop_strings: tuple[str, ...] = ()) -> str:
        return self._generate_stateful(prompt, print_output=print_output)


def is_genai_compatible(model_dir: Path) -> bool:
    if not (model_dir / "openvino_tokenizer.xml").exists() or not (model_dir / "openvino_detokenizer.xml").exists():
        return False
    model = ov.Core().read_model(model_dir / "openvino_model.xml")
    inputs = {port.get_any_name() for port in model.inputs}
    return "beam_idx" in inputs and static_dim(model.output(0)) is None


def load_generator(args: Any) -> Any:
    if not looks_like_openvino_dir(args.model):
        die(f"{args.model!r} does not look like an exported OpenVINO model. Run `plamo3-ov convert --output-dir ov-plamo3` first.")

    model_dir = Path(args.model)
    if is_genai_compatible(model_dir):
        return OpenVINOGenAIGenerator(args)

    print(
        "warning: model is not fully compatible with OpenVINO GenAI chat; falling back to direct OpenVINO generation.",
        file=sys.stderr,
    )
    return DirectOpenVINOGenerator(args)


def generate(args: Any) -> int:
    load_generator(args).generate(read_prompt(args), print_output=True)
    return 0


def chat(args: Any) -> int:
    generator = load_generator(args)
    print(
        "PLaMo 3 chat. Model is loaded once for this session. Type /exit or /quit to leave, /reset to clear history.",
        file=sys.stderr,
    )
    if not generator.uses_native_chat:
        print("Using prompt-formatted chat for a base model.", file=sys.stderr)

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
                prompt = format_chat_prompt(messages, args.system)
                assistant_text = generator.generate(prompt, print_output=True, stop_strings=CHAT_STOP_STRINGS).strip()
                messages.append({"role": "assistant", "content": assistant_text})
            turns += 1
            if args.max_turns is not None and turns >= args.max_turns:
                return 0
    finally:
        generator.finish_chat()
