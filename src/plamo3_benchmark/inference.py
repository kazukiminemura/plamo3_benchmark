from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .common import die, import_auto_tokenizer, looks_like_openvino_dir


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
    import numpy as np

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


def _print_generation_metrics(token_count: int, start_time: float, first_token_time: float | None) -> None:
    total_time = max(time.perf_counter() - start_time, 1e-9)
    fttt_text = "n/a" if first_token_time is None else f"{first_token_time - start_time:.3f}s"
    tokens_per_second = token_count / total_time if token_count else 0.0
    print(
        f"metrics: FTTT={fttt_text}, tokens={token_count}, total={total_time:.3f}s, "
        f"tokens/sec={tokens_per_second:.2f}",
        file=sys.stderr,
    )


def _resolve_fallback_max_new_tokens(prompt_len: int, requested_max_new_tokens: int, trace_len: int) -> int:
    available_new_tokens = trace_len - prompt_len
    if available_new_tokens <= 0:
        die(
            f"Prompt length ({prompt_len}) reaches the traced sequence length ({trace_len}). "
            "Use /reset to clear chat history, shorten the prompt, or re-run convert with a larger "
            f"`--max-seq-len`, for example `--max-seq-len {max(trace_len * 2, prompt_len + 128)}`."
        )

    if requested_max_new_tokens > available_new_tokens:
        print(
            f"warning: max_new_tokens ({requested_max_new_tokens}) exceeds the remaining traced "
            f"sequence length ({available_new_tokens}); generating at most {available_new_tokens} "
            "new tokens. Re-run convert with a larger "
            f"`--max-seq-len`, for example `--max-seq-len {prompt_len + requested_max_new_tokens}`.",
            file=sys.stderr,
        )
        return available_new_tokens

    return requested_max_new_tokens


class OpenVINOGenerator:
    def __init__(self, args: Any) -> None:
        import openvino as ov
        import numpy as np

        self.args = args
        self.model_dir = Path(args.model)
        self.tokenizer = import_auto_tokenizer().from_pretrained(self.model_dir, trust_remote_code=True, use_fast=False)
        core = ov.Core()
        ov_model = core.read_model(self.model_dir / "openvino_model.xml")
        self.model_output = ov_model.output(0)
        self.input_names = {item.get_any_name(): item for item in ov_model.inputs}
        self.input_dtypes = {
            name: self._numpy_dtype_for_ov_type(input_node.get_element_type(), np)
            for name, input_node in self.input_names.items()
        }

        result_shape = list(self.model_output.get_partial_shape())
        self.trace_len = (
            result_shape[1].get_length() if len(result_shape) >= 2 and result_shape[1].is_static else None
        )

        print(f"Using OpenVINO inference device: {args.device}", file=sys.stderr)
        self.compiled = core.compile_model(ov_model, args.device)
        self.compiled_output = self.compiled.output(0)
        self.outputs = {item.get_any_name(): item for item in self.compiled.outputs}
        self.info = self._read_info()

    def _read_info(self) -> dict[str, Any]:
        try:
            return json.loads((self.model_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _numpy_dtype_for_ov_type(ov_type: Any, np: Any) -> Any:
        type_name = str(ov_type)
        if type_name in {"i32", "<Type: 'int32_t'>"}:
            return np.int32
        if type_name in {"i64", "<Type: 'int64_t'>"}:
            return np.int64
        if type_name in {"f16", "<Type: 'float16'>"}:
            return np.float16
        if type_name in {"f32", "<Type: 'float32'>"}:
            return np.float32
        return np.int64

    def generate(self, prompt: str, *, print_output: bool) -> str:
        if self.info.get("uses_kv_cache"):
            return self._generate_with_kv_cache(prompt, print_output=print_output)
        return self._generate_full_context(prompt, print_output=print_output)

    def _generate_full_context(self, prompt: str, *, print_output: bool) -> str:
        import numpy as np

        encoded = self.tokenizer(prompt, return_tensors="np")
        input_dtype = self.input_dtypes.get("input_ids", np.int64)
        mask_dtype = self.input_dtypes.get("attention_mask", input_dtype)
        input_ids = encoded["input_ids"].astype(input_dtype)
        prompt_len = int(input_ids.shape[1])
        attention_mask = encoded["attention_mask"].astype(mask_dtype)
        trace_len = self.trace_len or prompt_len
        max_new_tokens = _resolve_fallback_max_new_tokens(prompt_len, self.args.max_new_tokens, trace_len)

        tokens = np.full(
            (1, trace_len),
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            dtype=input_dtype,
        )
        mask = np.zeros((1, trace_len), dtype=mask_dtype)
        tokens[:, :prompt_len] = input_ids
        mask[:, :prompt_len] = attention_mask

        generated: list[int] = []
        eos_ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        start_time = time.perf_counter()
        first_token_time: float | None = None

        if print_output and not self.args.skip_prompt:
            print(prompt, end="" if self.args.stream else "\n")

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

    def _empty_cache(self, np: Any) -> dict[str, Any]:
        try:
            layers = int(self.info["num_hidden_layers"])
            kv_heads = int(self.info["num_key_value_heads"])
            head_dim = int(self.info["head_dim"])
        except KeyError as exc:
            die(f"KV-cache model metadata is missing {exc.args[0]!r}; re-run convert with `--force`.")
        return {
            f"past.{layer_idx}.{kind}": np.zeros(
                (1, kv_heads, 0, head_dim),
                dtype=self.input_dtypes.get(f"past.{layer_idx}.{kind}", np.float32),
            )
            for layer_idx in range(layers)
            for kind in ("key", "value")
        }

    def _present_to_past(self, result: Any) -> dict[str, Any]:
        cache = {}
        for name, output in self.outputs.items():
            if name.startswith("present."):
                cache[name.replace("present.", "past.", 1)] = result[output]
        if not cache:
            die("KV-cache outputs were not found in the OpenVINO model; re-run convert with `--force`.")
        return cache

    def _generate_with_kv_cache(self, prompt: str, *, print_output: bool) -> str:
        import numpy as np

        encoded = self.tokenizer(prompt, return_tensors="np")
        input_dtype = self.input_dtypes.get("input_ids", np.int64)
        mask_dtype = self.input_dtypes.get("attention_mask", input_dtype)
        input_ids = encoded["input_ids"].astype(input_dtype)
        prompt_len = int(input_ids.shape[1])

        if print_output and not self.args.skip_prompt:
            print(prompt, end="" if self.args.stream else "\n")

        start_time = time.perf_counter()
        inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones((1, prompt_len), dtype=mask_dtype),
            **self._empty_cache(np),
        }
        result = self.compiled(inputs)
        logits = result[self.compiled_output][0, -1]
        cache = self._present_to_past(result)

        generated: list[int] = []
        eos_ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        first_token_time: float | None = None
        total_len = prompt_len

        for step in range(self.args.max_new_tokens):
            next_id = _sample_next_token(logits, self.args)
            if next_id in eos_ids:
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            generated.append(next_id)
            if self.args.stream and print_output:
                print(self.tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)

            if step == self.args.max_new_tokens - 1:
                break
            total_len += 1
            result = self.compiled(
                {
                    "input_ids": np.array([[next_id]], dtype=input_dtype),
                    "attention_mask": np.ones((1, total_len), dtype=mask_dtype),
                    **cache,
                }
            )
            logits = result[self.compiled_output][0, -1]
            cache = self._present_to_past(result)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if self.args.stream and print_output:
            print()
        elif print_output:
            print(text)
        _print_generation_metrics(len(generated), start_time, first_token_time)
        return text


def load_generator(args: Any) -> Any:
    model_source = args.model
    if not looks_like_openvino_dir(model_source):
        die(f"{model_source!r} does not look like an exported OpenVINO model. Run `plamo3-ov convert --output-dir ov-plamo3` first.")
    return OpenVINOGenerator(args)


def generate(args: Any) -> int:
    prompt = read_prompt(args)
    generator = load_generator(args)
    generator.generate(prompt, print_output=True)
    return 0


def format_chat_prompt(messages: list[dict[str, str]], system_prompt: str | None) -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt.strip()}")
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        parts.append(f"{role}: {message['content'].strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def chat(args: Any) -> int:
    if not looks_like_openvino_dir(args.model):
        die(f"{args.model!r} does not look like an exported OpenVINO model. Run `plamo3-ov convert --output-dir ov-plamo3` first.")

    generator = load_generator(args)
    print(
        "PLaMo 3 chat. Model is loaded once for this session. "
        "Type /exit or /quit to leave, /reset to clear history.",
        file=sys.stderr,
    )
    messages: list[dict[str, str]] = []
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
            messages.clear()
            turns = 0
            print("history reset", file=sys.stderr)
            continue

        messages.append({"role": "user", "content": user_text})
        prompt = format_chat_prompt(messages, args.system)
        print("assistant> ", end="", flush=True)
        assistant_text = generator.generate(prompt, print_output=True).strip()
        messages.append({"role": "assistant", "content": assistant_text})
        turns += 1

        if args.max_turns is not None and turns >= args.max_turns:
            return 0
