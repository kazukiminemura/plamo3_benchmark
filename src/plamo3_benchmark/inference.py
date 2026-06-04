from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .common import die, import_transformers, looks_like_openvino_dir, sampling_kwargs


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


def _import_openvino_genai() -> Any:
    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        die("openvino-genai is not installed. Run `uv sync` first.")
        raise exc
    return ov_genai


def _sample_next_token(logits: Any, args: Any) -> int:
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


def _print_generation_metrics(token_count: int, start_time: float, first_token_time: float | None) -> None:
    total_time = max(time.perf_counter() - start_time, 1e-9)
    fttt_text = "n/a" if first_token_time is None else f"{first_token_time - start_time:.3f}s"
    tokens_per_second = token_count / total_time if token_count else 0.0
    print(
        f"metrics: FTTT={fttt_text}, tokens={token_count}, total={total_time:.3f}s, "
        f"tokens/sec={tokens_per_second:.2f}",
        file=sys.stderr,
    )


class OpenVINOCoreGenerator:
    def __init__(self, args: Any) -> None:
        import openvino as ov

        AutoTokenizer, _, _ = import_transformers()

        self.args = args
        self.model_dir = Path(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        core = ov.Core()
        ov_model = core.read_model(self.model_dir / "openvino_model.xml")
        self.model_output = ov_model.output(0)
        self.input_names = {item.get_any_name(): item for item in ov_model.inputs}

        result_shape = list(self.model_output.get_partial_shape())
        self.trace_len = (
            result_shape[1].get_length() if len(result_shape) >= 2 and result_shape[1].is_static else None
        )

        print(f"Using OpenVINO inference device: {args.device}", file=sys.stderr)
        self.compiled = core.compile_model(ov_model, args.device)
        self.compiled_output = self.compiled.output(0)

    def generate(self, prompt: str, *, print_output: bool) -> str:
        import numpy as np

        encoded = self.tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        prompt_len = int(input_ids.shape[1])
        attention_mask = encoded["attention_mask"].astype(np.int64)
        trace_len = self.trace_len or prompt_len

        if prompt_len + self.args.max_new_tokens > trace_len:
            die(
                f"Prompt length ({prompt_len}) + max_new_tokens ({self.args.max_new_tokens}) exceeds "
                f"the traced sequence length ({trace_len}). Re-run convert with a larger "
                "`--max-seq-len`, for example `--max-seq-len 512`."
            )

        tokens = np.full(
            (1, trace_len),
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0,
            dtype=np.int64,
        )
        mask = np.zeros((1, trace_len), dtype=np.int64)
        tokens[:, :prompt_len] = input_ids
        mask[:, :prompt_len] = attention_mask

        generated: list[int] = []
        eos_ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        start_time = time.perf_counter()
        first_token_time: float | None = None

        if print_output and not self.args.skip_prompt:
            print(prompt, end="" if self.args.stream else "\n")

        for position in range(prompt_len, min(trace_len, prompt_len + self.args.max_new_tokens)):
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


class OpenVINOGenAIGenerator:
    def __init__(self, args: Any) -> None:
        self.args = args
        ov_genai = _import_openvino_genai()
        try:
            print(f"Using OpenVINO GenAI inference device: {args.device}", file=sys.stderr)
            self.pipe = ov_genai.LLMPipeline(args.model, args.device)
        except RuntimeError as exc:
            die(f"OpenVINO GenAI could not load the model directory: {exc}")

    def generate(self, prompt: str, *, print_output: bool) -> str:
        generation_kwargs = sampling_kwargs(self.args)
        generation_kwargs["echo"] = not self.args.skip_prompt
        start_time = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []

        def streamer(token: str) -> bool:
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if print_output and self.args.stream:
                print(token, end="", flush=True)
            chunks.append(token)
            return False

        try:
            self.pipe.generate(prompt, streamer=streamer, **generation_kwargs)
        except RuntimeError as exc:
            die(
                "OpenVINO GenAI generation failed. If this model was produced by "
                "`plamo3-ov convert`, it may not have the stateful KV-cache LLM graph "
                f"that LLMPipeline expects. Original error: {exc}"
            )
        text = "".join(chunks)
        if print_output and self.args.stream:
            print()
        elif print_output:
            print(text)
        _print_generation_metrics(len(chunks), start_time, first_token_time)
        return text


def load_generator(args: Any) -> Any:
    model_source = args.model
    if not looks_like_openvino_dir(model_source):
        die(f"{model_source!r} does not look like an exported OpenVINO model. Run `plamo3-ov convert --output-dir ov-plamo3` first.")

    has_openvino_tokenizer = (Path(model_source) / "openvino_tokenizer.xml").exists()
    if not has_openvino_tokenizer:
        return OpenVINOCoreGenerator(args)
    return OpenVINOGenAIGenerator(args)


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
