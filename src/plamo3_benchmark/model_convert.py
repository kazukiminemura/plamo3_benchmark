from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path
from typing import Any

from openvino_tokenizers import convert_tokenizer

from .common import check_hugging_face_access, die, import_auto_tokenizer, is_local_model_path


def _import_openvino() -> Any:
    try:
        import openvino as ov
    except ImportError as exc:
        die("openvino is not installed. Run `uv sync` first.")
        raise exc
    return ov


def _import_weight_compression() -> tuple[Any, Any, Any, Any]:
    try:
        from nncf import CompressWeightsMode, GroupSizeFallbackMode, compress_weights
        from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    except ImportError as exc:
        die("NNCF is required for int8/int4 weight compression. Run `uv sync` first.")
        raise exc
    return compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters


@contextmanager
def _patch_gqa_attention(torch: Any) -> Any:
    original = torch.nn.functional.scaled_dot_product_attention

    def patched(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("enable_gqa", None)
        if query.shape[-3] != key.shape[-3]:
            repeat = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)
        return original(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def _is_npu(device: str | None) -> bool:
    return bool(device and any(part.startswith("NPU") for part in device.upper().replace(":", ",").split(",")))


def _weight_format(args: Any) -> str:
    if _is_npu(args.target_device) and args.weight_format == "fp32":
        print("warning: NPU target requested with fp32; saving FP16 weights instead.", file=sys.stderr)
        return "fp16"
    return args.weight_format


def _compression_mode(weight_format: str, *, npu: bool) -> str | None:
    if weight_format == "int8":
        return "INT8_SYM" if npu else "INT8_ASYM"
    if weight_format == "int4":
        return "INT4_SYM" if npu else "INT4_ASYM"
    return None


def _compress_weights(ov_model: Any, weight_format: str, *, npu: bool) -> Any:
    mode_name = _compression_mode(weight_format, npu=npu)
    if mode_name is None:
        return ov_model

    compress_weights, CompressWeightsMode, GroupSizeFallbackMode, AdvancedCompressionParameters = (
        _import_weight_compression()
    )
    mode = getattr(CompressWeightsMode, mode_name)
    kwargs: dict[str, Any] = {}
    advanced_parameters = None
    if weight_format == "int4":
        advanced_parameters = AdvancedCompressionParameters(group_size_fallback_mode=GroupSizeFallbackMode.ADJUST)
        if npu:
            kwargs = {"ratio": 1.0, "group_size": -1}

    print(f"Compressing OpenVINO model weights to {mode_name} with NNCF...", file=sys.stderr)
    return compress_weights(ov_model, mode=mode, advanced_parameters=advanced_parameters, **kwargs)


def _build_fast_unigram_tokenizer(tokenizer: Any) -> Any:
    """Rebuild Plamo3Tokenizer as a Hugging Face fast Unigram tokenizer.

    openvino_tokenizers cannot convert the custom Plamo3Tokenizer class, but its
    vocabulary (`tokenizer.jsonl`) is a plain Unigram model with byte fallback, so an
    equivalent fast tokenizer converts cleanly. Known differences: the
    `break_around_repeated_chars_threshold` segmentation heuristic is not reproduced
    (the decoded text is still identical), and the detokenizer drops one leading space
    when the decoded sequence starts with two or more spaces.
    """
    import math

    from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    data = getattr(tokenizer, "data", None)
    if not data:
        raise ValueError("tokenizer does not expose a Unigram vocabulary (`data` attribute)")

    def quantize(value: Any) -> float:
        # Plamo3Tokenizer quantizes scores with round(score * 1e4) when building its
        # trie; mirroring that keeps Viterbi segmentation identical.
        score = float(value)
        return round(score * 1e4) / 1e4 if math.isfinite(score) else score

    vocab = [(str(row[0]), quantize(row[1])) for row in data]
    unk_ids = [idx for idx, row in enumerate(data) if len(row) > 2 and row[2] == "UNKNOWN"]
    fast_core = Tokenizer(Unigram(vocab, unk_id=unk_ids[0] if unk_ids else 0, byte_fallback=True))
    spaces_threshold = getattr(tokenizer, "break_around_consecutive_spaces_threshold", None)
    if spaces_threshold:
        fast_core.pre_tokenizer = pre_tokenizers.Split(
            Regex(f" {{{int(spaces_threshold)},}}"), behavior="isolated"
        )
    fast_core.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    fast_core.add_special_tokens(
        [
            AddedToken(str(row[0]), special=True, normalized=False)
            for row in data
            if len(row) > 2 and row[2] == "CONTROL"
        ]
    )
    bos_token = str(tokenizer.bos_token)
    if getattr(tokenizer, "add_bos_token", False) and tokenizer.bos_token_id is not None:
        # processors.TemplateProcessing cannot be built directly because it parses the
        # ":" inside "<|plamo:bos|>" as a type_id separator, so inject the processor
        # through the serialized tokenizer state instead.
        bos_piece = {"SpecialToken": {"id": bos_token, "type_id": 0}}
        state = json.loads(fast_core.to_str())
        state["post_processor"] = {
            "type": "TemplateProcessing",
            "single": [bos_piece, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [
                bos_piece,
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": bos_token, "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                bos_token: {"id": bos_token, "ids": [int(tokenizer.bos_token_id)], "tokens": [bos_token]},
            },
        }
        fast_core = Tokenizer.from_str(json.dumps(state))
    return PreTrainedTokenizerFast(
        tokenizer_object=fast_core,
        unk_token=str(tokenizer.unk_token),
        bos_token=bos_token,
        eos_token=str(tokenizer.eos_token),
        pad_token=str(tokenizer.pad_token),
        clean_up_tokenization_spaces=False,
    )


def _convert_tokenizer_to_ir(tokenizer: Any) -> tuple[Any, Any]:
    try:
        return convert_tokenizer(tokenizer, with_detokenizer=True)
    except Exception:
        fast_tokenizer = _build_fast_unigram_tokenizer(tokenizer)
        return convert_tokenizer(fast_tokenizer, with_detokenizer=True, clean_up_tokenization_spaces=False)


def _write_json_if_present(model: str, output_dir: Path, filename: str, *, local_files_only: bool) -> None:
    if is_local_model_path(model):
        source = Path(model) / filename
        if source.exists():
            (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return

    try:
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(repo_id=model, filename=filename, local_files_only=local_files_only))
    except Exception:
        return
    (output_dir / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _save_tokenizer_and_configs(ov: Any, tokenizer: Any, args: Any, output_dir: Path) -> None:
    for name in (
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
        "tokenizer.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    tokenizer.save_pretrained(output_dir)
    try:
        ov_tokenizer, ov_detokenizer = _convert_tokenizer_to_ir(tokenizer)
        ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
    except Exception as exc:
        print(
            "warning: failed to convert tokenizer to OpenVINO IR; inference will use the "
            f"Hugging Face tokenizer fallback. Original error: {exc}",
            file=sys.stderr,
        )
    _write_json_if_present(args.model, output_dir, "config.json", local_files_only=args.local_files_only)
    _write_json_if_present(args.model, output_dir, "generation_config.json", local_files_only=args.local_files_only)
    if not (output_dir / "generation_config.json").exists():
        (output_dir / "generation_config.json").write_text('{"max_new_tokens": 128}\n', encoding="utf-8")


def _read_info(output_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((output_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_info(output_dir: Path, info: dict[str, Any]) -> None:
    (output_dir / "plamo3_ov_conversion.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _trace_len(ov: Any, xml_path: Path) -> int | None:
    try:
        shape = list(ov.Core().read_model(xml_path).outputs[0].get_partial_shape())
    except Exception:
        return None
    return shape[1].get_length() if len(shape) >= 2 and shape[1].is_static else None


def _save_model(ov: Any, ov_model: Any, xml_path: Path, *, fp16: bool) -> None:
    tmp_xml = xml_path.with_name(f"{xml_path.stem}.tmp{xml_path.suffix}")
    ov.save_model(ov_model, tmp_xml, compress_to_fp16=fp16)
    del ov_model
    gc.collect()
    tmp_xml.with_suffix(".bin").replace(xml_path.with_suffix(".bin"))
    tmp_xml.replace(xml_path)


def convert(args: Any) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    check_hugging_face_access(args.model, local_files_only=args.local_files_only)

    npu = _is_npu(args.target_device)
    if npu and args.kv_cache:
        args.kv_cache = False
        print("NPU target requested; exporting a static full-context model without KV cache.", file=sys.stderr)
    if npu and args.max_seq_len is None:
        args.max_seq_len = 512
        print("NPU target requested; using --max-seq-len 512.", file=sys.stderr)

    ov = _import_openvino()
    tokenizer = _load_tokenizer(args)
    xml_path = output_dir / "openvino_model.xml"
    if xml_path.exists() and not args.force:
        return _update_existing(args, output_dir, xml_path, tokenizer, ov)
    return _convert_new(args, output_dir, tokenizer, ov)


def _load_tokenizer(args: Any) -> Any:
    try:
        return import_auto_tokenizer().from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=False,
        )
    except Exception as exc:
        die(f"Failed to load tokenizer for {args.model!r}: {exc}")


def _update_existing(args: Any, output_dir: Path, xml_path: Path, tokenizer: Any, ov: Any) -> int:
    print(f"Reusing existing OpenVINO model: {xml_path}", file=sys.stderr)
    info = _read_info(output_dir)
    weight_format = _weight_format(args)
    trace_len = _trace_len(ov, xml_path)

    uses_kv_cache = bool(info.get("uses_kv_cache", False))
    if args.kv_cache != uses_kv_cache:
        die("Existing model KV-cache setting differs from --kv-cache; re-run with `--force`.")
    if uses_kv_cache and info.get("kv_cache_format_version") != 4:
        die("Existing KV-cache model uses the old non-stateful layout; re-run convert with `--force`.")
    input_names = {item.get_any_name() for item in ov.Core().read_model(xml_path).inputs}
    if "beam_idx" not in input_names:
        die("Existing model is missing the GenAI `beam_idx` input; re-run convert with `--force`.")
    if not uses_kv_cache and args.max_seq_len is not None and trace_len is not None and int(args.max_seq_len) != trace_len:
        die(
            f"openvino_model.xml already exists with traced sequence length {trace_len}, "
            f"but --max-seq-len {args.max_seq_len} was requested. Re-run with `--force`."
        )
    if info.get("weight_format") and info.get("weight_format") != weight_format:
        die(f"Existing model is {info['weight_format']}; re-run with `--force` to save {weight_format}.")
    if _is_npu(args.target_device) and (info.get("static_shapes") is not True or info.get("input_dtype") != "int32"):
        die("Existing model is not NPU static-shape/int32 IR; re-run with `--force`.")

    _save_tokenizer_and_configs(ov, tokenizer, args, output_dir)
    _write_info(output_dir, _conversion_info(args, weight_format, trace_len, uses_kv_cache=uses_kv_cache))
    print(f"Updated existing OpenVINO model directory: {output_dir}", file=sys.stderr)
    return 0


def _convert_new(args: Any, output_dir: Path, tokenizer: Any, ov: Any) -> int:
    try:
        from torch import nn
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        die("transformers and torch are required for conversion. Run `uv sync` first.")
        raise exc

    class LogitsOnly(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            beam_idx: torch.Tensor,
        ) -> torch.Tensor:
            logits = self.model(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask.to(torch.long),
                use_cache=False,
                return_dict=False,
            )[0]
            return logits + beam_idx.to(logits.dtype).reshape(-1, 1, 1) * 0

    class StatefulKV(nn.Module):
        """PLaMo3 decoder with flat KV-cache inputs/outputs and GenAI-style inputs.

        Inputs: input_ids [1, seq], attention_mask [1, past+seq] (consumed shape-free;
        batch-1 generation has no padding), position_ids [1, seq], beam_idx [1] and
        past.* [1, kv_heads, past_len, head_dim]. Attention masks are built purely from
        position tensors so torch.export does not create guards coupling the dynamic
        dimensions, and the sliding window is enforced by masking instead of clipping
        the cache (the cache grows to max_position_embeddings at most).
        """

        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model
            module = __import__(model.__class__.__module__, fromlist=["_rotary_pos_emb"])
            self.rotary_pos_emb = module._rotary_pos_emb
            self.window = int(model.config.window_size)

        def attention(
            self,
            mixer: nn.Module,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            past_key: torch.Tensor,
            past_value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            import torch.nn.functional as F

            bsz, q_len, _ = hidden_states.size()
            qkv = mixer.qkv_proj(hidden_states)
            query_states, key_states, value_states = torch.split(
                qkv, [mixer.q_proj_dim, mixer.k_proj_dim, mixer.v_proj_dim], dim=-1
            )
            query_states = query_states.view(bsz, q_len, mixer.q_num_heads, mixer.qk_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, mixer.k_num_heads, mixer.qk_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, mixer.v_num_heads, mixer.v_dim).transpose(1, 2)

            attn_dtype = query_states.dtype
            query_states = mixer.q_norm(query_states)
            key_states = mixer.k_norm(key_states)
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

            kv_len = key_states.shape[-2]
            max_pos = mixer.config.max_position_embeddings
            q_pos = position_ids.to(torch.long)  # [1, q_len]
            k_pos = (
                torch.arange(kv_len, dtype=torch.long, device=hidden_states.device)
                + q_pos[:, -1:]
                - kv_len
                + 1
            )  # [1, kv_len]

            cos, sin = mixer.rotary_emb(value_states, seq_len=max_pos)
            query_states = self.rotary_pos_emb(query_states, cos, sin, q_pos.clamp(0, max_pos - 1)).to(attn_dtype)
            rotated_key_states = self.rotary_pos_emb(key_states, cos, sin, k_pos.clamp(0, max_pos - 1)).to(attn_dtype)
            value_states = value_states.to(attn_dtype)

            visible = k_pos[:, None, :] <= q_pos[:, :, None]  # causal, [1, q_len, kv_len]
            if not mixer.full_attn:
                visible = visible & (k_pos[:, None, :] > q_pos[:, :, None] - self.window)
            attn_mask = torch.where(visible, 0.0, float("-inf")).to(attn_dtype)[:, None]

            attn_output = F.scaled_dot_product_attention(
                query_states,
                rotated_key_states,
                value_states,
                attn_mask=attn_mask,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, mixer.q_num_heads * mixer.v_dim)
            attn_output = mixer.o_proj(attn_output)
            return attn_output, key_states, value_states

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            beam_idx: torch.Tensor,
            *past: torch.Tensor,
        ) -> tuple[Any, ...]:
            base = self.model.model
            inputs_embeds = base.embed_tokens(input_ids.to(torch.long))
            hidden_states = inputs_embeds
            beam = beam_idx.to(torch.long)
            flat_cache = []
            for layer_idx, layer in enumerate(base.layers.layers):
                past_key = past[layer_idx * 2].index_select(0, beam)
                past_value = past[layer_idx * 2 + 1].index_select(0, beam)
                residual = hidden_states
                hidden_states = layer.pre_mixer_norm(hidden_states)
                attn_out, key, value = self.attention(
                    layer.mixer, hidden_states, position_ids, past_key, past_value
                )
                hidden_states = residual + layer.post_mixer_norm(attn_out)
                residual = hidden_states
                hidden_states = residual + layer.post_mlp_norm(layer.mlp(layer.pre_mlp_norm(hidden_states)))
                flat_cache.extend([key, value])

            # GenAI reads logits as f32 regardless of the traced precision.
            logits = self.model.lm_head(base.norm(hidden_states)).to(torch.float32)
            # Consume attention_mask without coupling its length to other dims.
            logits = logits + attention_mask.to(logits.dtype).sum() * 0
            return (logits, *flat_cache)

    weight_format = _weight_format(args)
    npu = _is_npu(args.target_device)
    dtype = torch.float16 if weight_format in {"fp16", "int8", "int4"} else torch.float32

    print("Loading Hugging Face model with trust_remote_code=True...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    model.config.use_cache = False

    print("Converting PyTorch model with openvino.convert_model...", file=sys.stderr)
    if args.kv_cache:
        trace_len = None
        layers = int(getattr(model.config, "num_hidden_layers"))
        kv_heads = int(getattr(model.config, "num_key_value_heads"))
        head_dim = int(getattr(model.config, "head_dim"))
        seq_example, past_example = 8, 16
        example_input = (
            torch.ones((1, seq_example), dtype=torch.int64),
            torch.ones((1, past_example + seq_example), dtype=torch.int64),
            torch.arange(past_example, past_example + seq_example, dtype=torch.int64)[None],
            torch.tensor([0], dtype=torch.int32),
            *(torch.zeros((1, kv_heads, past_example, head_dim), dtype=dtype) for _ in range(layers * 2)),
        )
        dynamic_shapes = (
            {1: torch.export.Dim.DYNAMIC},
            {1: torch.export.Dim.DYNAMIC},
            {1: torch.export.Dim.DYNAMIC},
            {0: torch.export.Dim.STATIC},
            tuple({2: torch.export.Dim.DYNAMIC} for _ in range(layers * 2)),
        )
        with _patch_gqa_attention(torch):
            exported = torch.export.export(
                StatefulKV(model).eval(), example_input, dynamic_shapes=dynamic_shapes, strict=False
            )
            ov_model = ov.convert_model(exported)
        del exported

        for node, name in zip(ov_model.inputs[:4], ("input_ids", "attention_mask", "position_ids", "beam_idx")):
            node.get_tensor().set_names({name})
        ov_model.outputs[0].get_tensor().set_names({"logits"})
        state_pairs = {}
        for idx, node in enumerate(ov_model.inputs[4:]):
            node.get_tensor().set_names({f"past.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"})
        for idx, node in enumerate(ov_model.outputs[1:]):
            node.get_tensor().set_names({f"present.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"})
            state_pairs[f"past.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"] = (
                f"present.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"
            )

        # Pin head count / head size of the cache inputs so the initial (empty) state
        # created by ReadValue has shape [1, kv_heads, 0, head_dim] and concatenates
        # cleanly with the new keys/values.
        ov_model.reshape(
            {
                node.get_any_name(): ov.PartialShape([1, kv_heads, -1, head_dim])
                for node in ov_model.inputs
                if node.get_any_name().startswith("past.")
            }
        )

        from openvino._offline_transformations import apply_make_stateful_transformation

        apply_make_stateful_transformation(ov_model, state_pairs)
    else:
        if args.max_seq_len is None:
            args.max_seq_len = 512
            print("Full-context export requested; using --max-seq-len 512.", file=sys.stderr)
        token_dtype = torch.int32 if npu else torch.long
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        example_input = (
            torch.full((1, args.max_seq_len), pad_id, dtype=token_dtype),
            torch.ones((1, args.max_seq_len), dtype=token_dtype),
            torch.tensor([0], dtype=torch.int32),
        )
        trace_len = int(args.max_seq_len)
        with _patch_gqa_attention(torch):
            ov_model = ov.convert_model(
                LogitsOnly(model).eval(),
                example_input=example_input,
                dynamo=True,
            )

        ov_model.inputs[0].get_tensor().set_names({"input_ids"})
        ov_model.inputs[1].get_tensor().set_names({"attention_mask"})
        ov_model.inputs[2].get_tensor().set_names({"beam_idx"})
        ov_model.outputs[0].get_tensor().set_names({"logits"})

        shapes = {
            "input_ids": ov.PartialShape([1, trace_len] if npu else [-1, -1]),
            "attention_mask": ov.PartialShape([1, trace_len] if npu else [-1, -1]),
            "beam_idx": ov.PartialShape([1] if npu else [-1]),
        }
        try:
            ov_model.reshape(shapes)
        except Exception:
            if npu:
                raise
            print("warning: converted model kept the traced prompt shape.", file=sys.stderr)

    del model
    gc.collect()
    ov_model = _compress_weights(ov_model, weight_format, npu=npu)
    _save_model(ov, ov_model, output_dir / "openvino_model.xml", fp16=weight_format == "fp16")
    _save_tokenizer_and_configs(ov, tokenizer, args, output_dir)
    _write_info(output_dir, _conversion_info(args, weight_format, trace_len, uses_kv_cache=bool(args.kv_cache)))
    print(f"Saved OpenVINO model directory to: {output_dir}", file=sys.stderr)
    return 0


def _conversion_info(args: Any, weight_format: str, trace_len: int | None, *, uses_kv_cache: bool) -> dict[str, Any]:
    npu = _is_npu(args.target_device)
    info = {
        "model": args.model,
        "weight_format": weight_format,
        "target_device": args.target_device,
        "compression_mode": _compression_mode(weight_format, npu=npu),
        "trace_sequence_length": trace_len,
        "static_shapes": npu,
        "input_dtype": "int32" if npu else "int64",
        "uses_kv_cache": uses_kv_cache,
    }
    if uses_kv_cache:
        info["kv_cache_format_version"] = 4
        info["stateful"] = True
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
