from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from torch import nn

from .quantization import is_npu


@contextmanager
def patch_gqa_attention(torch: Any) -> Any:
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


def export_openvino_model(args: Any, tokenizer: Any, model: Any, ov: Any, torch: Any, dtype: Any) -> tuple[Any, int | None]:
    if is_npu(args.target_device):
        return _export_npu_model(args, tokenizer, model, ov, torch)
    return _export_stateful_model(model, ov, torch, dtype)


def _export_npu_model(args: Any, tokenizer: Any, model: Any, ov: Any, torch: Any) -> tuple[Any, int]:
    trace_len = int(args.max_seq_len)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    example_input = (
        torch.full((1, trace_len), pad_id, dtype=torch.int32),
        torch.ones((1, trace_len), dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )
    with patch_gqa_attention(torch):
        ov_model = ov.convert_model(LogitsOnly(model).eval(), example_input=example_input, dynamo=True)

    for node, name in zip(ov_model.inputs, ("input_ids", "attention_mask", "beam_idx")):
        node.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"logits"})
    ov_model.reshape(
        {
            "input_ids": ov.PartialShape([1, trace_len]),
            "attention_mask": ov.PartialShape([1, trace_len]),
            "beam_idx": ov.PartialShape([1]),
        }
    )
    return ov_model, trace_len


def _export_stateful_model(model: Any, ov: Any, torch: Any, dtype: Any) -> tuple[Any, None]:
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
    with patch_gqa_attention(torch):
        exported = torch.export.export(
            StatefulKV(model).eval(), example_input, dynamic_shapes=dynamic_shapes, strict=False
        )
        ov_model = ov.convert_model(exported)
    del exported

    for node, name in zip(ov_model.inputs[:4], ("input_ids", "attention_mask", "position_ids", "beam_idx")):
        node.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"logits"})

    def cache_name(prefix: str, idx: int) -> str:
        return f"{prefix}.{idx // 2}.{'key' if idx % 2 == 0 else 'value'}"

    state_pairs = {}
    for idx, node in enumerate(ov_model.inputs[4:]):
        node.get_tensor().set_names({cache_name("past", idx)})
    for idx, node in enumerate(ov_model.outputs[1:]):
        node.get_tensor().set_names({cache_name("present", idx)})
        state_pairs[cache_name("past", idx)] = cache_name("present", idx)

    ov_model.reshape(
        {
            node.get_any_name(): ov.PartialShape([1, kv_heads, -1, head_dim])
            for node in ov_model.inputs
            if node.get_any_name().startswith("past.")
        }
    )

    from openvino._offline_transformations import apply_make_stateful_transformation

    apply_make_stateful_transformation(ov_model, state_pairs)
    return ov_model, None


class LogitsOnly(nn.Module):
    """Fixed-shape export used for NPU.

    NPU currently needs static token shapes and int32 inputs, so we export a
    full-context model that recomputes the prompt window instead of a dynamic
    stateful KV-cache model.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Any, attention_mask: Any, beam_idx: Any) -> Any:
        import torch

        logits = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            use_cache=False,
            return_dict=False,
        )[0]
        return logits + beam_idx.to(logits.dtype).reshape(-1, 1, 1) * 0


class StatefulKV(nn.Module):
    """PLaMo3 decoder with explicit KV-cache tensors before OpenVINO state conversion.

    A normal Hugging Face `use_cache=True` trace exposes nested Python cache
    objects and shape guards that OpenVINO GenAI cannot consume directly. This
    wrapper flattens `past.*` / `present.*`, adds GenAI's `beam_idx`, then
    `apply_make_stateful_transformation` turns those tensors into internal
    `ReadValue` / `Assign` state.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        module = __import__(model.__class__.__module__, fromlist=["_rotary_pos_emb"])
        self.rotary_pos_emb = module._rotary_pos_emb
        self.window = int(model.config.window_size)

    def attention(
        self,
        mixer: Any,
        hidden_states: Any,
        position_ids: Any,
        past_key: Any,
        past_value: Any,
    ) -> tuple[Any, Any, Any]:
        import torch
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
        q_pos = position_ids.to(torch.long)
        k_pos = torch.arange(kv_len, dtype=torch.long, device=hidden_states.device) + q_pos[:, -1:] - kv_len + 1

        cos, sin = mixer.rotary_emb(value_states, seq_len=max_pos)
        query_states = self.rotary_pos_emb(query_states, cos, sin, q_pos.clamp(0, max_pos - 1)).to(attn_dtype)
        rotated_key_states = self.rotary_pos_emb(key_states, cos, sin, k_pos.clamp(0, max_pos - 1)).to(attn_dtype)
        value_states = value_states.to(attn_dtype)

        # The original model uses sliding-window attention. We keep the cache
        # append-only and express the window as a mask so the exported cache
        # shape stays monotonic and easier for OpenVINO state conversion.
        visible = k_pos[:, None, :] <= q_pos[:, :, None]
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
        return mixer.o_proj(attn_output), key_states, value_states

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        beam_idx: Any,
        *past: Any,
    ) -> tuple[Any, ...]:
        import torch

        base = self.model.model
        hidden_states = base.embed_tokens(input_ids.to(torch.long))
        beam = beam_idx.to(torch.long)
        flat_cache = []
        for layer_idx, layer in enumerate(base.layers.layers):
            past_key = past[layer_idx * 2].index_select(0, beam)
            past_value = past[layer_idx * 2 + 1].index_select(0, beam)
            residual = hidden_states
            hidden_states = layer.pre_mixer_norm(hidden_states)
            attn_out, key, value = self.attention(layer.mixer, hidden_states, position_ids, past_key, past_value)
            hidden_states = residual + layer.post_mixer_norm(attn_out)
            residual = hidden_states
            hidden_states = residual + layer.post_mlp_norm(layer.mlp(layer.pre_mlp_norm(hidden_states)))
            flat_cache.extend([key, value])

        logits = self.model.lm_head(base.norm(hidden_states)).to(torch.float32)
        logits = logits + attention_mask.to(logits.dtype).sum() * 0
        return (logits, *flat_cache)
