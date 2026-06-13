from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

from openvino_tokenizers import convert_tokenizer

from .common import is_local_model_path


def build_fast_unigram_tokenizer(tokenizer: Any) -> Any:
    """Rebuild Plamo3Tokenizer as a Hugging Face fast Unigram tokenizer."""
    import math

    from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    data = getattr(tokenizer, "data", None)
    if not data:
        raise ValueError("tokenizer does not expose a Unigram vocabulary (`data` attribute)")

    def quantize(value: Any) -> float:
        # Plamo3Tokenizer quantizes scores in its trie. Mirroring this keeps
        # Viterbi segmentation aligned with the custom slow tokenizer.
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
        # TemplateProcessing cannot be built directly because it parses the ":"
        # inside "<|plamo:bos|>" as a type_id separator, so inject serialized state.
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


def convert_tokenizer_to_ir(tokenizer: Any) -> tuple[Any, Any]:
    try:
        return convert_tokenizer(tokenizer, with_detokenizer=True)
    except Exception:
        # openvino_tokenizers does not understand PLaMo's custom slow tokenizer.
        # Its vocabulary is still a normal Unigram model, so rebuild a fast
        # tokenizer and convert that equivalent representation instead.
        fast_tokenizer = build_fast_unigram_tokenizer(tokenizer)
        return convert_tokenizer(fast_tokenizer, with_detokenizer=True, clean_up_tokenization_spaces=False)


def save_tokenizer_and_configs(ov: Any, tokenizer: Any, args: Any, output_dir: Path) -> None:
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
    patch_chat_template_for_openvino_tokenizer(output_dir)
    try:
        ov_tokenizer, ov_detokenizer = convert_tokenizer_to_ir(tokenizer)
        ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
    except Exception as exc:
        print(
            "warning: failed to convert tokenizer to OpenVINO IR; inference will use the "
            f"Hugging Face tokenizer fallback. Original error: {exc}",
            file=sys.stderr,
        )

    write_json_if_present(args.model, output_dir, "config.json", local_files_only=args.local_files_only)
    write_json_if_present(args.model, output_dir, "generation_config.json", local_files_only=args.local_files_only)
    if not (output_dir / "generation_config.json").exists():
        (output_dir / "generation_config.json").write_text('{"max_new_tokens": 128}\n', encoding="utf-8")


def patch_chat_template_for_openvino_tokenizer(output_dir: Path) -> None:
    template_path = output_dir / "chat_template.jinja"
    if not template_path.exists():
        return

    text = template_path.read_text(encoding="utf-8")
    patched = text.replace("{{- bos_token + '<|plamo:tag|>' -}}", "{{- '<|plamo:tag|>' -}}", 1)
    if patched != text:
        template_path.write_text(patched, encoding="utf-8")


def write_json_if_present(model: str, output_dir: Path, filename: str, *, local_files_only: bool) -> None:
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


def read_info(output_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((output_dir / "plamo3_ov_conversion.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_info(output_dir: Path, info: dict[str, Any]) -> None:
    (output_dir / "plamo3_ov_conversion.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def trace_len(ov: Any, xml_path: Path) -> int | None:
    try:
        shape = list(ov.Core().read_model(xml_path).outputs[0].get_partial_shape())
    except Exception:
        return None
    return shape[1].get_length() if len(shape) >= 2 and shape[1].is_static else None


def save_model_atomic(ov: Any, ov_model: Any, xml_path: Path, *, fp16: bool) -> None:
    tmp_xml = xml_path.with_name(f"{xml_path.stem}.tmp{xml_path.suffix}")
    ov.save_model(ov_model, tmp_xml, compress_to_fp16=fp16)
    del ov_model
    gc.collect()
    tmp_xml.with_suffix(".bin").replace(xml_path.with_suffix(".bin"))
    tmp_xml.replace(xml_path)
