"""
The trained adapter as a model the ladder can run: merged into the base weights, saved in
Hugging Face's format, then converted to CTranslate2 -- the format faster-whisper loads, which is
what every ASR rung in s2s-lab uses (s2slab.r0.WhisperASR). The merged model answers exactly as
base + adapter does (tested), so the bench measures the adapter and nothing else.

    uv run src/export.py --adapter logs/train/runs/.../adapters/step-000900 --out ../data/ft/ft2-export
    -> <out>/hf   the merged transformers model + processor
       <out>/ct2  faster-whisper's model directory (float16 by default; pass --quantization int8 for a CPU)
"""

from __future__ import annotations

import argparse
import pathlib

from peft import PeftModel
from transformers import WhisperProcessor

from src.models.whisper_lora_module import build_base

CT2_COPY = ["tokenizer.json", "preprocessor_config.json"]


def merge(adapter_dir, out_dir, model_name: str = "openai/whisper-small.en",
          model_config_overrides: dict | None = None) -> pathlib.Path:
    """Base + adapter, fused: <out_dir>/hf."""
    out = pathlib.Path(out_dir) / "hf"
    model = PeftModel.from_pretrained(build_base(model_name, model_config_overrides), str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.feature_extractor.save_pretrained(str(out))     # preprocessor_config.json: the two files
    processor.tokenizer.save_pretrained(str(out))             # tokenizer.json: faster-whisper needs
    return out


def to_ctranslate2(hf_dir, out_dir, quantization: str = "float16") -> pathlib.Path:
    """<hf_dir> -> <out_dir>/ct2, the directory faster_whisper.WhisperModel(path) loads."""
    from ctranslate2.converters import TransformersConverter
    out = pathlib.Path(out_dir) / "ct2"
    TransformersConverter(str(hf_dir), copy_files=CT2_COPY).convert(str(out), quantization=quantization, force=True)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="the adapter directory (adapter_model.safetensors in it)")
    ap.add_argument("--out", required=True, help="where hf/ and ct2/ go")
    ap.add_argument("--model", default="openai/whisper-small.en")
    ap.add_argument("--quantization", default="float16", help="ct2: float16 (GPU), int8 (CPU), float32")
    ap.add_argument("--no-ct2", action="store_true", help="stop after the merged transformers model")
    args = ap.parse_args(argv)
    hf = merge(args.adapter, args.out, args.model)
    print(f"merged -> {hf}")
    if not args.no_ct2:
        print(f"ctranslate2 ({args.quantization}) -> {to_ctranslate2(hf, args.out, args.quantization)}")


if __name__ == "__main__":
    main()
