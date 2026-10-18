"""
M38's FastPitch renders (s2slab.leadsilence): NeMo FastPitch + HiFi-GAN, the LJSpeech base voice and FT-3's
fine-tuned 9017, neutral (the model's own durations and pitch), over R1's 200 first clauses. The models are
held to PR-012's SHA-256 pins (src.ft3.pr012.model_files) before anything loads. The padding is the duration
the model gave the space it puts before the first phoneme, in its frames.

    cd ft && PYTHONPATH=.:../src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.leadsilence_fp
    -> ../results/m38/fastpitch-ljspeech/ and ../results/m38/fastpitch-9017/, as s2slab.leadsilence.render saves
"""

from __future__ import annotations

import time

import numpy as np

from s2slab import leadsilence
from src.ft3 import pr012, synth

ENGINES = {"fastpitch-ljspeech": "tts_en_fastpitch_align", "fastpitch-9017": "fastpitch_9017"}   # engine -> PR-012's voice


def row(v: synth.Voice, text: str) -> dict:
    """One neutral render, the padding read off the first token's predicted duration when it is a pause."""
    pred = synth.predict(v, text)
    audio, _ = synth.render(v, pred, None)
    pad_ms = float(pred.durs[0] * v.hop_s * 1000) if len(pred.durs) and pred.is_pause[0] else 0.0
    tokens = list(v.fastpitch.vocab.tokens)
    ids = pred.tokens[0].cpu().numpy().tolist()
    return {"audio": np.asarray(audio, dtype=np.float32), "sr": v.sr, "pad_ms": pad_ms,
            "phonemes": "".join(tokens[i] if 0 <= i < len(tokens) else "?" for i in ids)}


def main() -> None:
    files = pr012.model_files({voice: pr012.VOICES[voice] for voice in ENGINES.values()})
    clauses = leadsilence.first_clauses(root=pr012.REPO)
    for engine, voice in ENGINES.items():
        v = synth.load(files[voice]["fastpitch"], files[voice]["hifigan"])
        rows = {}
        for clip, text in clauses.items():
            t0 = time.perf_counter()
            rows[clip] = dict(row(v, text), text=text, render_s=round(time.perf_counter() - t0, 3))
        folder = leadsilence.save_renders(engine, rows, pr012.REPO / leadsilence.OUT)
        print(f"{engine}: {len(rows)} renders -> {folder}")


if __name__ == "__main__":
    main()
