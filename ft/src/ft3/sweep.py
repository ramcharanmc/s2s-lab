"""
FT-3's sweep: every sentence rendered at every score of each condition through FastPitch's own
inputs, measured with severity's measures, in severity.run's exact results shape -- so
severity.report prints it and PR-009's severity.criteria scores it, unchanged.

    PYTHONPATH=ft:src python -m src.ft3.sweep --sentences dev [--listen]
        -> results/ft3/sweep-<voice>-dev.json and the two tables PR-009 had
    PYTHONPATH=ft:src python -m src.ft3.sweep --sentences dev --fastpitch path/to/finetuned.nemo --hifigan path/to/hifigan.nemo

A FastPitch checkpoint is one voice, so `voice` in the rows is the checkpoint's name. The
neutral clip of each sentence is the model's own prediction through the same forward pass; its
SHA-256 is recorded as severity.run records Kokoro's, and with --listen every render is written
as a WAV named and tagged synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from s2slab import severity
from src.ft3 import control, synth

SENTENCE_SETS = {"dev": severity.DEV_SENTENCES}      # PR-012's eight sentences are pr012.py's, behind its lock


def run(v: synth.Voice, sentences: dict, scores: dict = severity.SCORES, listen: pathlib.Path | None = None,
        centre_mode: str = synth.CENTRE_MODE, range_gain: float = control.RANGE_GAIN) -> dict:
    """severity.run for FastPitch: {"sr", "tolerance", "rows"}, one row per sentence x condition,
    plus `voice_info` (the checkpoint's pitch statistics and hop)."""
    rows = []
    for sid, text in sentences.items():
        pred = synth.predict(v, text)
        neutral, _ = synth.render(v, pred, None)
        centre = synth.centre_of(v, neutral, pred, centre_mode)
        digest = hashlib.sha256(np.ascontiguousarray(neutral, dtype=np.float32).tobytes()).hexdigest()
        if listen is not None:
            severity.write_synthetic(listen / severity.synthetic_name(v.name, sid), neutral, v.sr, v.name, text, subtype="FLOAT")
        for condition, levels in scores.items():
            steps = []
            for score in levels:
                t = severity.CONDITIONS[condition](score)
                audio, p = synth.render(v, pred, t, centre, range_gain)
                got = severity.achieved(neutral, control.as_render(audio, p), v.sr)
                steps.append((t, got))
                if listen is not None:
                    severity.write_synthetic(listen / severity.synthetic_name(v.name, sid, t), audio, v.sr, v.name, text, t)
            rows.append({"voice": v.name, "sentence": sid, "condition": condition,
                         "neutral_s": round(len(neutral) / v.sr, 2), "neutral_sha256": digest,
                         "centre_hz": round(v.pitch_mean + centre * v.pitch_std, 1), "pauses": len(p.runs),
                         "monotone": severity.monotone(steps),
                         "steps": [{"score": t.score, "target": {f: getattr(t, f) for f in severity.FIELDS},
                                    "achieved": dict(a._asdict()), "misses": severity.misses(a, t)} for t, a in steps]})
    return {"sr": v.sr, "tolerance": severity.TOLERANCE, "rows": rows,
            "settings": {"centre_mode": centre_mode, "range_gain": range_gain, "min_pause_s": control.MIN_PAUSE_S},
            "voice_info": {"name": v.name, "pitch_mean_hz": v.pitch_mean, "pitch_std_hz": v.pitch_std,
                           "hop_ms": round(1000 * v.hop_s, 3), "renderer": "NeMo FastPitch + HiFi-GAN, forced durations and pitch"}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", choices=sorted(SENTENCE_SETS), default="dev")
    ap.add_argument("--fastpitch", default="tts_en_fastpitch")
    ap.add_argument("--hifigan", default="tts_en_hifigan")
    ap.add_argument("--out", default="results/ft3")
    ap.add_argument("--listen", action="store_true", help="write every render under <out>/listen-<voice>-<set>/")
    ap.add_argument("--centre", choices=("audio", "tokens"), default=synth.CENTRE_MODE, help="dev knob: where the range is scaled about")
    ap.add_argument("--range-gain", type=float, default=control.RANGE_GAIN, help="dev knob: a_eff = 1 + (a - 1) * gain")
    args = ap.parse_args(argv)
    t0 = time.time()
    v = synth.load(args.fastpitch, args.hifigan)
    out = pathlib.Path(args.out)
    listen = out / f"listen-{v.name}-{args.sentences}" if args.listen else None
    if listen is not None:
        listen.mkdir(parents=True, exist_ok=True)
    results = run(v, SENTENCE_SETS[args.sentences], listen=listen, centre_mode=args.centre, range_gain=args.range_gain)
    results["seconds"] = round(time.time() - t0)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if (args.centre, args.range_gain) == (synth.CENTRE_MODE, control.RANGE_GAIN) else f"-{args.centre}-gain{args.range_gain:g}"
    path = out / f"sweep-{v.name}-{args.sentences}{tag}.json"
    path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(severity.report(results))
    print(f"\n{v.name}: pitch mean {v.pitch_mean:.1f} Hz, std {v.pitch_std:.1f} Hz; centres "
          f"{sorted({r['centre_hz'] for r in results['rows']})} Hz; {results['seconds']} s on the CPU")
    print(f"saved -> {path}" + (f"; listen: {listen}/" if listen else ""))


if __name__ == "__main__":
    main()
