"""
The cascade at the plans' sizes (M45; v1 §4, P1 week 1; v2 §3), against R1 as built: the two runs of one A10
container read side by side, and the replies' refusals counted by a pattern written before the run
(docs/m45-log.md). A measurement with its predictions written first; nothing registered, nothing "supported".

    PYTHONPATH=src python -m s2slab.plansize      reads the two run files and prints the comparison

    r1-gpu        R1 as built: Whisper base.en (fp16) -> Qwen2.5-0.5B-Instruct -> Kokoro, per clause
    r1-gpu-plan   the plans' sizes: Whisper large-v3-turbo (int8) -> Qwen2.5-7B-Instruct (fp16) -> Kokoro, per clause

A REFUSAL is a reply that matches REFUSAL anywhere, case aside: what Qwen2.5-0.5B says to harmless eval clips
("I'm sorry, but I can't assist with that request."), and its near forms. Every eval clip is a harmless sentence
from LibriSpeech, EdAcc or DisfluencySpeech, so every refusal is a wrong one.
"""

from __future__ import annotations

import json
import pathlib
import re

from .evalset import EVAL_ROOT
from .timing import summarize
from .wer import ZERO, score

REFUSAL = re.compile(r"\b(can't|cannot|can not|won't|unable to|not able to) (help|assist|provide|answer|do that)\b",
                     re.IGNORECASE)
PLAN = pathlib.Path("runs/r1-gpu-plan/evalset-modal-a10.json")
BASE = pathlib.Path("runs/r1-gpu/evalset-modal-a10-plan.json")


def refusals(results: dict) -> list[str]:
    """The clips whose reply matches REFUSAL, in the run's order."""
    return [clip for clip, t in results["turns"].items() if REFUSAL.search(t.get("reply_text") or "")]


def _pct(values) -> dict:
    s = summarize(v for v in values if v is not None)
    return {"n": s.n, "p50": round(s.p50, 1), "p90": round(s.p90, 1)}


def _ms(results: dict, key: str) -> dict:
    return _pct(t.get(key) for t in results["turns"].values())


def _wer(results: dict, root=EVAL_ROOT) -> dict:
    scored = score({c: t["transcript"] or "" for c, t in results["turns"].items()}, root)
    out = {cohort: round(100 * v["raw"].rate, 2) for cohort, v in scored.items()}
    total = ZERO
    for v in scored.values():
        total = total + v["raw"]
    out["all"] = round(100 * total.rate, 2)
    return out


def compare(plan: dict, base: dict, root=EVAL_ROOT) -> dict:
    """Both runs measured the same way; the clips must be the same."""
    if set(plan["turns"]) != set(base["turns"]):
        raise ValueError("the two runs did not hear the same clips")
    out = {}
    for name, r in (("r1-gpu", base), ("r1-gpu-plan", plan)):
        stages = {s: _pct(t["stages_ms"].get(s) for t in r["turns"].values()) for s in ("asr", "llm", "tts")}
        out[name] = {"first_audio_ms": _ms(r, "first_audio_ms"), "first_sound_ms": _ms(r, "first_sound_ms"),
                     "total_ms": _ms(r, "total_ms"), "stages_ms": stages, "wer_raw": _wer(r, root),
                     "refusals": refusals(r), "peak_vram_mb": r.get("peak_vram_mb"), "machine": r.get("machine"),
                     "commit": r.get("commit")}
    return out


def format_report(c: dict) -> str:
    a, b = c["r1-gpu"], c["r1-gpu-plan"]
    rows = [("first audio, p50", a["first_audio_ms"]["p50"], b["first_audio_ms"]["p50"], "ms"),
            ("first audio, p90", a["first_audio_ms"]["p90"], b["first_audio_ms"]["p90"], "ms"),
            ("first sound, p50", a["first_sound_ms"]["p50"], b["first_sound_ms"]["p50"], "ms"),
            ("whole reply, p50", a["total_ms"]["p50"], b["total_ms"]["p50"], "ms"),
            ("whole reply, p90", a["total_ms"]["p90"], b["total_ms"]["p90"], "ms")]
    rows += [(f"{s}, p50", a["stages_ms"][s]["p50"], b["stages_ms"][s]["p50"], "ms") for s in ("asr", "llm", "tts")]
    rows += [(f"WER raw, {k}", a["wer_raw"][k], b["wer_raw"][k], "%") for k in a["wer_raw"]]
    rows += [("refusals, of 200", len(a["refusals"]), len(b["refusals"]), ""),
             ("peak GPU memory", a["peak_vram_mb"], b["peak_vram_mb"], "MB")]
    lines = [f"R1 as built against R1 at the plans' sizes, one container: {b['machine']}", "",
             f"{'':24}{'r1-gpu':>12}{'r1-gpu-plan':>14}"]
    lines += [f"{name:24}{x!s:>12}{y!s:>14} {unit}" for name, x, y, unit in rows]
    lines += ["", "refused by r1-gpu-plan: " + (", ".join(b["refusals"]) or "none")]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    plan, base = (json.loads(p.read_text(encoding="utf-8")) for p in (PLAN, BASE))
    print(format_report(compare(plan, base)))


if __name__ == "__main__":
    main()
