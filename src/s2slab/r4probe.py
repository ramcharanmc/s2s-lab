"""
Does R4's voice react to the caller's voice? The laptop side of PR-008.

The v2 plan, R4: "Measure whether output F0 variance actually moves with
sentiment. If it doesn't, the rung bought nothing -- and that is the finding."

THE DESIGN. The prosody pairs (same actor, same sentence, once calm and once
upset) go in as the CALLER's turn. Each rung answers both clips of a pair; the
reply to the upset clip is compared with the reply to its calm twin:

    d_range  F0 range of the reply to the upset clip minus that of the reply to
             the calm clip, semitones (prosodycheck.features, 10th-90th percentile)
    d_pitch  median F0, the same way;   d_level  90th-percentile frame level, dB

and a pair "moves" when d_range > 0: the reply to the upset caller is sung over
a wider pitch range. Nothing is tuned on these pairs; the rungs answer as built.

    R4  Qwen2.5-Omni hears the clip and speaks its own reply -- the model chooses
        the delivery, which is what the plan asks about.
    R3  Qwen2-Audio hears the same clip, writes a reply, Kokoro speaks it. Kokoro
        takes its delivery from the text alone, so any effect in R3's column is
        the upset clip changing the WORDS ("I'm so sorry to hear that!"), which
        Kokoro then voices. It is the comparator: R4 must beat text-leaking-into-
        voice, not only a coin.

Replies are 24 kHz; they are resampled to 16 kHz before measuring, because
prosodycheck's frames and pitch tracker are set for it. Reply audio is written
to results/ (ignored) for listening; only measurements are committed.

    modal run modal_app/r4_bench.py::pairs --split dev                 the 21 dev pairs, both rungs
    modal run modal_app/r4_bench.py::pairs --split test --confirm PR-008 --save runs/prosody/pr008.json
"""

from __future__ import annotations

import io
import json
import math
import pathlib

import numpy as np

from .probes import TARGET_SR
from .prosodyset import EMOTIONS

RUNGS = ("r3-gpu", "r4-gpu")
NAMES = {"r3-gpu": "R3 (Qwen2-Audio -> Kokoro)", "r4-gpu": "R4 (Qwen2.5-Omni speaks)"}


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def answer_clips(rung, clips: dict[str, bytes], repeat: int = 2) -> dict:
    """What the container does per rung, testable with a fake one.

    Takes:   a Rung; {clip: WAV bytes of the caller's turn}; how many clips to answer
             a second time, at the end, to check that a reply depends on its clip alone.
    Calls:   rung.load(), then rung.respond() per clip.
    Returns: {"replies": {clip: {"text", "wav" (the reply, WAV bytes), "total_ms"}},
              "repeat": {clip: largest sample difference between the two answers, 0 if identical,
                         or "length differs"}}.
    """
    import soundfile as sf
    rung.load()
    replies, first = {}, {}
    for name, data in clips.items():
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        turn = rung.respond(audio if audio.ndim == 1 else audio.mean(axis=1), sr)
        first[name] = turn.audio
        replies[name] = {"text": turn.reply_text, "wav": wav_bytes(turn.audio, turn.sr),
                         "total_ms": round(turn.total_ms, 1)}
    again = {}
    for name in list(clips)[:repeat]:
        audio, sr = sf.read(io.BytesIO(clips[name]), dtype="float32")
        a2 = rung.respond(audio if audio.ndim == 1 else audio.mean(axis=1), sr).audio
        again[name] = ("length differs" if len(a2) != len(first[name])
                       else round(float(np.max(np.abs(a2 - first[name]))) if len(a2) else 0.0, 6))
    return {"replies": replies, "repeat": again}


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Polyphase resampling to TARGET_SR, as probes.load_wav does."""
    from scipy.signal import resample_poly
    audio = np.asarray(audio, dtype=np.float32)
    if sr == TARGET_SR:
        return audio
    g = math.gcd(sr, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)


def measure(wav: bytes) -> dict:
    """One reply's delivery. Takes: WAV bytes (any rate). Returns: {"pitch_hz", "range_st",
    "level_db", "seconds"}; pitch and range are None when under 0.1 s of it is voiced."""
    import soundfile as sf
    from .prosodycheck import features
    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    f = features(to_16k(audio, sr))
    finite = lambda x: None if x is None or not math.isfinite(x) else round(float(x), 3)
    return {"pitch_hz": finite(f.f0_median_hz), "range_st": finite(f.f0_range_st),
            "level_db": finite(f.level_db), "seconds": round(len(audio) / sr, 3)}


def pair_deltas(pairs, measures: dict[str, dict], texts: dict[str, str]) -> list[dict]:
    """Per pair: upset-reply minus calm-reply, for one rung.

    Takes:   the pairs; {clip: measure()} and {clip: reply text} for that rung.
    Returns: [{"id", "emotion", "d_range_st", "d_pitch_st", "d_level_db", "moves",
               "words_calm", "words_upset"}]; a delta is None when either reply had
             too little voicing to measure, and such a pair does not move.
    """
    out = []
    for p in pairs:
        n, u = measures[p.neutral], measures[p.loaded]
        dr = None if n["range_st"] is None or u["range_st"] is None else round(u["range_st"] - n["range_st"], 3)
        dp = (None if not n["pitch_hz"] or not u["pitch_hz"]
              else round(12 * math.log2(u["pitch_hz"] / n["pitch_hz"]), 3))
        dl = None if n["level_db"] is None or u["level_db"] is None else round(u["level_db"] - n["level_db"], 3)
        out.append({"id": p.id, "emotion": p.emotion, "d_range_st": dr, "d_pitch_st": dp, "d_level_db": dl,
                    "moves": dr is not None and dr > 0,
                    "words_calm": len((texts.get(p.neutral) or "").split()),
                    "words_upset": len((texts.get(p.loaded) or "").split())})
    return out


def _median(values) -> str:
    v = [x for x in values if x is not None]
    return f"{np.median(v):+.1f}" if v else "-"


def format_report(pairs, deltas: dict[str, list[dict]], repeat: dict[str, dict] | None = None) -> str:
    """Per rung and emotion: how many pairs move, and the median deltas; then the
    reply lengths, the seeding check, and one row per pair with both rungs side by side."""
    lines = ["| rung | emotion | pairs | reply to the upset clip has the wider F0 range | "
             "median d range (st) | median d pitch (st) | median d level (dB) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for rung, rows in deltas.items():
        for emotion in [e for e in EMOTIONS if any(r["emotion"] == e for r in rows)] + ["all"]:
            mine = [r for r in rows if emotion in ("all", r["emotion"])]
            lines.append(f"| {NAMES.get(rung, rung)} | {emotion} | {len(mine)} | {sum(r['moves'] for r in mine)} of {len(mine)} | "
                         f"{_median(r['d_range_st'] for r in mine)} | {_median(r['d_pitch_st'] for r in mine)} | "
                         f"{_median(r['d_level_db'] for r in mine)} |")
    lines.append("")
    for rung, rows in deltas.items():
        wc = [r["words_calm"] for r in rows]
        wu = [r["words_upset"] for r in rows]
        lines.append(f"{NAMES.get(rung, rung)}: reply words, median, to the calm clip {np.median(wc):.0f}, "
                     f"to the upset clip {np.median(wu):.0f}")
    if repeat:
        for rung, rep in repeat.items():
            lines.append(f"{NAMES.get(rung, rung)}, the same clip answered twice: " + ", ".join(
                f"{c} {'identical' if v == 0 else v}" for c, v in rep.items()))
    rungs = list(deltas)
    lines += ["", "| pair | " + " | ".join(f"{NAMES.get(r, r)}: d range (st)" for r in rungs) + " |",
              "|---|" + "---:|" * len(rungs)]
    by = {r: {row["id"]: row for row in deltas[r]} for r in rungs}
    for p in pairs:
        cells = []
        for r in rungs:
            d = by[r][p.id]["d_range_st"]
            cells.append("-" if d is None else f"{d:+.1f}" + (" *" if d > 0 else ""))
        lines.append(f"| {p.id} | " + " | ".join(cells) + " |")
    lines.append("(* the reply to the upset clip has the wider range)")
    return "\n".join(lines)


def write_replies(answers: dict, folder: pathlib.Path) -> None:
    """Every reply as a WAV under results/ (ignored by git): <folder>/<rung>/<clip>.wav, for listening."""
    for rung, got in answers.items():
        d = pathlib.Path(folder) / rung
        d.mkdir(parents=True, exist_ok=True)
        for clip, a in got["replies"].items():
            (d / f"{clip}.wav").write_bytes(a["wav"])


def save(path: pathlib.Path, head: str, pairs, answers: dict, measures: dict, deltas: dict) -> None:
    """Texts, timings, measurements and deltas -- no audio. Never overwrites."""
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replies = {r: {c: {"text": a["text"], "total_ms": a["total_ms"], **measures[r][c]}
                   for c, a in got["replies"].items()} for r, got in answers.items()}
    path.write_text(json.dumps({"run": head, "pairs": [p._asdict() for p in pairs], "replies": replies,
                                "repeat": {r: got.get("repeat", {}) for r, got in answers.items()},
                                "deltas": deltas}, indent=1) + "\n", encoding="utf-8")


def analyse(pairs, answers: dict) -> tuple[dict, dict]:
    """answers from the container -> ({rung: {clip: measure}}, {rung: pair_deltas})."""
    measures = {r: {c: measure(a["wav"]) for c, a in got["replies"].items()} for r, got in answers.items()}
    texts = {r: {c: a["text"] for c, a in got["replies"].items()} for r, got in answers.items()}
    return measures, {r: pair_deltas(pairs, measures[r], texts[r]) for r in answers}
