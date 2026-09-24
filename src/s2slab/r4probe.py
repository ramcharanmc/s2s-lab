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

and each pair gets a DIRECTION: "wider" when d_range > +TIE_ST, "narrower" when
d_range < -TIE_ST, "same" in between (None when a reply had too little voicing
to measure). Nothing is tuned on these pairs; the rungs answer as built.

THE BAND, TIE_ST = 0.5 semitones, was set on the dev pairs: R3's Kokoro is not
bit-deterministic (answering a clip twice differed by up to 0.18 in sample value),
and where R3 wrote the same text for both clips its d_range stayed within
+-0.05 st -- a jitter-level difference that is not a reaction. 0.5 st is ten
times that, and it treats "wider" and "narrower" alike.

    R4  Qwen2.5-Omni hears the clip and speaks its own reply -- the model chooses
        the delivery, which is what the plan asks about.
    R3  Qwen2-Audio hears the same clip, writes a reply, Kokoro speaks it. Kokoro
        takes its delivery from the text alone, so any effect in R3's column is
        the upset clip changing the WORDS ("I'm so sorry to hear that!"), which
        Kokoro then voices. It is the comparator: R4 must beat text-leaking-into-
        voice, not only a coin.

THE DEV RUN (2026-09-23, `a677969`, L40S, $0.20), 21 pairs, counted with the band:

    R3   wider 4   same 14   narrower 3     median d range +0.0 st; words 17 -> 21
    R4   wider 5   same 5    narrower 11    median d range -0.6 st; words 17 -> 17
         (fear: wider 1, narrower 5; median -1.2 st)

R4's voice did not widen for the upset caller; it went flatter more often
than not, with the words unchanged in length. R4 answered a clip twice with
byte-identical audio (the per-turn seed works). PR-008 registers the plan's
directional claim anyway (wider), and the two-sided question beside it.

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
TIE_ST = 0.5                  # |d_range| up to this is "same": see the docstring
P_MAX = 0.05
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
    Returns: [{"id", "emotion", "d_range_st", "d_pitch_st", "d_level_db", "direction",
               "words_calm", "words_upset"}]; direction is "wider", "narrower" or
             "same" by TIE_ST; a delta is None when either reply had too little
             voicing to measure, and then the direction is None too.
    """
    out = []
    for p in pairs:
        n, u = measures[p.neutral], measures[p.loaded]
        dr = None if n["range_st"] is None or u["range_st"] is None else round(u["range_st"] - n["range_st"], 3)
        dp = (None if not n["pitch_hz"] or not u["pitch_hz"]
              else round(12 * math.log2(u["pitch_hz"] / n["pitch_hz"]), 3))
        dl = None if n["level_db"] is None or u["level_db"] is None else round(u["level_db"] - n["level_db"], 3)
        out.append({"id": p.id, "emotion": p.emotion, "d_range_st": dr, "d_pitch_st": dp, "d_level_db": dl,
                    "direction": direction(dr),
                    "words_calm": len((texts.get(p.neutral) or "").split()),
                    "words_upset": len((texts.get(p.loaded) or "").split())})
    return out


def direction(d_range: float | None) -> str | None:
    if d_range is None:
        return None
    return "wider" if d_range > TIE_ST else "narrower" if d_range < -TIE_ST else "same"


def counts(rows: list[dict]) -> dict:
    return {k: sum(r["direction"] == k for r in rows) for k in ("wider", "same", "narrower", None)}


def _median(values) -> str:
    v = [x for x in values if x is not None]
    return f"{np.median(v):+.1f}" if v else "-"


def format_report(pairs, deltas: dict[str, list[dict]], repeat: dict[str, dict] | None = None) -> str:
    """Per rung and emotion: how many pairs move, and the median deltas; then the
    reply lengths, the seeding check, and one row per pair with both rungs side by side."""
    lines = [f"reply to the upset clip against the reply to its calm twin; 'same' is within +-{TIE_ST} st", "",
             "| rung | emotion | pairs | wider | same | narrower | unmeasured | "
             "median d range (st) | median d pitch (st) | median d level (dB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rung, rows in deltas.items():
        for emotion in [e for e in EMOTIONS if any(r["emotion"] == e for r in rows)] + ["all"]:
            mine = [r for r in rows if emotion in ("all", r["emotion"])]
            c = counts(mine)
            lines.append(f"| {NAMES.get(rung, rung)} | {emotion} | {len(mine)} | {c['wider']} | {c['same']} | "
                         f"{c['narrower']} | {c[None]} | "
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
            mark = {"wider": " +", "narrower": " -"}.get(by[r][p.id]["direction"], "")
            cells.append("-" if d is None else f"{d:+.1f}{mark}")
        lines.append(f"| {p.id} | " + " | ".join(cells) + " |")
    lines.append(f"(+ wider, - narrower: beyond +-{TIE_ST} st)")
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


# --------------------------------------------------------------------------- #
# PR-008's criteria
# --------------------------------------------------------------------------- #


def compare(deltas: dict) -> dict:
    """Everything PR-008 asks, as numbers.

    Takes:   {rung: pair_deltas} for "r3-gpu" and "r4-gpu", the same pairs in the same order.
    Returns: {"r4": counts, "r3": counts,
              "p_one": P(R4's wider-vs-narrower split | coin), one-sided (S1),
              "p_two": the same, two-sided (the registered secondary),
              "fixed": ids where R4 is wider and R3 is not, "broken": the reverse,
              "p_s2": one-sided sign test on fixed against broken,
              "S1": bool, "S2": bool}
    """
    from .bargein_eval import sign_test
    from .baseline import sign_test_one_sided
    r4, r3 = deltas["r4-gpu"], deltas["r3-gpu"]
    c4 = counts(r4)
    fixed = sorted(a["id"] for a, b in zip(r4, r3) if a["direction"] == "wider" and b["direction"] != "wider")
    broken = sorted(a["id"] for a, b in zip(r4, r3) if b["direction"] == "wider" and a["direction"] != "wider")
    p_one = sign_test_one_sided(c4["wider"], c4["narrower"])
    p_s2 = sign_test_one_sided(len(fixed), len(broken))
    return {"r4": c4, "r3": counts(r3), "p_one": p_one, "p_two": sign_test(c4["wider"], c4["narrower"]),
            "fixed": fixed, "broken": broken, "p_s2": p_s2,
            "S1": c4["wider"] > c4["narrower"] and p_one < P_MAX,
            "S2": len(fixed) > len(broken) and p_s2 < P_MAX}


def format_criteria(result: dict, confirm: str = "") -> str:
    """S1, S2 and the two-sided secondary as a table, and the verdicts (only for a registered run)."""
    fmt = lambda p: f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    c4, c3 = result["r4"], result["r3"]
    lines = ["| criterion | R4 | R3 | bar | |", "|---|---|---|---|---|",
             f"| S1 R4's reply wider for the upset caller | {c4['wider']} wider, {c4['narrower']} narrower, "
             f"{c4['same']} same | | more wider, one-sided sign test {fmt(result['p_one'])} < {P_MAX} | "
             f"{'PASS' if result['S1'] else 'FAIL'} |",
             f"| S2 R4 against R3, pair by pair | wider where R3 is not: {len(result['fixed'])} | "
             f"wider where R4 is not: {len(result['broken'])} | more, one-sided sign test {fmt(result['p_s2'])} < {P_MAX} | "
             f"{'PASS' if result['S2'] else 'FAIL'} |",
             f"| R3 for reference | | {c3['wider']} wider, {c3['narrower']} narrower, {c3['same']} same | | |",
             f"  secondary, two-sided: R4's delivery moves consistently one way -- {c4['wider']} wider against "
             f"{c4['narrower']} narrower, {fmt(result['p_two'])}"]
    if confirm:
        ok = result["S1"] and result["S2"]
        two = ("narrower" if c4["narrower"] > c4["wider"] else "wider") if result["p_two"] < P_MAX else None
        lines += ["", f"{confirm}: " + ("SUPPORTED: R4's own voice widens its pitch range for the upset caller, "
                                        "beyond what R3's words do through Kokoro" if ok else
                                        "NOT SUPPORTED: failed " + ", ".join(s for s in ("S1", "S2") if not result[s])),
                  f"{confirm} secondary: " + (f"R4's delivery moves consistently -- {two} for the upset caller"
                                              if two else "no consistent direction")]
    return "\n".join(lines)
