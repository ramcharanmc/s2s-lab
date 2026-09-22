"""
The own-voice mini set: ten takes by the author, the contamination check for R3 and the tagger.

R3 ranked the upset clip above its calm twin on 39 of 39 CREMA-D pairs
(PR-006). CREMA-D is a standard emotion benchmark and Qwen2-Audio's training
data is not published, so a perfect ranking is consistent with HEARING and with
FAMILIARITY. Ten takes of the author's voice -- the plan's lexically neutral
phrases, each once calm and once upset -- are audio no model has seen: a fresh
voice, a fresh microphone, fresh sentences. Five pairs cannot prove the CREMA-D
result is clean; they can show whether the ranking survives on them. They also
give the tagger (fitted on 16 CREMA-D actors and one microphone) its first
genuinely new speaker, which is the per-caller-baseline argument in one table.

DEV-ONLY: nothing is registered, the audio is never committed (*.wav is
ignored), and the result is reported as descriptive, whichever way it goes.

THE READING, fixed before the run (2026-09-22): chance ranking of 5 pairs is
1 in 32. R3 ranks 4 or 5 of 5 right -> "hearing generalizes to a fresh voice";
3 or fewer -> "the CREMA-D ranking may lean on familiarity; this set cannot say
which". Predictions, to be scored: R3 ranks 5 of 5 and answers 3 of 5 strictly
right (its Yes-lean catches a calm take or two); the tagger tags 3-4 of the 5
upset takes non-neutral and 2-3 of the 5 calm takes neutral.

THE WINDOW. The takes were recorded one file each and left exactly as recorded
(the standing rule: never edit audio). They run 5.8-7.8 s with 0.8-2.3 s of
speech inside, and several start with the click of the record button, where
CREMA-D's clips are 1.5-4.1 s of speech with little silence. The eGeMAPS
functionals the tagger reads are taken over the whole clip, so three seconds
of room tone would move them for a reason that has nothing to do with the
voice. So every take is WINDOWED in code, identically and deterministically:
the longest run of 20 ms frames within FLOOR_DB of the take's loudest frame
(runs closer than GAP_S merged), plus PAD_S of the surrounding silence on each
side. The files on disk are untouched; both the windowed and the as-recorded
audio go to R3 and to the tagger, and the windowed one is primary.

    python -m s2slab.ownvoice look      each take: its window, length, level, pitch; each pair's deltas
    python -m s2slab.ownvoice freeze    manifest.json: SHA-256 of pairs.tsv and the ten takes
    python -m s2slab.ownvoice verify
    python -m s2slab.ownvoice tag       the saved tagger on the five pairs, windowed and as recorded
    modal run modal_app/r3_probe.py::own --save runs/prosody/ownvoice-r3.json      R3, every wording
"""

from __future__ import annotations

import datetime
import io
import json
import math
import pathlib

import numpy as np

from .probes import TARGET_SR, load_wav
from .prosodyset import CLIP_SR, MANIFEST, PAIRS, Pair, _fingerprints, read_pairs, verify
from .vad import frame_rms

OWN_ROOT = pathlib.Path("probes/prosody_own")
PHRASES = {"p1": "I'm fine", "p2": "That's great", "p3": "Sure, whatever you say",
           "p4": "I'll be there at six", "p5": "Okay, I'll do that"}
CALM = "CAL"
RECORDED = "2026-09-22"
FRAME_MS = 20.0
FLOOR_DB = -30.0          # a frame counts as sound within this of the take's loudest frame
GAP_S = 0.5               # sounds closer than this are one run (a pause inside a phrase)
PAD_S = 0.3               # room tone kept on each side of the speech
RAW = ".raw"              # suffix of a clip name that means "as recorded, not windowed"


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


def window(audio: np.ndarray, sr: int = TARGET_SR) -> tuple[float, float]:
    """(start, end) in seconds of the speech in one take.

    Takes:   mono float audio.
    Calls:   vad.frame_rms() on FRAME_MS frames.
    Returns: the longest run of frames louder than the loudest frame + FLOOR_DB,
             after merging runs closer than GAP_S, widened by PAD_S each side
             and clamped to the take. A record-button click at the start is a
             short run far from the words, so it is never the longest.
    Raises:  ValueError on a take too short to frame, or all silent.
    """
    rms = frame_rms(np.asarray(audio, dtype=np.float32), sr, FRAME_MS)
    if rms.size == 0 or float(rms.max()) <= 1e-6:
        raise ValueError("no sound in this take")
    db = 20 * np.log10(rms)
    loud = np.flatnonzero(db > db.max() + FLOOR_DB)
    hop = FRAME_MS / 1000
    runs: list[list[int]] = [[int(loud[0]), int(loud[0])]]
    for i in loud[1:]:
        if (i - runs[-1][1]) * hop < GAP_S:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return max(0.0, start * hop - PAD_S), min(len(audio) / sr, (end + 1) * hop + PAD_S)


def windowed(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    start, end = window(audio, sr)
    return np.asarray(audio, dtype=np.float32)[round(start * sr): round(end * sr)]


def wav_bytes(audio: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """16-bit PCM WAV bytes, the form the Modal side reads."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# The set: freeze, verify, load
# --------------------------------------------------------------------------- #


def freeze(root: pathlib.Path = OWN_ROOT) -> pathlib.Path:
    """manifest.json for the own-voice set: the SHA-256 of pairs.tsv and every take.
    Refuses a second freeze, a missing take, or a take that is not 16 kHz mono."""
    import soundfile as sf
    root = pathlib.Path(root)
    out = root / MANIFEST
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen set is never re-frozen; make a new version")
    pairs = read_pairs(root / PAIRS)
    for clip in sorted({c for p in pairs for c in (p.neutral, p.loaded)}):
        path = root / "audio" / f"{clip}.wav"
        if not path.exists():
            raise FileNotFoundError(f"not recorded yet: {clip}.wav")
        info = sf.info(str(path))
        if (info.samplerate, info.channels) != (CLIP_SR, 1) or info.duration < 0.5:
            raise ValueError(f"{clip}.wav: {info.samplerate} Hz, {info.channels} channel(s), "
                             f"{info.duration:.2f} s -- expected {CLIP_SR} Hz mono, at least 0.5 s")
    out.write_text(json.dumps({"set": "prosody-own", "version": 1, "recorded": RECORDED,
                               "frozen": datetime.date.today().isoformat(), "pairs": len(pairs),
                               "window": {"floor_db": FLOOR_DB, "gap_s": GAP_S, "pad_s": PAD_S},
                               **_fingerprints(root)}, indent=2) + "\n", encoding="utf-8")
    return out


def load(root: pathlib.Path = OWN_ROOT) -> list[tuple[Pair, np.ndarray, np.ndarray]]:
    """[(pair, calm take, upset take)] as recorded, mono float32 at 16 kHz.
    Calls prosodyset.verify() first: a changed or missing take stops everything."""
    root = pathlib.Path(root)
    problems = verify(root)
    if problems:
        raise RuntimeError("the own-voice set is not the frozen one:\n  " + "\n  ".join(problems[:10]))
    return [(p, load_wav(root / "audio" / f"{p.neutral}.wav"), load_wav(root / "audio" / f"{p.loaded}.wav"))
            for p in read_pairs(root / PAIRS)]


def clips(root: pathlib.Path = OWN_ROOT) -> tuple[list[Pair], dict[str, bytes]]:
    """What the Modal side gets: every take twice, windowed under its own name
    and as recorded under name + RAW. Returns (pairs, {name: WAV bytes})."""
    root = pathlib.Path(root)
    pairs, data = [], {}
    for p, calm, upset in load(root):
        pairs.append(p)
        for name, audio in ((p.neutral, calm), (p.loaded, upset)):
            data[name] = wav_bytes(windowed(audio))
            data[name + RAW] = (root / "audio" / f"{name}.wav").read_bytes()
    return pairs, data


def split_answers(answers: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """The container's answers, separated: (windowed, by clip name), (as recorded, by clip name)."""
    return ({c: a for c, a in answers.items() if not c.endswith(RAW)},
            {c[: -len(RAW)]: a for c, a in answers.items() if c.endswith(RAW)})


# --------------------------------------------------------------------------- #
# Looking, and the tagger
# --------------------------------------------------------------------------- #


def look(root: pathlib.Path = OWN_ROOT) -> str:
    """Each take: the window found, the speech inside it, its level and median
    pitch (prosodycheck's measures, on the windowed audio); then each pair's
    upset-minus-calm deltas, to read against the CREMA-D dev medians."""
    from .prosodycheck import features
    lines = ["| take | length (s) | window (s) | speech (s) | level (dB) | pitch (Hz) |", "|---|---:|---|---:|---:|---:|"]
    rows = []
    for p, calm, upset in load(root):
        for name, audio in ((p.neutral, calm), (p.loaded, upset)):
            s, e = window(audio)
            f = features(windowed(audio))
            rows.append((p, f))
            lines.append(f"| {name} | {len(audio) / TARGET_SR:.2f} | {s:.2f}-{e:.2f} | {e - s - 2 * PAD_S:.2f} | "
                         f"{f.level_db:.1f} | {f.f0_median_hz:.0f} |")
    lines += ["", "| pair | phrase | upset | pitch (st) | pitch range (st) | level (dB) |", "|---|---|---|---:|---:|---:|"]
    for (p, fc), (_, fu) in zip(rows[::2], rows[1::2]):
        lines.append(f"| {p.id} | {PHRASES.get(p.sentence, p.sentence)} | {p.emotion} | "
                     f"{12 * math.log2(fu.f0_median_hz / fc.f0_median_hz):+.1f} | "
                     f"{fu.f0_range_st - fc.f0_range_st:+.1f} | {fu.level_db - fc.level_db:+.1f} |")
    return "\n".join(lines)


def tag(tagger, root: pathlib.Path = OWN_ROOT, cache: pathlib.Path | None = None, extract=None) -> str:
    """The tagger on the five pairs, windowed and as recorded: affect.tag_pairs twice.
    Takes: a Tagger; the set's root; a cache folder or None; the extractor (default eGeMAPS)."""
    from .affect import egemaps, features_for, tag_pairs
    extract = extract or egemaps
    root = pathlib.Path(root)
    pairs = [p for p, _, _ in load(root)]
    out = []
    for label, fn, name in (("windowed", lambda a: extract(windowed(a)), "own-windowed.npz"),
                            ("as recorded", extract, "own-raw.npz")):
        c = None if cache is None else pathlib.Path(cache) / name
        xn = features_for([p.neutral for p in pairs], root / "audio", c, extract=fn)
        xl = features_for([p.loaded for p in pairs], root / "audio", c, extract=fn)
        out.append(f"{label}:\n" + tag_pairs(tagger, pairs, xn, xl))
    return "\n\n".join(out)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    if cmd == "look":
        print(look())
    elif cmd == "freeze":
        print(f"frozen -> {freeze()}")
    elif cmd == "verify":
        problems = verify(OWN_ROOT)
        print("\n".join(problems) if problems else "intact")
        sys.exit(1 if problems else 0)
    elif cmd == "tag":
        from .affect import CACHE, TAGGER, Tagger
        print(f"the tagger on the {len(read_pairs(OWN_ROOT / PAIRS))} own-voice pairs, one clip at a time\n")
        print(tag(Tagger.load(TAGGER), cache=CACHE))
    else:
        sys.exit("usage: python -m s2slab.ownvoice look | freeze | verify | tag")
