"""
Real-speech probe items: load recordings, propose labels, build Utterances.

The simulator produces `Utterance` objects whose true end of turn is known by
construction. This module produces THE SAME TYPE from a real recording plus a
label file, so `sweep.evaluate`, `diagnose.interruptions` and every policy run
on real speech unchanged. That is the payoff of the TurnContext interface.

Labels are one Audacity region per SPEECH CHUNK -- a stretch of talking between
two pauses -- carrying the words spoken in it:

    0.080  3.220  i would like to change the
    4.540  7.100  delivery address for my order

Everything else follows from those regions: the speaker started at the first
region's start, finished at the last region's end (the true end of turn), and
paused in every gap between regions.

Two things a real recording has that the simulator never did:

  * SILENCE BEFORE THE FIRST WORD. Every policy counts silence from the start
    of the buffer, so leading silence would make each one "endpoint" before the
    speaker said anything. A real endpointer only arms once speech has begun;
    the loader mimics that by cropping to just before the first word, which
    leaves the policy code untouched.
  * NO WORD TIMINGS. Without forced alignment we only know where each chunk
    starts and ends, so word end times are spread evenly across their chunk.
    At any pause between chunks, the last word before it ends exactly at the
    chunk's end -- so the approximation never changes what T1 sees at a pause.

Paths are relative to the repo root; run everything from there.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np

from .vad import FRAME_MS, frame_speech_prob

PROBE_ROOT = pathlib.Path("probes/endpointing")
TARGET_SR = 16_000
LEAD_KEEP_S = 0.10     # silence kept before the first word after cropping
MIN_TRAIL_S = 1.2      # slowest decision: T2 after a function word with rising pitch, 500+400+150 = 1050 ms, plus margin
MIN_PAUSE_S = 0.30     # a gap shorter than this is not a pause
MIN_CHUNK_S = 0.15     # a sound shorter than this is a click, not a word
MIN_VOICED_S = 0.10    # a stretch with less voice than this is a click, breath or noise
MANIFEST = "manifest.json"   # checksums of a frozen set, written by freeze()

def read_script(path: pathlib.Path = PROBE_ROOT / "script.tsv") -> dict[str, dict]:
    """The recording script, keyed by item id.

    Takes:   path to script.tsv (columns id, type, text).
    Returns: {"f01": {"type": "function", "text": "...", "chunks": [...]}, ...}
             `chunks` is the text split at "|" -- the words expected between
             pauses. Natural items have none: their text is a prompt, and what
             was actually said comes from the labels.
    """
    items = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            text = row["text"].strip()
            chunks = [] if row["type"] == "natural" else \
                     [c.strip() for c in text.split("|")]
            items[row["id"]] = {"type": row["type"], "text": text, "chunks": chunks}
    return items

def load_wav(path: pathlib.Path) -> np.ndarray:
    """Read a recording as mono float32 at 16 kHz, whatever it was saved as.

    Takes:   path to a WAV file.
    Returns: 1-D float32 array at TARGET_SR.

    Stereo is averaged to mono. Any other sample rate is resampled with a
    polyphase filter. Treating a 44.1 kHz file as 16 kHz would silently
    stretch it to 2.76x its length and drop its pitch by the same factor.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, sr // g).astype("float32")
    return mono

@dataclass(frozen=True)
class Chunk:
    """One stretch of speech between pauses, as labelled."""

    start_s: float
    end_s: float
    text: str


def read_labels(path: pathlib.Path) -> list[Chunk]:
    """Read an Audacity label export.

    Takes:   path to a .txt written by Audacity's File > Export > Labels.
    Returns: chunks sorted by start time.

    Lines starting with a backslash hold spectral-selection bounds, which
    Audacity writes for some labels; they are skipped.
    """
    chunks = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        start, end, *text = line.split("\t")
        chunks.append(Chunk(float(start), float(end), text[0].strip() if text else ""))
    return sorted(chunks, key=lambda c: c.start_s)


def write_labels(chunks: list[Chunk], path: pathlib.Path) -> pathlib.Path:
    """Write chunks in Audacity's label format: start<TAB>end<TAB>text."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c.start_s:.6f}\t{c.end_s:.6f}\t{c.text}\n"
                            for c in chunks), encoding="utf-8")
    return path

def detect_chunks(audio: np.ndarray, sr: int = TARGET_SR) -> list[tuple[float, float]]:
    """Find stretches of speech, using the same VAD the policies use.

    Takes:   mono audio at `sr`.
    Returns: [(start_s, end_s), ...] for each stretch of speech, after merging
             stretches separated by less than MIN_PAUSE_S and dropping sounds
             shorter than MIN_CHUNK_S (clicks, lip smacks).
    """
    speech = [p >= 0.5 for p in frame_speech_prob(audio, sr)]
    step = FRAME_MS / 1000
    runs, start = [], None
    for i, s in enumerate(speech + [False]):      # the extra False closes a final run
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start * step, i * step])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < MIN_PAUSE_S:
            merged[-1][1] = r[1]                  # gap too short to be a pause: join
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= MIN_CHUNK_S]

def _voiced_s(track: list[float], start_s: float, end_s: float,
              hop_s: float = 0.010) -> float:
    """Seconds of detectable voice (pitch) between start_s and end_s.

    Takes:   a pitch track from _pitch_track() and a time span.
    Returns: the number of voiced frames in the span, times hop_s.

    A click, a breath or a burst of room noise is loud but has no pitch, so
    energy calls it speech and this calls it nothing.
    """
    return sum(1 for f in track[int(start_s / hop_s): int(end_s / hop_s)] if f > 0) * hop_s


def _first_voiced_run_s(track: list[float], start_s: float, end_s: float,
                        hop_s: float = 0.010, run: int = 5) -> float | None:
    """Start of the first stretch of `run` consecutive voiced frames in a span.

    Takes:   a pitch track and a time span.
    Returns: that start in seconds, or None if the span never holds voice
             for run x hop_s (50 ms) in a row.

    Requiring 50 ms in a row skips the odd single frame the pitch tracker
    finds in noise -- f04 has two before your first word, which would
    otherwise pull the start back into the noise.
    """
    count = 0
    for k in range(int(start_s / hop_s), min(int(end_s / hop_s), len(track))):
        count = count + 1 if track[k] > 0 else 0
        if count == run:
            return (k - run + 1) * hop_s
    return None

def propose_labels(item_id: str, root: pathlib.Path = PROBE_ROOT,
                   overwrite: bool = False) -> pathlib.Path:
    """Write a first-draft label file for one recording, for you to correct.

    Takes:   an item id ("f01") and the probe folder.
    Calls:   read_script(), load_wav(), detect_chunks(), write_labels().
    Returns: the path of labels/<id>.txt.

    For a scripted item with k pauses, the k LONGEST gaps are taken as the
    scripted pauses and every other gap is merged, so a stray short gap inside
    a sentence does not split a chunk. The script's words are then filled in.
    Natural items keep every detected pause, with "?" as the text for you to
    replace with what you actually said.
    """
    item = read_script(root / "script.tsv")[item_id]
    audio = load_wav(root / "audio" / f"{item_id}.wav")
    track = _pitch_track(audio)
    # Drop stretches with no voice in them: the record click, a breath, a stop
    # click after you finish. Left in, the gap before a stray sound at the end
    # is often the longest one, and the draft picks it as "the pause".
    spans = [s for s in detect_chunks(audio) if _voiced_s(track, *s) >= MIN_VOICED_S]
    if not spans:
        raise ValueError(f"{item_id}: no speech detected")

    expected = item["chunks"]
    if expected and len(spans) > len(expected):
        gaps = sorted(range(len(spans) - 1),
                      key=lambda i: spans[i + 1][0] - spans[i][1], reverse=True)
        keep = sorted(gaps[: len(expected) - 1])   # gap indices that stay pauses
        merged, first = [], 0
        for i in keep + [len(spans) - 1]:
            merged.append((spans[first][0], spans[i][1]))
            first = i + 1
        spans = merged

    # Start the first region where the voice starts, not at the record click
    # or breath before it. 30 ms of margin: a crop keeps LEAD_KEEP_S more, and
    # the two together must stay under the shortest policy threshold (150 ms).
    onset = _first_voiced_run_s(track, *spans[0])
    if onset is not None:
        spans[0] = (max(spans[0][0], onset - 0.03), spans[0][1])

    texts = expected if len(expected) == len(spans) else ["?"] * len(spans)
    out = root / "labels" / f"{item_id}.txt"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists and may hold your corrections; "
                              f"use --force to replace it with a fresh draft")
    return write_labels([Chunk(a, b, t) for (a, b), t in zip(spans, texts)], out)

def _validate(item_id: str, chunks: list[Chunk], duration_s: float) -> None:
    """Refuse labels that would produce a wrong answer rather than an error."""
    if not chunks:
        raise ValueError(f"{item_id}: no labelled chunks")
    if any(not c.text or c.text == "?" for c in chunks):
        raise ValueError(f"{item_id}: a chunk has no words -- fill it in Audacity")
    if any(c.end_s <= c.start_s for c in chunks):
        raise ValueError(f"{item_id}: a region ends before it starts (a point label?)")
    if any(b.start_s < a.end_s for a, b in zip(chunks, chunks[1:])):
        raise ValueError(f"{item_id}: regions overlap")
    trail = duration_s - chunks[-1].end_s
    if trail < MIN_TRAIL_S:
        raise ValueError(f"{item_id}: only {trail:.2f}s after the last word; the "
                         f"slowest policy needs {MIN_TRAIL_S}s -- re-record it")

def _pitch_track(audio: np.ndarray, hop_s: float = 0.010) -> list[float]:
    """F0 per 10 ms frame, 0.0 where unvoiced -- the simulator's convention.

    Same Praat settings as prosody.extract_prosody and the same 10 ms hop as
    simulate.F0_HOP_S, so T2 reads real pitch exactly the way it reads
    synthetic pitch. Each Praat frame is placed by its own timestamp: Praat's
    first frame sits half an analysis window into the audio, and assuming it
    started at zero would shift the whole track.
    """
    import parselmouth

    pitch = parselmouth.Sound(audio.astype("float64"), sampling_frequency=TARGET_SR) \
        .to_pitch(time_step=hop_s, pitch_floor=60, pitch_ceiling=400)
    track = [0.0] * int(math.ceil(len(audio) / TARGET_SR / hop_s))
    for t, f in zip(pitch.xs(), pitch.selected_array["frequency"]):
        k = int(t / hop_s)
        if 0 <= k < len(track):
            track[k] = float(f)
    return track

def load_item(item_id: str, root: pathlib.Path = PROBE_ROOT):
    """One labelled recording as an Utterance -- the simulator's own type.

    Takes:   an item id ("f01") and the probe folder.
    Calls:   read_script(), load_wav(), read_labels(), _validate(),
             _pitch_track(), prosody.count_syllable_nuclei().
    Returns: an Utterance, cropped to start LEAD_KEEP_S before the first word,
             whose true_end_s is the end of the last labelled chunk.
    """
    from .prosody import count_syllable_nuclei
    from .simulate import Utterance

    item = read_script(root / "script.tsv")[item_id]
    audio = load_wav(root / "audio" / f"{item_id}.wav")
    chunks = read_labels(root / "labels" / f"{item_id}.txt")
    _validate(item_id, chunks, len(audio) / TARGET_SR)

    # Crop: a real endpointer only starts listening once speech has begun.
    t0 = max(0.0, chunks[0].start_s - LEAD_KEEP_S)
    audio = audio[int(round(t0 * TARGET_SR)):]
    chunks = [Chunk(c.start_s - t0, c.end_s - t0, c.text) for c in chunks]

    # Spread each chunk's words evenly across it; the last word ends exactly
    # at the chunk's end, which is all T1 ever reads at a pause.
    words: list[tuple[str, float]] = []
    for c in chunks:
        ws = c.text.lower().split()
        for k, w in enumerate(ws, start=1):
            words.append((w, c.start_s + (c.end_s - c.start_s) * k / len(ws)))

    return Utterance(
        audio=audio,
        sr=TARGET_SR,
        true_end_s=chunks[-1].end_s,
        cohort=f"probe_{item['type']}",
        words=words,
        n_syllables=count_syllable_nuclei(audio, TARGET_SR),
        mid_turn_pauses_s=[b.start_s - a.end_s for a, b in zip(chunks, chunks[1:])],
        f0_track=_pitch_track(audio),
    )

#crop at max(0, 0.080 − 0.100) = 0.000 s   → nothing to crop; you started speaking at once
#true_end_s        = 7.840
#mid_turn_pauses_s = [1.34]
#chunk 1: 0.080–3.320, 6 words → one word every 0.54 s
#words[:6] = [("i", 0.62), ("would", 1.16), ("like", 1.70),
#            ("to", 2.24), ("change", 2.78), ("the", 3.32)]   ← "the" ends at the chunk's end

def load_set(root: pathlib.Path = PROBE_ROOT) -> dict:
    """Every item with both a recording and a label file, in script order.

    Takes:   the probe folder.
    Returns: {"f01": Utterance, ...}. Items not yet recorded or labelled are
             skipped rather than treated as errors, so the set is usable while
             it is still being recorded.
    """
    out = {}
    for item_id in read_script(root / "script.tsv"):
        if (root / "audio" / f"{item_id}.wav").exists() and \
           (root / "labels" / f"{item_id}.txt").exists():
            out[item_id] = load_item(item_id, root)
    return out

CHECK_POLICIES = ("T0_fixed_500ms", "T0_fixed_700ms", "T0_fixed_900ms",
                  "T1_semantic_500ms", "T2_prosodic_500ms")
_SHORT = {"T0_fixed_500ms": "T0 500", "T0_fixed_700ms": "T0 700",
          "T0_fixed_900ms": "T0 900", "T1_semantic_500ms": "T1 500",
          "T2_prosodic_500ms": "T2 500"}


def check(root: pathlib.Path = PROBE_ROOT) -> str:
    """Run a handful of policies on every labelled recording.

    Takes:   the probe folder.
    Calls:   load_set(); each policy's decide(); diagnose.classify_tail() to
             name the word an interruption followed.
    Returns: a markdown table. A cell reads "+420 ms" when the policy waited
             that long after the true end, "CUT after 'the'" when it
             interrupted, and "no decision" when it never fired.
    """
    from .diagnose import classify_tail
    from .policies import get

    lines = ["| item | type | pauses s | " + " | ".join(_SHORT[p] for p in CHECK_POLICIES) + " |",
             "|---|---|---|" + "---|" * len(CHECK_POLICIES)]
    for item_id, u in load_set(root).items():
        ctx = u.context()
        cells = []
        for name in CHECK_POLICIES:
            d = get(name).decide(ctx)
            if not d.fired:
                cells.append("no decision")
            elif d.endpoint_s < u.true_end_s:
                word, _ = classify_tail(ctx.partial_at(d.endpoint_s))
                cells.append(f"CUT after '{word}'")
            else:
                cells.append(f"+{(d.endpoint_s - u.true_end_s) * 1000:.0f} ms")
        pauses = ", ".join(f"{p:.2f}" for p in u.mid_turn_pauses_s) or "—"
        lines.append(f"| {item_id} | {u.cohort.removeprefix('probe_')} | {pauses} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)

def sha256_file(path: pathlib.Path, text: bool = False) -> str:
    """The SHA-256 of a file's bytes, as 64 hex characters.

    Change a single bit anywhere in the file and roughly half of those
    characters change. With text=True, Windows line endings (CRLF) are read as
    LF first: Audacity on Windows saves labels with CRLF, git stores them with
    LF, and both copies must get the same checksum.
    """
    data = path.read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _fingerprints(root: pathlib.Path) -> dict:
    """What a frozen set pins down: the script, and each item's two files.

    Returns: {"script_sha256": "...",
              "items": {"f01": {"type": "function",
                                "wav_sha256": "..." or None,
                                "labels_sha256": "..." or None}, ...}}
             None marks a file that does not exist.
    """
    items = {}
    for item_id, item in read_script(root / "script.tsv").items():
        wav = root / "audio" / f"{item_id}.wav"
        lab = root / "labels" / f"{item_id}.txt"
        items[item_id] = {"type": item["type"],
                          "wav_sha256": sha256_file(wav) if wav.exists() else None,
                        "labels_sha256": sha256_file(lab, text=True) if lab.exists() else None}
    return {"script_sha256": sha256_file(root / "script.tsv", text=True), "items": items}


def freeze(root: pathlib.Path = PROBE_ROOT) -> pathlib.Path:
    """Write manifest.json: a checksum of every recording and label file.

    Takes:   the probe folder. Every item in script.tsv must have a recording
             and labels that pass load_item's checks.
    Calls:   _fingerprints(); load_item() on every item.
    Returns: the manifest's path.
    Raises:  FileExistsError if the set is already frozen: a frozen set never
             changes, and additions become a new version. ValueError if any
             item is unrecorded or unlabelled. Either way, nothing is written.
    """
    path = root / MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists: a frozen set never changes")
    fp = _fingerprints(root)
    missing = [i for i, v in fp["items"].items()
               if v["wav_sha256"] is None or v["labels_sha256"] is None]
    if missing:
        raise ValueError(f"cannot freeze: no recording or labels for {', '.join(missing)}")
    for item_id in fp["items"]:
        load_item(item_id, root)          # raises if an item's labels are unusable
    manifest = {"set": root.name, "version": 1,
                "frozen": datetime.date.today().isoformat(), **fp}
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def verify(root: pathlib.Path = PROBE_ROOT) -> list[str]:
    """Compare every file against the manifest.

    Takes:   the probe folder.
    Calls:   _fingerprints() to hash the files as they are now.
    Returns: one line per problem, e.g. "c05: recording changed" or
             "f03: labels missing". An empty list means the set is intact.
    """
    path = root / MANIFEST
    if not path.exists():
        return [f"no {MANIFEST}: the set is not frozen yet"]
    want = json.loads(path.read_text(encoding="utf-8"))
    have = _fingerprints(root)
    problems = []
    if have["script_sha256"] != want["script_sha256"]:
        problems.append("script.tsv changed")
    for item_id, w in want["items"].items():
        h = have["items"].get(item_id)
        if h is None:
            problems.append(f"{item_id}: removed from script.tsv")
            continue
        for key, what in (("wav_sha256", "recording"), ("labels_sha256", "labels")):
            if h[key] is None:
                problems.append(f"{item_id}: {what} missing")
            elif h[key] != w[key]:
                problems.append(f"{item_id}: {what} changed")
    for item_id in have["items"]:
        if item_id not in want["items"]:
            problems.append(f"{item_id}: not in the manifest")
    return problems


if __name__ == "__main__":
    import sys

    cmd, *ids = sys.argv[1:] or ["help"]
    if cmd == "propose" and ids:
        force = "--force" in ids
        ids = [i for i in ids if i != "--force"]
        script = read_script()
        for item_id in ids:
            try:
                path = propose_labels(item_id, overwrite=force)
            except FileExistsError:
                print(f"{item_id}: labels already exist -- kept yours (use --force to redraft)")
                continue
            n_found = len(read_labels(path))
            n_expected = len(script[item_id]["chunks"])
            note = "" if not n_expected or n_found == n_expected else \
                   f"   <-- expected {n_expected} chunks: fix in Audacity"
            print(f"{item_id}: {n_found} chunks -> {path}{note}")
    elif cmd == "check":
        print(check())
    elif cmd == "freeze":
        path = freeze()
        n = len(json.loads(path.read_text(encoding="utf-8"))["items"])
        print(f"froze {n} items -> {path}")
    elif cmd == "verify":
        problems = verify()
        print("\n".join(problems) or "intact: every recording and label file matches the manifest")
        sys.exit(1 if problems else 0)
    else:
        print("usage: python -m s2slab.probes propose f01 c01 n01\n"
              "       python -m s2slab.probes check\n"
              "       python -m s2slab.probes freeze\n"
              "       python -m s2slab.probes verify")