"""
The barge-in probe set: render the agent, check the recordings, freeze the set.

While the agent speaks, the caller makes a sound. Should the agent stop? The
set (probes/README.md, "Barge-in probe set") has 20 items: stop or continue,
crossed with a short or a long sound from the caller.

    python -m s2slab.bargein render        the five agent replies -> agent/A1.wav ...
    python -m s2slab.bargein calibrate     how far the recording lags the playback
    python -m s2slab.bargein propose       a first-draft label per recording
    python -m s2slab.bargein check         every item, and what a duration threshold does
    python -m s2slab.bargein freeze        checksums of everything -> manifest.json
    python -m s2slab.bargein verify        has anything changed since?

ONE TIMELINE. A recording starts when the agent's reply starts playing, so a
time in the recording is a time in the playback -- once the overdub lag that
`calibrate` measures is taken off. Everything a policy is later judged on hangs
from one number per item:

    onset_s        when the caller's sound starts, on the agent's timeline
    duration_s     how long the sound lasts -- measured, whatever the cell is called
    entry_delay_s  onset minus the end of the phrase the script said to come in after
    agent_words    which agent words had finished by then (what state truncation needs)

RENDER ONCE. Kokoro's audio is not bit-identical between two runs, even on one
machine. The labels are only meaningful against the audio that was playing, so
`render` refuses to overwrite, and the agent WAVs are hashed in the manifest.
Kokoro also reports when it speaks each word; those times go to
agent_words.tsv, which is committed.
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib
from dataclasses import dataclass

import numpy as np

from .probes import (MIN_VOICED_S, TARGET_SR, Chunk, _pitch_track, _voiced_s, detect_chunks,
                     load_wav, read_labels, sha256_file, write_labels)

BARGEIN_ROOT = pathlib.Path("probes/bargein")
MANIFEST = "manifest.json"
CALIBRATION = "calibration.json"
AGENT_WORDS = "agent_words.tsv"
CELLS = ("stop-long", "stop-short", "continue-short", "continue-long")
PER_CELL = 5
SHORT_MAX_S, LONG_MIN_S = 0.8, 1.2     # what "short" and "long" mean when checking a recording
EARLY_S = 0.5                          # a sound this long before the phrase ends is not the item's
MAX_LAG_S = 1.0                        # calibrate looks for the lag within +/- this
MIN_SHARPNESS = 8.0                    # correlation peak over its background; below this, redo it
THRESHOLDS_MS = (200, 400, 800, 1500)  # the naive policy previewed by check()



# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


def _tsv(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _plain(text: str) -> list[str]:
    """Words only: lower case, punctuation dropped. "Half past two." -> [half, past, two]."""
    return [p for p in ("".join(c for c in w.lower() if c.isalnum()) for w in text.split()) if p]


def _count_phrase(words: list[str], phrase: list[str]) -> int:
    return sum(words[k:k + len(phrase)] == phrase for k in range(len(words) - len(phrase) + 1))


def read_agent(root: pathlib.Path = BARGEIN_ROOT) -> dict[str, str]:
    """The agent's replies: {"A1": "Thanks for calling. ...", ...}."""
    return {row["id"]: row["text"].strip() for row in _tsv(pathlib.Path(root) / "agent.tsv")}


def read_script(root: pathlib.Path = BARGEIN_ROOT) -> dict[str, dict]:
    """The items, keyed by id, checked against the design.

    Returns: {"ss03": {"cell": "stop-short", "kind": "speech", "agent": "A3",
                       "after": "the fourteenth", "text": "no, wait",
                       "should_stop": True}, ...}
    Raises:  ValueError if a cell does not have PER_CELL items, an agent line is
             used twice in a cell, or an `after` phrase is not in its agent
             line exactly once -- each would quietly unbalance the set.
    """
    agent = read_agent(root)
    items = {}
    for row in _tsv(pathlib.Path(root) / "script.tsv"):
        if row["cell"] not in CELLS or row["kind"] not in ("speech", "noise"):
            raise ValueError(f"{row['id']}: unknown cell or kind")
        if _count_phrase(_plain(agent[row["agent"]]), _plain(row["after"])) != 1:
            raise ValueError(f"{row['id']}: {row['after']!r} must occur exactly once in {row['agent']}")
        items[row["id"]] = {**{k: row[k].strip() for k in ("cell", "kind", "agent", "after", "text")},
                            "should_stop": row["cell"].startswith("stop")}
    for cell in CELLS:
        used = sorted(it["agent"] for it in items.values() if it["cell"] == cell)
        if len(used) != PER_CELL or len(set(used)) != len(used):
            raise ValueError(f"{cell}: needs {PER_CELL} items, each on a different agent line; has {used}")
    return items



# --------------------------------------------------------------------------- #
# The agent's side
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float


def kokoro_synth(text: str, _tts=[]) -> tuple[np.ndarray, int, list[Word]]:
    """Takes: a reply. Returns: (audio, sample rate, when each word is spoken).

    Kokoro predicts a duration for every token, and reports it. A long text
    comes back in several results; each one's times start at zero, so they are
    shifted by the audio that came before. Punctuation is kept out of the words.
    """
    if not _tts:                                  # load once per process
        from .r0 import KokoroTTS
        _tts.append(KokoroTTS())
        _tts[0].load()
    tts = _tts[0]
    parts, words, offset = [], [], 0.0
    for result in tts.pipeline(text, voice=tts.voice):
        if result.audio is None:
            continue
        for tok in result.tokens or []:
            if tok.start_ts is not None and tok.end_ts is not None and any(c.isalnum() for c in tok.text):
                words.append(Word(tok.text, offset + tok.start_ts, offset + tok.end_ts))
        audio = np.asarray(result.audio, dtype=np.float32)
        parts.append(audio)
        offset += len(audio) / tts.SR
    return np.concatenate(parts), tts.SR, words


def render(root: pathlib.Path = BARGEIN_ROOT, synth=kokoro_synth) -> list[str]:
    """Synthesize every agent reply, once.

    Takes:   the probe folder; synth(text) -> (audio, sr, [Word]) (tests pass a fake).
    Writes:  agent/<id>.wav, and agent_words.tsv with every word's times.
    Returns: one line per reply, for printing.
    Raises:  FileExistsError if any of it exists: the recordings are made
             against THIS audio, and a second render would not be the same audio.
    """
    import soundfile as sf

    root = pathlib.Path(root)
    agent = read_agent(root)
    existing = [p for p in [root / AGENT_WORDS, *(root / "agent" / f"{a}.wav" for a in agent)] if p.exists()]
    if existing:
        raise FileExistsError(f"already rendered ({existing[0]}); the recordings are made against "
                              "that audio. Delete agent/ and agent_words.tsv only if nothing has "
                              "been recorded yet.")
    (root / "agent").mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for agent_id, text in agent.items():
        audio, sr, words = synth(text)
        sf.write(str(root / "agent" / f"{agent_id}.wav"), audio, sr, subtype="PCM_16")
        rows += [(agent_id, k, w.text, f"{w.start_s:.4f}", f"{w.end_s:.4f}") for k, w in enumerate(words)]
        lines.append(f"{agent_id}: {len(audio) / sr:5.1f} s, {len(words)} words -> agent/{agent_id}.wav")
    with open(root / AGENT_WORDS, "w", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["agent", "index", "word", "start_s", "end_s"])
        out.writerows(rows)
    return lines


def read_agent_words(root: pathlib.Path = BARGEIN_ROOT) -> dict[str, list[Word]]:
    """{"A3": [Word("Your", 0.275, 0.425), ...]} from agent_words.tsv."""
    words: dict[str, list[Word]] = {}
    for row in _tsv(pathlib.Path(root) / AGENT_WORDS):
        words.setdefault(row["agent"], []).append(Word(row["word"], float(row["start_s"]), float(row["end_s"])))
    return words


def phrase_span_s(words: list[Word], phrase: str) -> tuple[float, float]:
    """When the agent starts and finishes saying `phrase`.

    Takes:   an agent line's words, and a phrase from the script's `after` column.
    Returns: (start of its first word, end of its last). ValueError unless the
             phrase is found exactly once, word for word.
    """
    target = _plain(phrase)
    spoken = [(_plain(w.text) or [""])[0] for w in words]
    hits = [k for k in range(len(spoken) - len(target) + 1) if spoken[k:k + len(target)] == target]
    if len(hits) != 1:
        raise ValueError(f"{phrase!r} found {len(hits)} times in the agent's words")
    return words[hits[0]].start_s, words[hits[0] + len(target) - 1].end_s



# --------------------------------------------------------------------------- #
# Calibration: how far the recording lags the playback
# --------------------------------------------------------------------------- #


def measure_lag(recorded: np.ndarray, played: np.ndarray, sr: int = TARGET_SR) -> tuple[float, float]:
    """Takes: the calibration recording (a headphone held to the microphone)
             and the agent audio that was playing, both at `sr`.
    Returns: (lag in ms -- positive if the recording is late, sharpness).

    Cross-correlation: slide one signal over the other and find the shift where
    they line up best. Sharpness is that peak over the typical level elsewhere;
    a recording that is mostly room noise has no clear peak, and says so.
    """
    from scipy.signal import correlate

    n = min(len(recorded), len(played))
    a = recorded[:n] - recorded[:n].mean()
    b = played[:n] - played[:n].mean()
    corr = np.abs(correlate(a, b, mode="full", method="fft"))
    zero, span = n - 1, int(MAX_LAG_S * sr)
    window = corr[zero - span: zero + span + 1]
    peak = int(np.argmax(window))
    sharpness = float(window[peak] / (np.median(window) + 1e-12))
    return (peak - span) / sr * 1000, sharpness


def calibrate(root: pathlib.Path = BARGEIN_ROOT, agent_id: str = "A3") -> str:
    """Measure the overdub lag from audio/calibration.wav and write calibration.json."""
    root = pathlib.Path(root)
    lag_ms, sharpness = measure_lag(load_wav(root / "audio" / "calibration.wav"),
                                    load_wav(root / "agent" / f"{agent_id}.wav"))
    if sharpness < MIN_SHARPNESS:
        raise ValueError(f"no clear match with {agent_id} (sharpness {sharpness:.1f}, need "
                         f"{MIN_SHARPNESS}): was the headphone against the microphone, and was it {agent_id}?")
    (root / CALIBRATION).write_text(json.dumps(
        {"agent": agent_id, "lag_ms": round(lag_ms, 1), "sharpness": round(sharpness, 1),
         "measured": datetime.date.today().isoformat()}, indent=2) + "\n", encoding="utf-8")
    return (f"the recording lags the playback by {lag_ms:+.1f} ms (sharpness {sharpness:.0f}); "
            f"written to {CALIBRATION}")


def read_lag_s(root: pathlib.Path = BARGEIN_ROOT) -> float:
    path = pathlib.Path(root) / CALIBRATION
    if not path.exists():
        raise FileNotFoundError(f"no {CALIBRATION}: run `python -m s2slab.bargein calibrate` first")
    return json.loads(path.read_text(encoding="utf-8"))["lag_ms"] / 1000



# --------------------------------------------------------------------------- #
# The caller's side
# --------------------------------------------------------------------------- #


def propose_label(item_id: str, root: pathlib.Path = BARGEIN_ROOT, overwrite: bool = False) -> pathlib.Path:
    """Write a first-draft label for one recording: one region, first sound to last.

    Calls:   probes.detect_chunks() (the same VAD the policies use) and, for
             speech items, the pitch check that tells a voice from a click. A
             cough has no pitch, so `noise` items skip that check.
    Ignores: anything that starts more than EARLY_S before the agent finishes
             the `after` phrase -- the record click, a breath while listening.
    """
    root = pathlib.Path(root)
    item = read_script(root)[item_id]
    out = root / "labels" / f"{item_id}.txt"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists and may hold your corrections; "
                              f"use --force to replace it with a fresh draft")
    audio = load_wav(root / "audio" / f"{item_id}.wav")
    earliest = phrase_span_s(read_agent_words(root)[item["agent"]], item["after"])[1] - EARLY_S
    spans = [s for s in detect_chunks(audio) if s[0] >= earliest]
    if item["kind"] == "speech":
        track = _pitch_track(audio)
        spans = [s for s in spans if _voiced_s(track, *s) >= MIN_VOICED_S]
    if not spans:
        raise ValueError(f"{item_id}: no sound found after {item['after']!r}")
    return write_labels([Chunk(spans[0][0], spans[-1][1], item["text"])], out)


@dataclass
class BargeIn:
    """One item, ready for a policy: the caller's audio on the agent's timeline."""

    id: str
    cell: str
    kind: str                 # "speech" or "noise"
    should_stop: bool
    agent_id: str
    audio: np.ndarray         # the caller's channel, 16 kHz, time zero = the agent starts
    onset_s: float            # caller's sound starts, on the agent's timeline (lag removed)
    duration_s: float
    entry_delay_s: float      # onset minus the end of the `after` phrase
    agent_s: float            # how long the agent's reply is
    agent_words: list[Word]

    def words_finished_by(self, t_s: float) -> int:
        """How many agent words had been fully spoken by time t -- what the
        caller can have heard if the agent stops then."""
        return sum(1 for w in self.agent_words if w.end_s <= t_s)


def load_item(item_id: str, root: pathlib.Path = BARGEIN_ROOT) -> BargeIn:
    """One recording plus its label, checked.

    Raises ValueError for anything that would give a wrong answer rather than
    an error: not exactly one region, an empty region, a sound that starts
    before the agent even began the `after` phrase, or a recording that stops
    before the agent's reply does.
    """
    import soundfile as sf

    root = pathlib.Path(root)
    item = read_script(root)[item_id]
    audio = load_wav(root / "audio" / f"{item_id}.wav")
    regions = read_labels(root / "labels" / f"{item_id}.txt")
    if len(regions) != 1:
        raise ValueError(f"{item_id}: needs exactly one region, has {len(regions)}")
    region = regions[0]
    if region.end_s <= region.start_s or region.end_s > len(audio) / TARGET_SR:
        raise ValueError(f"{item_id}: the region is empty or runs past the recording")
    words = read_agent_words(root)[item["agent"]]
    agent_s = sf.info(str(root / "agent" / f"{item['agent']}.wav")).duration
    lag_s = read_lag_s(root)
    onset_s = region.start_s - lag_s
    after_start, after_end = phrase_span_s(words, item["after"])
    if onset_s < after_start:
        raise ValueError(f"{item_id}: the sound starts at {onset_s:.2f} s, before the agent even "
                         f"began {item['after']!r} ({after_start:.2f} s) -- re-record it")
    if len(audio) / TARGET_SR - lag_s < agent_s:
        raise ValueError(f"{item_id}: the recording ends before the agent's reply does "
                         f"({len(audio) / TARGET_SR:.1f} s against {agent_s:.1f} s) -- re-record it")
    return BargeIn(item_id, item["cell"], item["kind"], item["should_stop"], item["agent"], audio,
                   onset_s, region.end_s - region.start_s, onset_s - after_end, agent_s, words)


def load_set(root: pathlib.Path = BARGEIN_ROOT) -> dict[str, BargeIn]:
    """Every item that has a recording and a label, in script order."""
    root = pathlib.Path(root)
    return {i: load_item(i, root) for i in read_script(root)
            if (root / "audio" / f"{i}.wav").exists() and (root / "labels" / f"{i}.txt").exists()}


def duration_policy_errors(items, threshold_ms: float) -> dict[str, int]:
    """The naive policy: stop if the caller's sound lasts at least threshold_ms.
    Returns: wrong decisions per cell."""
    wrong = {cell: 0 for cell in CELLS}
    for it in items:
        if (it.duration_s * 1000 >= threshold_ms) != it.should_stop:
            wrong[it.cell] += 1
    return wrong


def check(root: pathlib.Path = BARGEIN_ROOT) -> str:
    """Every loadable item, then what a duration threshold would do with them."""
    root = pathlib.Path(root)
    script, lines, loaded = read_script(root), [], []
    lines += ["| item | cell | kind | onset s | lasts s | after the phrase by | agent words heard | note |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for item_id in script:
        if not (root / "audio" / f"{item_id}.wav").exists():
            continue
        try:
            it = load_item(item_id, root)
        except (ValueError, FileNotFoundError) as e:
            lines.append(f"| {item_id} | {script[item_id]['cell']} | | | | | | **{e}** |")
            continue
        loaded.append(it)
        short = it.cell.endswith("short")
        note = ("longer than a short item should be" if short and it.duration_s > SHORT_MAX_S else
                "shorter than a long item should be" if not short and it.duration_s < LONG_MIN_S else "")
        lines.append(f"| {it.id} | {it.cell} | {it.kind} | {it.onset_s:.2f} | {it.duration_s:.2f} | "
                     f"{it.entry_delay_s * 1000:+.0f} ms | {it.words_finished_by(it.onset_s)} of "
                     f"{len(it.agent_words)} | {note} |")
    lines += ["", f"{len(loaded)} of {len(script)} items load.", "",
              "Wrong decisions if the agent stops whenever the caller's sound lasts at least N ms:", "",
              "| N ms | " + " | ".join(CELLS) + " | total |", "|---:|" + "---:|" * (len(CELLS) + 1)]
    for ms in THRESHOLDS_MS:
        wrong = duration_policy_errors(loaded, ms)
        lines.append(f"| {ms} | " + " | ".join(str(wrong[c]) for c in CELLS) + f" | {sum(wrong.values())} |")
    return "\n".join(lines)



# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #


def _fingerprints(root: pathlib.Path) -> dict:
    """SHA-256 of everything the set consists of. Text files are hashed with
    line endings normalized, as in probes.py, so a checkout on Windows verifies."""
    root = pathlib.Path(root)
    script = read_script(root)
    return {
        "files": {name: sha256_file(root / name, text=True)
                  for name in ("agent.tsv", "script.tsv", AGENT_WORDS, CALIBRATION)},
        "agent": {a: sha256_file(root / "agent" / f"{a}.wav") for a in read_agent(root)},
        "items": {i: {"wav": sha256_file(root / "audio" / f"{i}.wav"),
                      "labels": sha256_file(root / "labels" / f"{i}.txt", text=True)} for i in script},
    }


def freeze(root: pathlib.Path = BARGEIN_ROOT) -> pathlib.Path:
    """Write manifest.json. Refuses if one exists, or if any item does not load."""
    root = pathlib.Path(root)
    out = root / MANIFEST
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen set is never re-frozen; make a new version")
    missing = [i for i in read_script(root) if not (root / "audio" / f"{i}.wav").exists()
               or not (root / "labels" / f"{i}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"not recorded and labelled yet: {', '.join(missing)}")
    for item_id in read_script(root):
        load_item(item_id, root)
    out.write_text(json.dumps({"set": "bargein", "version": 1,
                               "frozen": datetime.date.today().isoformat(),
                               **_fingerprints(root)}, indent=2) + "\n", encoding="utf-8")
    return out


def verify(root: pathlib.Path = BARGEIN_ROOT) -> list[str]:
    """One line per file that is missing or changed; empty means intact."""
    root = pathlib.Path(root)
    if not (root / MANIFEST).exists():
        return [f"no {MANIFEST}: the set is not frozen yet"]
    frozen = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    try:
        now = _fingerprints(root)
    except FileNotFoundError as e:
        return [f"missing: {e.filename}"]
    problems = [f"{name}: changed" for name, h in frozen["files"].items() if now["files"].get(name) != h]
    problems += [f"agent/{a}.wav: changed" for a, h in frozen["agent"].items() if now["agent"].get(a) != h]
    for i, hashes in frozen["items"].items():
        problems += [f"{i}: {part} changed" for part, h in hashes.items()
                     if now["items"].get(i, {}).get(part) != h]
    return problems


if __name__ == "__main__":
    import sys

    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "render":
        print("\n".join(render()))
    elif cmd == "calibrate":
        print(calibrate(agent_id=args[0] if args else "A3"))
    elif cmd == "propose":
        force = "--force" in args
        wanted = [a for a in args if a != "--force"] or [
            i for i in read_script() if (BARGEIN_ROOT / "audio" / f"{i}.wav").exists()]
        for item_id in wanted:
            try:
                print("drafted", propose_label(item_id, overwrite=force))
            except FileExistsError:
                print(f"kept    {BARGEIN_ROOT / 'labels' / (item_id + '.txt')} (already there)")
    elif cmd == "check":
        print(check())
    elif cmd == "freeze":
        print("frozen ->", freeze())
    elif cmd == "verify":
        problems = verify()
        print("\n".join(problems) if problems else "intact")
        sys.exit(1 if problems else 0)
    else:
        sys.exit("usage: python -m s2slab.bargein render | calibrate [A3] | propose [ids] [--force] "
                 "| check | freeze | verify")