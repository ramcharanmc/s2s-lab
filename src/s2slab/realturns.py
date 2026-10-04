"""
Real speech as endpointing turns (M30, PR-014): the frozen eval set's 200 clips, and the 30 own-voice
recordings as DEV, built as the simulator's own `Utterance`, so every policy and `sweep.evaluate` run
on them unchanged.

    PYTHONPATH=src python -m s2slab.realturns dev        T0, T1 and T1b on the DEV recordings

WHAT A REAL TURN NEEDS THAT A CLIP DOES NOT HAVE
  * AN END. The DEV recordings have labels (probes.load_item): the end of the last labelled chunk.
    The eval clips have none, so their end is the end of the last stretch of speech the policies' own
    VAD finds, voiced for at least probes.MIN_VOICED_S (probes.detect_chunks, as the labels were
    drafted). The clips are corpus segments cut close to the last word, so nothing after the speech
    can be mistaken for more of it; on DEV this rule meets the labels within 100 ms on 26 of 30.
  * SILENCE AFTER IT. The accented and disfluent clips end within 20 ms of the last word. Every turn
    is padded to PAD_S after its end with noise at the clip's own floor (its 10th-percentile frame
    level, the floor the VAD measures), never digital silence: zeros would drag the VAD's floor down
    and make the room noise inside the clip read as speech. Seeded by the clip's id.
  * A START. As in probes.load_item: cropped to LEAD_KEEP_S before the first speech, because every
    policy counts silence from the start of the buffer.
  * WORDS. The eval clips' words come from a recognizer with word times (faster-whisper base.en, R0's,
    int8 on the CPU), lower-cased and stripped of punctuation as T1 reads them. It hears the whole clip,
    so its words are a little better than a streaming recognizer's would be; the same for T1 and T1b.

The eval turns are PR-014's confirmation's: load_eval refuses without confirm="PR-014".
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from .probes import LEAD_KEEP_S, MIN_VOICED_S, PROBE_ROOT, TARGET_SR, _pitch_track, detect_chunks, load_item, \
    read_script, _voiced_s
from .vad import frame_rms

PAD_S = 2.5                 # after the end: room for the slowest decision (T2 1,050 ms; VAP's fallback 1,500 ms)
REGISTRATION = "PR-014"
ASR_MODEL = "base.en"       # R0's recognizer


def floor_rms(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """The clip's noise floor as the VAD measures it: the 10th percentile of frame RMS, as a level."""
    rms = frame_rms(audio, sr)
    return float(np.percentile(rms, 10)) if rms.size else 0.0


def pad(audio: np.ndarray, seconds: float, key: str, sr: int = TARGET_SR) -> np.ndarray:
    """`audio` followed by `seconds` of Gaussian noise at its own floor, seeded by `key`."""
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    noise = np.random.default_rng(seed).normal(0.0, floor_rms(audio, sr), int(round(seconds * sr)))
    return np.concatenate([audio, noise.astype("float32")])


def speech_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    """Stretches of speech by the policies' VAD, each with some voice in it (as probes drafts labels)."""
    track = _pitch_track(audio)
    return [s for s in detect_chunks(audio) if _voiced_s(track, *s) >= MIN_VOICED_S]


def asr_words(audio: np.ndarray, model=None) -> list[tuple[str, float]]:
    """(word, end_s) for every word the recognizer writes, as T1 reads words."""
    if model is None:
        from faster_whisper import WhisperModel
        model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language="en", beam_size=1, word_timestamps=True,
                                   condition_on_previous_text=False)
    out = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip().lower().strip(".,!?;:\"'")
            if text:
                out.append((text, float(w.end)))
    return out


def turn(audio: np.ndarray, cohort: str, key: str, words: list[tuple[str, float]],
         spans: list[tuple[float, float]] | None = None):
    """One clip as an Utterance: cropped before its first speech, ended at its last, padded after."""
    from .prosody import count_syllable_nuclei
    from .simulate import Utterance
    spans = speech_spans(audio) if spans is None else spans
    if not spans:
        raise ValueError(f"{key}: no speech found")
    t0 = max(0.0, spans[0][0] - LEAD_KEEP_S)
    x = audio[int(round(t0 * TARGET_SR)):]
    end = spans[-1][1] - t0
    x = x[: int(round(end * TARGET_SR))]                       # nothing after the end but our own floor noise
    x = pad(x, PAD_S, key)
    return Utterance(
        audio=x, sr=TARGET_SR, true_end_s=end, cohort=cohort,
        words=[(w, e - t0) for w, e in words if e - t0 > 0],
        n_syllables=count_syllable_nuclei(x, TARGET_SR),
        mid_turn_pauses_s=[b[0] - a[1] for a, b in zip(spans, spans[1:])],
        f0_track=_pitch_track(x),
    )


def dev_turns(root: pathlib.Path = PROBE_ROOT) -> dict[str, list]:
    """The 30 own-voice recordings by their labels (probes.load_item), padded to PAD_S after the end.
    One cohort, "dev"; their words are the labels' (the recordings' words are known exactly)."""
    return {"dev": [u for _, u in dev_items(root)]}


def dev_items(root: pathlib.Path = PROBE_ROOT) -> list:
    """dev_turns' turns as (item id, Utterance), in script order."""
    from .simulate import Utterance
    out = []
    for item_id in read_script(root / "script.tsv"):
        u = load_item(item_id, root)
        keep = u.audio[: int(round(u.true_end_s * TARGET_SR))]
        x = pad(keep, PAD_S, item_id)
        out.append((item_id, Utterance(audio=x, sr=TARGET_SR, true_end_s=u.true_end_s, cohort="dev", words=u.words,
                                       n_syllables=u.n_syllables, mid_turn_pauses_s=u.mid_turn_pauses_s,
                                       f0_track=_pitch_track(x))))
    return out


def load_eval(confirm: str = "", words_fn=asr_words) -> dict[str, list]:
    """The 200 eval clips as turns, by cohort. PR-014's confirmation's: refused without confirm."""
    out: dict[str, list] = {}
    for _, u in eval_items(confirm, words_fn):
        out.setdefault(u.cohort, []).append(u)
    return out


def eval_items(confirm: str = "", words_fn=asr_words) -> list:
    """load_eval's turns as (clip id, Utterance), in manifest order."""
    if confirm != REGISTRATION:
        raise PermissionError(f"the eval clips' turns are {REGISTRATION}'s confirmation's: pass confirm=\"{REGISTRATION}\"")
    from .evalset import load, verify
    problems = verify()
    if problems:
        raise RuntimeError("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    model = None
    if words_fn is asr_words:
        from faster_whisper import WhisperModel
        model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    out = []
    for it in load():
        words = words_fn(it.audio, model) if model is not None else words_fn(it.audio)
        out.append((it.id, turn(it.audio, it.cohort, it.id, words)))
    return out


def dev_report(policies: list) -> dict:
    """Every policy on the DEV turns: cut-offs, waits, and which items it cut off."""
    from .sweep import evaluate
    turns = dev_turns()["dev"]
    ids = list(read_script(PROBE_ROOT / "script.tsv"))
    rows = []
    for p in policies:
        r = evaluate(p, turns)
        rows.append({"policy": p.name, "n": r.n, "cut": r.n_cutoff, "no_fire": r.n_never_fired,
                     "p50": r.latency.p50, "p90": r.latency.p90,
                     "cut_items": [i for i, c in zip(ids, r.cut_mask) if c]})
    return {"n": len(turns), "rows": rows}


DEV_BASELINES = pathlib.Path("runs/turn-taking/pr014-dev-baselines.json")


def format_dev(blob: dict) -> str:
    lines = [f"DEV: the {blob['n']} own-voice recordings, labelled ends, padded {PAD_S} s with their own floor noise", "",
             "| policy | cut off | median wait ms | p90 ms | items cut off |", "|---|---:|---:|---:|---|"]
    for r in blob["rows"]:
        p90 = "—" if r["p90"] != r["p90"] else f"{r['p90']:.0f}"
        lines.append(f"| {r['policy']} | {r['cut']} of {r['n']} | {r['p50']:.0f} | {p90} | {', '.join(r['cut_items'])} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    from .completion import t1b
    from .policies import get
    from .sweep import _git_commit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["dev"])
    ap.add_argument("--code", default="", help="the code the run used, when it is not a commit yet")
    a = ap.parse_args(argv)
    if DEV_BASELINES.exists():
        raise SystemExit(f"{DEV_BASELINES} exists: the DEV baselines are recorded once, before any VAP number")
    pols = [get(n) for n in ("T0_fixed_500ms", "T0_fixed_700ms", "T0_fixed_900ms", "T1_semantic_500ms")] + [t1b()]
    blob = {"commit": a.code or _git_commit(), **dev_report(pols)}
    print(format_dev(blob))
    DEV_BASELINES.parent.mkdir(parents=True, exist_ok=True)
    DEV_BASELINES.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved -> {DEV_BASELINES}")


if __name__ == "__main__":
    main()
