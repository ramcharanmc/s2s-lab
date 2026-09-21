"""
A first look at the prosody probe set -- DEV PAIRS ONLY, descriptive, no claim tested.

Two questions, asked before anything is built on the set:

  1. IS THE DIFFERENCE IN THE SOUND?  Raters heard it; can three plain numbers see it?
     For each pair, loaded minus neutral:
         pitch    median F0, in semitones            (12 * log2(loaded / neutral))
         range    F0 range, 10th to 90th percentile, in semitones
         level    how loud the loud parts are: the 90th-percentile 10 ms frame, in dB
     The same actor says the same sentence, so everything else -- voice, sex, age,
     words, microphone -- cancels inside the pair. That is what pairing is for.

     NOT MEASURED: SPEAKING RATE. Two ways of timing the spoken part were tried on
     the dev pairs, and both turned a difference in LEVEL into a difference in
     LENGTH: a threshold hung from the loudest frame swallows a quiet clip whole,
     and one standing on the noise floor trims a quiet clip's edges. A loaded
     clip is 10 to 15 dB louder than its twin, so neither can be trusted here.
     Rate needs word times from a recognizer, not an energy threshold.

  2. IS THE DIFFERENCE IN THE WORDS?  It must not be, or the set proves nothing.
     Both clips go through R1's own recognizer. If the transcripts are the same,
     R1's LLM is asked the same thing twice and, decoding greedily, says the same
     thing twice: the delivery is gone before the LLM sees anything. Where the
     transcripts differ, the difference is a recognition error, and it is listed.

The 39 test pairs are never loaded here: load("dev") only.

    python -m s2slab.prosodycheck acoustics     Praat only, a few seconds
    python -m s2slab.prosodycheck blind         loads Whisper base.en, about a minute
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .probes import TARGET_SR
from .prosodyset import EMOTIONS, LoadedPair

FRAME_S = 0.010
PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 75, 600    # NOT the 60-400 Hz that serves endpointing: fear and anger go above 400
MIN_VOICED_FRAMES = 10                        # under 0.1 s of pitch, F0 statistics are not reported
SENTENCES = {
    "IEO": "It's eleven o'clock", "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting", "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full", "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock", "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket", "ITS": "I think I've seen this before",
    "TSI": "The surface is slick", "WSI": "We'll stop in a couple of minutes",
}
MEASURES = (("pitch_st", "median pitch, semitones"), ("range_st", "pitch range, semitones"),
            ("level_db", "level, dB"))


class Features(NamedTuple):
    f0_median_hz: float
    f0_range_st: float
    level_db: float       # dB re full scale


class Delta(NamedTuple):
    pitch_st: float
    range_st: float
    level_db: float


def pitch_hz(audio: np.ndarray) -> np.ndarray:
    """F0 of the voiced 10 ms frames, in Hz (Praat, PITCH_FLOOR_HZ to PITCH_CEILING_HZ)."""
    import parselmouth
    sound = parselmouth.Sound(audio.astype("float64"), sampling_frequency=TARGET_SR)
    pitch = sound.to_pitch(time_step=FRAME_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
    f0 = pitch.selected_array["frequency"]
    return f0[f0 > 0]


def features(audio: np.ndarray) -> Features:
    """Three numbers from one clip. Takes mono float32 at 16 kHz. Calls
    pitch_hz(). F0 statistics are NaN when fewer than MIN_VOICED_FRAMES frames
    have pitch. The level is the 90th-percentile frame, so the silence around
    the speech does not move it -- as long as speech fills more than a tenth of
    the clip, which it does in every clip of this set (1.5 to 4.1 s each)."""
    voiced = pitch_hz(audio)
    if voiced.size >= MIN_VOICED_FRAMES:
        lo, hi = np.percentile(voiced, [10, 90])
        f0_median, f0_range = float(np.median(voiced)), 12 * math.log2(hi / lo)
    else:
        f0_median = f0_range = float("nan")
    n = int(FRAME_S * TARGET_SR)
    frames = audio[: len(audio) // n * n].reshape(-1, n).astype(np.float64)
    level = float(np.percentile(10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12), 90))
    return Features(f0_median, f0_range, level)


def delta(pair: LoadedPair) -> Delta:
    """Loaded minus neutral, inside one pair."""
    n, l = features(pair.neutral), features(pair.loaded)
    return Delta(12 * math.log2(l.f0_median_hz / n.f0_median_hz), l.f0_range_st - n.f0_range_st,
                 l.level_db - n.level_db)


def describe(pairs: list[LoadedPair]) -> str:
    """Per emotion and measure: the median change over its pairs, and how many
    pairs moved the way the median did -- 7 of 7 is a pattern, 4 of 7 is not."""
    lines = ["| emotion | pairs | " + " | ".join(label for _, label in MEASURES) + " |",
             "|---|---:|" + "---|" * len(MEASURES)]
    for emotion in EMOTIONS:
        deltas = [delta(p) for p in pairs if p.pair.emotion == emotion]
        if not deltas:
            continue
        cells = []
        for name, _ in MEASURES:
            values = np.array([getattr(d, name) for d in deltas])
            values = values[~np.isnan(values)]
            med = float(np.median(values))
            same_way = int(np.sum(np.sign(values) == np.sign(med)))
            cells.append(f"{med:+.1f} ({same_way} of {len(values)})")
        lines.append(f"| {emotion} | {len(deltas)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def blind(pairs: list[LoadedPair], recognizer, normalizer=None) -> str:
    """What a cascade's LLM would be given for each clip of each pair.

    Takes:   loaded pairs; a LOADED recognizer with transcribe(audio) -> str (R1's
             is rungs.get("r1").asr); the normalizer (default: wer.standard).
    Returns: per emotion, how many pairs have the same transcript for both clips
             and the word error rate of each side against the sentence that was
             read; then every pair whose transcripts differ, with both.
    """
    from .wer import ZERO, wer
    if normalizer is None:
        from .wer import standard as normalizer
    lines = ["| emotion | pairs | same transcript for both clips | WER, neutral clips | WER, loaded clips |",
             "|---|---:|---:|---:|---:|"]
    differing = []
    for emotion in EMOTIONS:
        mine = [p for p in pairs if p.pair.emotion == emotion]
        if not mine:
            continue
        same, err_n, err_l = 0, ZERO, ZERO
        for p in mine:
            heard_n, heard_l = recognizer.transcribe(p.neutral), recognizer.transcribe(p.loaded)
            ref = SENTENCES[p.pair.sentence]
            err_n, err_l = err_n + wer(ref, heard_n, normalizer), err_l + wer(ref, heard_l, normalizer)
            if normalizer(heard_n) == normalizer(heard_l):
                same += 1
            else:
                differing.append(f'  {p.pair.id} "{ref}": neutral heard as {heard_n!r}, loaded as {heard_l!r}')
        lines.append(f"| {emotion} | {len(mine)} | {same} | {100 * err_n.rate:.1f}% | {100 * err_l.rate:.1f}% |")
    lines += ["", "pairs whose transcripts differ (a recognition error, not a report of emotion):"]
    return "\n".join(lines + (differing or ["  none"]))


if __name__ == "__main__":
    import sys

    from .prosodyset import load

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    if cmd == "acoustics":
        pairs = load("dev")
        print(f"{len(pairs)} dev pairs, loaded minus neutral; median change (pairs that moved the same way)\n")
        print(describe(pairs))
    elif cmd == "blind":
        from .rungs import get
        asr = get("r1").asr
        asr.load()
        pairs = load("dev")
        print(f"{len(pairs)} dev pairs through R1's recognizer ({asr.model_name})\n")
        print(blind(pairs, asr))
    else:
        sys.exit("usage: python -m s2slab.prosodycheck acoustics | blind")
