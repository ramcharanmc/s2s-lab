"""
A synthetic barge-in set to DEVELOP policies on, so the frozen 20 stay a test.

The frozen set (probes/bargein) is scored once per registered claim. Anything
with a parameter to tune -- B1's word list, B2's thresholds -- needs somewhere
else to be tuned. This builds that somewhere:

    python -m s2slab.bargein_dev build                         about two minutes, once
    python -m s2slab.bargein_eval --root probes/bargein_dev    score policies on it, freely

Same layout and the same loader as the real set, so nothing downstream knows
the difference. 40 items, 10 per cell of the same 2 x 2, on 10 new agent lines:

    agent    Kokoro af_heart, as in the real set, with word times
    caller   Kokoro in FOUR OTHER voices, placed in a quiet room at an onset
             drawn around the end of the marked phrase (-200 to +450 ms: real
             callers anticipate as often as they wait)
    cough    two bursts of noise; laugh: a voice saying "ha ha ha"
    label    known by construction: where the rendered utterance starts and ends
    lag      zero -- nothing was recorded, so there is no overdub lag

WHAT IT IS NOT. A synthetic voice in a synthetic room is easier for a recognizer
than a person is; every line was written by someone who had read the test
set's script; and a TTS "ha ha ha" is speech, which a real laugh is not. It is
for finding bugs and setting parameters, not for claims. Claims are made on the
frozen set, under a registration.

Kokoro is not bit-identical between runs, so a second build is a different
set: build() refuses to overwrite, and only the two TSVs are committed.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .bargein import (CALIBRATION, loud_spans, phrase_span_s, read_agent_words, read_script,
                      render)
from .probes import TARGET_SR, Chunk, write_labels

DEV_ROOT = pathlib.Path("probes/bargein_dev")
SEED = 20260920
VOICES = ("am_adam", "am_michael", "af_bella", "af_sarah")     # any American voice but the agent's
ENTRY_S = (-0.20, 0.45)      # onset relative to the end of the marked phrase
ROOM_DB = -65.0              # the room's noise, re full scale
CALLER_PEAK = 0.2
TAIL_S = 3.0                 # silence kept after the agent and the caller have both finished
LAUGH_TEXT = "ha ha ha"



def kokoro_caller(text: str, voice: str, _tts=[]) -> np.ndarray:
    """Takes: what the caller says, and a Kokoro voice. Returns: 16 kHz audio."""
    from scipy.signal import resample_poly

    if not _tts:                                   # load once per process
        from .r0 import KokoroTTS
        _tts.append(KokoroTTS())
        _tts[0].load()
    parts = [np.asarray(r.audio, dtype=np.float32)
             for r in _tts[0].pipeline(text, voice=voice) if r.audio is not None]
    return resample_poly(np.concatenate(parts), 2, 3).astype(np.float32)      # 24 kHz -> 16 kHz


def cough(rng: np.random.Generator) -> np.ndarray:
    """Two short bursts of noise with a sharp attack and a fast decay: loud,
    broadband and without pitch, which is what makes a cough hard for a VAD."""
    def burst(seconds):
        n = int(seconds * TARGET_SR)
        return rng.standard_normal(n) * np.exp(-np.arange(n) / (0.04 * TARGET_SR))
    gap = np.zeros(int(rng.uniform(0.15, 0.30) * TARGET_SR))
    return np.concatenate([burst(0.18), gap, burst(0.14)]).astype(np.float32)


def place(sound: np.ndarray, voice_at_s: float, total_s: float,
          rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Put one caller sound into a quiet room so that its FIRST AUDIBLE SAMPLE is at voice_at_s.

    Takes:   the sound (TTS output carries silent padding at both ends), when
             its voice should start, and how long the take is.
    Returns: (the take, label start, label end) -- the label from loud_spans()
             on the sound alone, before the room is added.
    """
    if sound.size == 0 or np.abs(sound).max() < 1e-4:
        raise ValueError("the caller's sound is silent")
    spans = loud_spans(sound)
    first, last = spans[0][0], spans[-1][1]
    sound = sound * (CALLER_PEAK / np.abs(sound).max())
    take = (10 ** (ROOM_DB / 20) * rng.standard_normal(int(total_s * TARGET_SR))).astype(np.float32)
    at = int(round((voice_at_s - first) * TARGET_SR))
    if at < 0 or at + len(sound) > len(take):
        raise ValueError("the caller's sound does not fit inside the take")
    take[at: at + len(sound)] += sound
    return take, voice_at_s, voice_at_s + (last - first)



def build(root: pathlib.Path = DEV_ROOT, seed: int = SEED, agent_synth=None, caller_synth=kokoro_caller) -> list[str]:
    """Render the agent lines, then synthesize, place and label every item.

    Takes:   the dev folder (agent.tsv and script.tsv must be there); a seed for
             onsets, voices and coughs; the two synthesizers (tests pass fakes).
    Calls:   bargein.render() for the agent; caller_synth(text, voice) or
             cough() per item; place(); probes.write_labels().
    Returns: one line per item, for printing.
    Raises:  FileExistsError if anything was already built.
    """
    import soundfile as sf

    root = pathlib.Path(root)
    if (root / "audio").exists() and any((root / "audio").glob("*.wav")):
        raise FileExistsError(f"{root}/audio is already built; a second build would be a different "
                              "set. Delete agent/, audio/, labels/, agent_words.tsv and "
                              "calibration.json to start again.")
    script = read_script(root)
    render(root, **({"synth": agent_synth} if agent_synth else {}))
    (root / CALIBRATION).write_text(json.dumps({"agent": "none", "lag_ms": 0.0, "note":
                                                "synthetic: nothing was recorded"}, indent=2) + "\n",
                                    encoding="utf-8")
    words = read_agent_words(root)
    rng = np.random.default_rng(seed)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (item_id, item) in enumerate(script.items()):
        voice = VOICES[k % len(VOICES)]
        if item["text"] == "[cough]":
            sound, voice = cough(rng), "noise"
        else:
            sound = caller_synth(LAUGH_TEXT if item["kind"] == "noise" else item["text"], voice)
        after_end = phrase_span_s(words[item["agent"]], item["after"])[1]
        voice_at = after_end + rng.uniform(*ENTRY_S)
        agent_s = sf.info(str(root / "agent" / f"{item['agent']}.wav")).duration
        total_s = max(agent_s, voice_at + len(sound) / TARGET_SR) + TAIL_S
        take, start, end = place(sound, voice_at, total_s, rng)
        sf.write(str(root / "audio" / f"{item_id}.wav"), take, TARGET_SR, subtype="PCM_16")
        write_labels([Chunk(start, end, item["text"])], root / "labels" / f"{item_id}.txt")
        lines.append(f"{item_id} {item['cell']:15} {voice:11} onset {start:5.2f} s "
                     f"({(start - after_end) * 1000:+4.0f} ms), lasts {end - start:4.2f} s  {item['text']}")
    return lines


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["build"]:
        print("\n".join(build()))
    else:
        sys.exit("usage: python -m s2slab.bargein_dev build")