"""
Tests for s2slab.realturns: real clips as endpointing turns (PR-014). Synthetic audio stands in for the
recordings; the tests on the real DEV recordings are skipped where the audio is not on the machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.probes import PROBE_ROOT
from s2slab.realturns import PAD_S, asr_words, dev_turns, floor_rms, load_eval, pad, turn
from s2slab.vad import frame_speech_prob

SR = 16_000


def _clip(seed: int = 0) -> np.ndarray:
    """Room noise, then a voiced 1 s burst, a 0.8 s pause, a voiced 1.2 s burst, and nothing after it."""
    rng = np.random.default_rng(seed)
    t = lambda s: np.arange(int(s * SR)) / SR
    noise = lambda s: rng.normal(0, 0.002, int(s * SR)).astype("float32")
    tone = lambda s, f: (0.3 * np.sin(2 * np.pi * f * t(s)) * np.hanning(int(s * SR))).astype("float32")
    return np.concatenate([noise(0.6), tone(1.0, 160) + noise(1.0), noise(0.8), tone(1.2, 150) + noise(1.2)])


def test_padding_is_noise_at_the_clips_own_floor_and_repeatable():
    x = _clip()
    y, z, w = pad(x, PAD_S, "a"), pad(x, PAD_S, "a"), pad(x, PAD_S, "b")
    assert len(y) == len(x) + int(PAD_S * SR)
    assert np.array_equal(y, z) and not np.array_equal(y, w)
    assert np.std(y[len(x):]) == pytest.approx(floor_rms(x), rel=0.1)


def test_a_turn_is_cropped_before_its_speech_ends_at_its_last_and_its_padding_reads_as_silence():
    from s2slab.realturns import speech_spans
    x = _clip()
    spans = speech_spans(x)
    t0 = spans[0][0] - 0.1                                              # cropped 100 ms before the first speech
    u = turn(x, "clean", "c-00", [("hello", 1.5), ("there", 3.5)])
    assert len(spans) == 2 and 3.2 < spans[-1][1] <= 3.6                # inside the second burst's faded tail
    assert u.true_end_s == pytest.approx(spans[-1][1] - t0)
    assert len(u.audio) / SR == pytest.approx(u.true_end_s + PAD_S, abs=0.03)
    assert [w for w, _ in u.words] == ["hello", "there"] and u.words[0][1] == pytest.approx(1.5 - t0)
    p = frame_speech_prob(u.audio, SR)
    assert max(p[int((u.true_end_s + 0.1) * 50):]) < 0.5
    assert len(u.mid_turn_pauses_s) == 1 and u.mid_turn_pauses_s[0] == pytest.approx(0.8, abs=0.15)


def test_the_eval_turns_are_refused_without_the_registration():
    with pytest.raises(PermissionError, match="PR-014"):
        load_eval()


def test_recognizer_words_are_read_as_t1_reads_them():
    class W:
        def __init__(self, word, end):
            self.word, self.end = word, end

    class Seg:
        words = [W(" My", 0.4), W(" account,", 0.9), W(" ...", 1.0), W(" number.", 1.6)]

    class Model:
        def transcribe(self, audio, **kw):
            assert kw["word_timestamps"] is True
            return [Seg()], None
    assert asr_words(np.zeros(SR, dtype="float32"), Model()) == [("my", 0.4), ("account", 0.9), ("number", 1.6)]


@pytest.mark.skipif(not (PROBE_ROOT / "audio" / "f01.wav").exists(), reason="the DEV recordings are not in git")
def test_the_dev_turns_keep_their_labelled_ends_and_are_padded_after_them():
    from s2slab.probes import load_item
    turns = dev_turns()["dev"]
    assert len(turns) == 30 and {u.cohort for u in turns} == {"dev"}
    u, ref = turns[0], load_item("f01")
    assert u.true_end_s == ref.true_end_s
    assert len(u.audio) / SR == pytest.approx(u.true_end_s + PAD_S, abs=0.02)
    assert max(frame_speech_prob(u.audio, SR)[int((u.true_end_s + 0.1) * 50):]) < 0.5
