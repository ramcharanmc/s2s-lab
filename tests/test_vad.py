"""
Tests for s2slab.vad.

Two jobs here. Most tests pin the CONTRACT -- the shape, range and edge-case
behaviour every downstream policy relies on, and which must survive the swap to
Silero VAD unchanged. A final group pins the KNOWN WEAKNESSES of the energy
implementation, so that when the real VAD lands those tests fail loudly and
tell you exactly what improved. A limitation you have written a test for is a
documented scope boundary; one you have not is a bug waiting to be discovered
by a reviewer.
"""

from __future__ import annotations

import numpy as np
import pytest

from s2slab.vad import FRAME_MS, frame_rms, frame_signal, frame_speech_prob

SR = 16_000


def tone(seconds: float, amplitude: float = 0.3, hz: float = 200.0) -> np.ndarray:
    n = int(seconds * SR)
    return (amplitude * np.sin(2 * np.pi * hz * np.arange(n) / SR)).astype("float32")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype="float32")


def noise(seconds: float, amplitude: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(seconds * SR))).astype("float32")


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def test_frame_count_matches_duration():
    frames = frame_signal(tone(1.0), SR)          # 1 s / 20 ms = 50
    assert frames.shape == (50, int(SR * FRAME_MS / 1000))


def test_partial_trailing_frame_is_discarded_not_padded():
    """Zero-padding would invent a quiet final frame that reads as silence --
    and the last frame is exactly where an endpointer makes its decision."""
    frames = frame_signal(tone(1.0 + 0.013), SR)  # 13 ms of overhang
    assert frames.shape[0] == 50


def test_audio_shorter_than_one_frame_gives_no_frames():
    assert frame_signal(tone(0.005), SR).shape[0] == 0


def test_frame_signal_rejects_an_impossible_frame_length():
    with pytest.raises(ValueError):
        frame_signal(tone(1.0), SR, frame_ms=0.0)


def test_rms_is_higher_for_louder_audio():
    assert frame_rms(tone(0.5, 0.8), SR).mean() > frame_rms(tone(0.5, 0.05), SR).mean()


def test_rms_of_digital_silence_does_not_blow_up():
    """The +1e-12 guard: sqrt(0) is fine, but log10(0) downstream is not."""
    r = frame_rms(silence(0.5), SR)
    assert r.size > 0
    assert np.all(np.isfinite(r))


def test_rms_of_a_sine_is_close_to_amplitude_over_root_two():
    """Sanity-check the arithmetic against the analytic value."""
    assert frame_rms(tone(0.5, 0.4), SR).mean() == pytest.approx(0.4 / np.sqrt(2), rel=0.05)


# --------------------------------------------------------------------------- #
# The contract every policy depends on
# --------------------------------------------------------------------------- #


def test_probabilities_are_bounded():
    probs = frame_speech_prob(np.concatenate([silence(0.2), tone(0.5)]), SR)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_returns_one_probability_per_frame():
    audio = np.concatenate([silence(0.2), tone(0.5), silence(0.4)])
    assert len(frame_speech_prob(audio, SR)) == frame_signal(audio, SR).shape[0]


def test_returns_a_list_of_floats_not_an_array():
    """Policies index and slice this; keep it a plain list so a numpy bool
    cannot sneak into a branch and behave unexpectedly."""
    probs = frame_speech_prob(np.concatenate([silence(0.2), tone(0.3)]), SR)
    assert isinstance(probs, list)
    assert all(isinstance(p, float) for p in probs)


def test_too_short_returns_empty_list_not_an_error():
    """Callers must handle [] rather than assuming at least one frame."""
    assert frame_speech_prob(tone(0.005), SR) == []


def test_speech_scores_higher_than_the_silence_around_it():
    audio = np.concatenate([silence(0.3), tone(0.5), silence(0.3)])
    probs = np.array(frame_speech_prob(audio, SR))
    n = len(probs)
    lead, mid = probs[: n // 5], probs[2 * n // 5: 3 * n // 5]
    assert mid.mean() > lead.mean() + 0.5


def test_speech_region_lands_in_the_right_place():
    """0.2 s silence, 0.5 s tone, 0.4 s silence -> the active run should start
    near frame 10 and end near frame 35."""
    audio = np.concatenate([silence(0.2), tone(0.5), silence(0.4)])
    probs = np.array(frame_speech_prob(audio, SR))
    active = np.where(probs > 0.5)[0]
    assert active.size > 0
    assert active[0] == pytest.approx(10, abs=3)
    assert active[-1] == pytest.approx(34, abs=3)


# --------------------------------------------------------------------------- #
# Degenerate input -- the guard that stops silence reading as speech
# --------------------------------------------------------------------------- #


def test_pure_silence_is_never_speech():
    """Without the dynamic-range guard, floor == max, the ramp divides by ~0
    and the VAD reports confident speech for an empty room."""
    assert set(frame_speech_prob(silence(1.0), SR)) == {0.0}


def test_constant_tone_with_no_pauses_is_not_all_speech_by_accident():
    """A clip with no silence has no noise floor to estimate. The percentile
    lands inside speech, so the result is near-uniform -- documented, and the
    reason a real VAD is a model rather than a threshold."""
    probs = frame_speech_prob(tone(1.0), SR)
    assert len(probs) > 0
    assert max(probs) - min(probs) < 0.5


def test_dc_offset_is_not_mistaken_for_speech():
    dc = np.full(SR, 0.2, dtype="float32")
    probs = frame_speech_prob(dc, SR)
    assert max(probs) - min(probs) < 0.5


# --------------------------------------------------------------------------- #
# Level independence
# --------------------------------------------------------------------------- #


def test_gain_does_not_change_the_decision():
    """The noise floor is estimated per clip, so scaling the whole signal must
    not move the speech/silence boundary. Cohorts recorded at different levels
    have to stay comparable."""
    base = np.concatenate([silence(0.3), tone(0.5, 0.3), silence(0.3)])
    loud = base * 4.0
    quiet = base * 0.1
    mask = lambda a: [p > 0.5 for p in frame_speech_prob(a.astype("float32"), SR)]
    assert mask(base) == mask(loud) == mask(quiet)


# --------------------------------------------------------------------------- #
# Known weaknesses -- these SHOULD fail once Silero lands
# --------------------------------------------------------------------------- #


def test_known_weakness_quiet_speech_is_lost_once_a_room_has_noise():
    """The fairness mechanism, demonstrated rather than asserted.

    A quiet high-frequency burst (~-48 dB) stands in for /s/ or /f/. In CLEAN
    audio this VAD detects it fine. Add a realistic room floor (~-26 dB) and the
    percentile floor rises above the fricative, so it reads as silence while the
    vowel beside it still reads as speech.

    That is how a fixed-threshold endpointer cuts off a speaker mid-word on
    "...to access" -- and it lands hardest on quiet, breathy and disfluent
    speakers, in exactly the rooms where recordings are noisiest. When a
    model-based VAD replaces this, the fricative survives and this test flips.
    """
    fricative = noise(0.25, amplitude=0.004, seed=1)
    clean = np.concatenate([silence(0.3), tone(0.5, 0.3), fricative, silence(0.3)])
    room = noise(len(clean) / SR, amplitude=0.05, seed=3)[: len(clean)]
    noisy = (clean + room).astype("float32")

    per_frame = int(SR * FRAME_MS / 1000)
    n_lead = int(0.3 * SR) // per_frame
    n_tone = int(0.5 * SR) // per_frame
    probs = np.array(frame_speech_prob(noisy, SR))

    vowel = probs[n_lead + 2: n_lead + n_tone - 2]
    fric = probs[n_lead + n_tone + 2: n_lead + n_tone + 10]
    assert vowel.mean() > 0.4, "the vowel should still survive the noise"
    assert fric.mean() < 0.2, "fricative now survives — swap the VAD and update this test"


def test_known_weakness_probabilities_saturate_with_silent_padding():
    """Documents that this is a DETECTOR, not a loudness ranker.

    The floor is a percentile of frame energy. Digitally-silent padding sits at
    -120 dB, so every audible frame lands far above the 20 dB ramp and clips to
    1.0 -- a -48 dB fricative and a -13 dB vowel score identically.

    That matters because the decision-theoretic endpointing policy wants a
    graded posterior, and against clean synthetic audio it will not get one.
    Real recordings have a real noise floor, which is part of why this cannot
    be validated on synthetic input alone.
    """
    quiet = noise(0.3, amplitude=0.004, seed=4)
    loud = tone(0.3, 0.3)
    audio = np.concatenate([silence(0.3), quiet, loud, silence(0.3)])
    probs = np.array(frame_speech_prob(audio, SR))
    per_frame = int(SR * FRAME_MS / 1000)
    n_lead = int(0.3 * SR) // per_frame
    n_seg = int(0.3 * SR) // per_frame
    q = probs[n_lead + 2: n_lead + n_seg - 2]
    l = probs[n_lead + n_seg + 2: n_lead + 2 * n_seg - 2]
    assert q.mean() == pytest.approx(l.mean(), abs=0.05), \
        "the VAD now discriminates loudness — update this test"


def test_known_weakness_background_noise_raises_the_floor():
    """Add steady noise and the percentile floor rises with it, compressing the
    speech/silence contrast. A model-based VAD is far less sensitive to this."""
    clean = np.concatenate([silence(0.3), tone(0.5, 0.3), silence(0.3)])
    noisy = (clean + noise(len(clean) / SR, amplitude=0.05, seed=2)[: len(clean)]
             ).astype("float32")
    contrast = lambda a: max(frame_speech_prob(a, SR)) - min(frame_speech_prob(a, SR))
    assert contrast(noisy) < contrast(clean)
