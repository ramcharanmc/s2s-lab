"""
Tests for s2slab.watermark and s2slab.watermarkdev (M43): the wrapper around AudioSeal, with stand-ins for its
networks; the development check on a synthetic voice and a stand-in watermark; the rule written before the check.
The one test with AudioSeal's real models is slow, and skips where AudioSeal is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from s2slab import severity, watermarkdev
from s2slab.watermark import MARK_RATE, MESSAGE, Watermarker, level_db, to_rate

SR = 24_000


def voice(groups=3, syll=4, syll_s=0.2, pause_s=0.35, lead_s=0.32, f0=190.0, sr=SR) -> np.ndarray:
    """A harmonic voice with pauses between syllable groups, after Kokoro's 0.32 s of leading silence."""
    out, phase, t0 = [np.zeros(int(lead_s * sr))], 0.0, 0.0
    for g in range(groups):
        for _ in range(syll):
            n = int(syll_s * sr)
            f = f0 + 30 * np.sin(2 * np.pi * 0.7 * (t0 + np.arange(n) / sr))
            ph = phase + np.cumsum(2 * np.pi * f / sr)
            phase = ph[-1]
            out.append(0.15 * np.sin(np.pi * np.arange(n) / n) ** 0.5 * sum(np.sin(k * ph) / k for k in range(1, 12)))
            t0 += syll_s
        if g < groups - 1:
            out.append(np.zeros(int(pause_s * sr)))
            t0 += pause_s
    return np.concatenate(out).astype(np.float32)


TONE_HZ, TONE_AMP = 100.0, 1e-3          # a stand-in mark: a quiet 100 Hz tone, under the rungs' 40 dB


def tone(n: int, sr: int = SR) -> np.ndarray:
    return (TONE_AMP * np.sin(2 * np.pi * TONE_HZ * np.arange(n) / sr)).astype(np.float32)


def tone_level(x, sr: int = SR) -> float:
    """The 100 Hz tone's amplitude in x, by projection (the voices here have nothing at 100 Hz)."""
    x = np.asarray(x, np.float64)
    t = np.arange(len(x)) / sr
    return 2 * float(np.hypot(np.mean(x * np.sin(2 * np.pi * TONE_HZ * t)), np.mean(x * np.cos(2 * np.pi * TONE_HZ * t))))


class FakeNets:
    """AudioSeal's two networks' shape: get_watermark(x, message) -> the mark; detect_watermark(x) -> (share, bits)."""

    def get_watermark(self, x, message=None):
        import torch
        assert x.ndim == 3 and x.shape[:2] == (1, 1) and message.shape == (1, 16)
        return torch.from_numpy(tone(x.shape[-1])).view(1, 1, -1)

    def detect_watermark(self, x):
        import torch
        share = 1.0 if tone_level(x.view(-1).numpy()) > TONE_AMP / 2 else 0.0
        bits = torch.tensor([MESSAGE if share else [1 - b for b in MESSAGE]])
        return torch.tensor([share]), bits


class FakeMark:
    """The same stand-in mark, at the level the check calls it: embed(audio, sr) and detect(audio, sr)."""

    def embed(self, audio, sr):
        assert sr == MARK_RATE
        return (np.asarray(audio, np.float32) + tone(len(audio))).astype(np.float32)

    def detect(self, audio, sr):
        found = tone_level(to_rate(audio, sr)) > TONE_AMP / 2
        return {"share": 1.0 if found else 0.0, "bits": 16 if found else 0, "marked": found}


def test_the_wrapper_marks_at_24_khz_and_reads_any_rate_at_24():
    pytest.importorskip("torch")
    wm = Watermarker(generator=FakeNets(), detector=FakeNets())
    x = voice()
    m = wm.embed(x, SR)
    assert m.dtype == np.float32 and len(m) == len(x) and np.allclose(m - x, tone(len(x)), atol=1e-6)
    assert wm.detect(m, SR) == {"share": 1.0, "bits": 16, "marked": True}
    assert wm.detect(to_rate(m, SR, 16_000), 16_000)["marked"]                   # resampled to 24 kHz to be read
    assert wm.detect(x, SR) == {"share": 0.0, "bits": 0, "marked": False}
    with pytest.raises(ValueError, match="24000 Hz"):
        wm.embed(x, 16_000)
    assert wm.embed(np.zeros(0, np.float32), SR).size == 0 and not wm.detect(np.zeros(0), SR)["marked"]
    assert np.max(wm.embed(np.full(10, 0.9999, np.float32), SR)) <= 1.0          # clipped
    assert level_db(x, m) < -30 and level_db(x, x) == float("-inf")


def test_the_check_on_a_synthetic_voice_says_go_and_each_failure_says_stop(monkeypatch):
    monkeypatch.setattr(severity, "DEV_VOICES", ("v1",))
    monkeypatch.setattr(severity, "DEV_SENTENCES", {"d1": "one sentence. And another."})
    monkeypatch.setattr(severity, "SCORES", {"depression": (0, 15), "mania": (0, 26)})
    noise = np.random.default_rng(0).normal(0, 0.05, 32_000).astype(np.float32)
    blob = watermarkdev.run(lambda text, v: voice(), FakeMark(), [("eval-0", noise, 16_000)], opus=False)
    assert [r["id"] for r in blob["renders"]] == ["v1-d1-depression-0", "v1-d1-depression-15", "v1-d1-mania-0",
                                                  "v1-d1-mania-26"]
    g = watermarkdev.gate(blob)
    assert g["go"], g["checks"]
    assert g["silence_ms"] < 20 and all(v < 0.01 for k, v in g["worst"].items() if k != "f0_offset_hz")
    text = watermarkdev.format_dev(blob)
    assert "GO: wire the mark in" in text and "found              PASS" in text

    lost = {**blob, "renders": [dict(blob["renders"][0], via16k={"share": 0.0, "bits": 3, "marked": False})] + blob["renders"][1:]}
    assert not watermarkdev.gate(lost)["go"] and watermarkdev.gate(lost)["lost"] == ["v1-d1-depression-0"]
    alarm = {**blob, "controls": [{"id": "eval-0", "share": 0.6, "bits": 9, "marked": True}]}
    assert watermarkdev.gate(alarm)["alarms"] == ["eval-0"] and not watermarkdev.gate(alarm)["go"]
    r0 = blob["renders"][1]
    shifted = dict(r0, marked_measures=dict(r0["marked_measures"], f0_offset_hz=r0["plain"]["f0_offset_hz"] + 0.31))
    assert not watermarkdev.gate({**blob, "renders": [shifted]})["checks"]["render unchanged"][0]
    slow = dict(r0, marked_measures=dict(r0["marked_measures"], rate=r0["plain"]["rate"] * 1.011))
    assert not watermarkdev.gate({**blob, "renders": [slow]})["checks"]["render unchanged"][0]
    late = {**blob, "neutrals": [dict(blob["neutrals"][0], silence_ms=[320.0, 340.0])]}
    assert not watermarkdev.gate(late)["checks"]["silence unchanged"][0]


def test_a_16_bit_wav_is_what_its_samples_read_back_as():
    x = np.array([0.0, 0.5, -1.0, 1.2], np.float32)
    assert np.allclose(watermarkdev.quantize16(x), [0.0, 16384 / 32767, -1.0, 1.0], atol=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("audioseal") is None, reason="AudioSeal is not installed")
def test_audioseals_real_mark_is_found_and_absent_where_it_was_not_made():
    wm = Watermarker().load()
    x = voice()
    m = wm.embed(x, SR)
    assert wm.detect(m, SR)["marked"] and wm.detect(m, SR)["bits"] == 16
    assert wm.detect(to_rate(m, SR, 16_000), 16_000)["marked"] and not wm.detect(x, SR)["marked"]
