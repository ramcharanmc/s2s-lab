"""
Tests for s2slab.r4profile -- R4's stages timed, with a fake Omni model built like the real one:
a thinker, a talker, and a token2wav with a DiT and a BigVGAN inside it. No GPU, no weights.
"""

from __future__ import annotations

import importlib.util
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

from s2slab.r4 import OUT_SR
from s2slab.r4profile import (BLOCKS, PROFILE, StageClock, estimates, format_profile, profile_turn, run, save,
                              share, summary)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPC = 480                                   # the fake's samples per code: 50 codes a second at 24 kHz


class Part:
    def __init__(self, fn):
        self.forward = fn

    def __call__(self, *a, **k):            # as nn.Module: __call__ goes through self.forward
        return self.forward(*a, **k)


class FakeT2W(Part):
    def __init__(self, delay_s):
        self.code2wav_dit_model = SimpleNamespace(sample=lambda cond, ref, code, **k: (time.sleep(delay_s), np.ones(2 * code.shape[-1]))[1])
        self.code2wav_bigvgan_model = Part(lambda mel: np.full(len(mel) * SPC // 2, 0.1, dtype=np.float32))
        super().__init__(self._forward)
        self.calls = []

    def _forward(self, code, conditioning=None, reference_mel=None, **kwargs):
        self.calls.append((code.shape[-1], conditioning, reference_mel))
        return self.code2wav_bigvgan_model(self.code2wav_dit_model.sample(conditioning, reference_mel, code))


class FakeOmni:
    """speak() calls the three stages the way Qwen2.5-Omni's generate does."""

    def __init__(self, n_codes=120, n_text=12, delay_s=0.002):
        self.n_codes, self.n_text, self.delay_s, self.model, self.loads = n_codes, n_text, delay_s, None, 0

    def load(self):
        self.loads += 1
        d = self.delay_s
        thinker = SimpleNamespace(generate=lambda input_ids, **k: (time.sleep(d), SimpleNamespace(
            sequences=np.zeros((1, input_ids.shape[1] + self.n_text))))[1])
        talker = SimpleNamespace(generate=lambda **k: (time.sleep(d), np.zeros((1, self.n_codes)))[1])
        self.model = SimpleNamespace(thinker=thinker, talker=talker, token2wav=FakeT2W(d))

    def speak(self, audio, sr):
        self.model.thinker.generate(input_ids=np.zeros((1, 30)), output_hidden_states=True)
        codes = self.model.talker.generate(input_ids=np.zeros((1, 3)))
        wav = self.model.token2wav(codes, conditioning="cond", reference_mel="ref")
        return "Sure, I can help.", wav


def loaded(**kw) -> FakeOmni:
    omni = FakeOmni(**kw)
    omni.load()
    return omni


def test_the_clock_times_each_stage_and_leaves_the_reply_as_it_was():
    plain = loaded().speak(np.zeros(16000), 16000)
    omni = loaded()
    syncs = []
    clock = StageClock(omni.model, sync=lambda: syncs.append(1))
    clock.start()
    text, wav = omni.speak(np.zeros(16000), 16000)
    ms = clock.stop()
    assert text == plain[0] and np.array_equal(wav, plain[1])
    assert set(ms) == {"thinker", "talker", "token2wav", "dit", "bigvgan"}
    assert ms["token2wav"] >= ms["dit"] >= 2 and ms["thinker"] >= 2 and ms["talker"] >= 2
    assert clock.text_tokens == 12 and clock.t2w_call[0].shape[-1] == 120
    assert clock.t2w_call[1] == {"conditioning": "cond", "reference_mel": "ref"}
    assert len(syncs) == 10                                     # before and after each of the five calls


def test_outside_a_turn_the_clock_neither_times_nor_keeps_anything():
    omni = loaded()
    clock = StageClock(omni.model, sync=lambda: None)
    omni.speak(np.zeros(16000), 16000)
    assert clock.ms == {} and clock.t2w_call is None and clock.text_tokens is None


def test_a_turn_records_its_stages_counts_and_first_blocks():
    omni = loaded(n_codes=60)
    clock = StageClock(omni.model, sync=lambda: None)
    rec = profile_turn(omni, clock, np.zeros(16000), 16000)
    assert rec["codes"] == 60 and rec["text_tokens"] == 12 and rec["words"] == 4
    assert rec["reply_s"] == pytest.approx(60 * SPC / OUT_SR)
    assert set(rec["block_ms"]) == {"25", "50", "100"} and rec["block_ms"]["100"] is None   # 60 codes: no block of 100
    assert set(rec["block_ms"]["25"]) == {"first", "again"} and rec["samples_per_code"] == SPC
    # the blocks are the reply's own first codes, decoded with the turn's own conditioning, twice each
    assert [c[0] for c in omni.model.token2wav.calls] == [60, 25, 25, 50, 50]
    assert all(c[1:] == ("cond", "ref") for c in omni.model.token2wav.calls)
    assert clock.t2w_call is None and set(rec["stages_ms"]) == {"thinker", "talker", "token2wav", "dit", "bigvgan"}


def test_a_first_block_that_fails_is_recorded_and_the_run_goes_on():
    omni = loaded(n_codes=60)
    clock = StageClock(omni.model, sync=lambda: None)
    real = omni.model.token2wav.forward

    def picky(code, **kw):
        if code.shape[-1] == 50:
            raise RuntimeError("no block of 50")
        return real(code, **kw)

    omni.model.token2wav.forward = picky
    rec = profile_turn(omni, clock, np.zeros(16000), 16000)
    assert rec["block_ms"]["50"] == {"error": "RuntimeError: no block of 50"} and "again" in rec["block_ms"]["25"]
    rec["estimates"] = estimates(rec)
    assert set(rec["estimates"]["blocks"]) == {"25"}
    blob = {"load_ms": 1.0, "blocks": list(BLOCKS), "turns": {"a": rec}}
    assert "a first block failed: RuntimeError: no block of 50" in format_profile(blob)
    assert summary(blob)["token2wav, first 50 codes ms"]["n"] == 0


def test_the_estimates_are_the_arithmetic_written_in_the_docstring():
    rec = {"stages_ms": {"thinker": 800.0, "talker": 3000.0, "token2wav": 2000.0}, "codes": 250, "reply_s": 5.0,
           "block_ms": {"25": {"first": 900.0, "again": 150.0}, "50": {"first": 300.0, "again": 250.0}, "100": None}}
    e = estimates(rec)
    assert e["codes_per_s"] == 50 and e["talker_rtf"] == 0.6 and e["token2wav_rtf"] == 0.4
    # block 25: 800 + 25 x 12 ms + 150 = 1,250 ms; 150 ms of decoding for 500 ms of audio: 0.6 + 0.3
    assert e["blocks"]["25"] == {"first_audio_ms": 1250.0, "stream_rtf": 0.9}
    assert e["blocks"]["50"] == {"first_audio_ms": 1650.0, "stream_rtf": 0.85}
    assert "100" not in e["blocks"]
    assert estimates({**rec, "codes": 0})["blocks"] == {}


def test_run_loads_once_warms_up_untimed_and_profiles_every_clip():
    omni = FakeOmni(n_codes=120)
    blob = run(omni, {"a": np.zeros(16000), "b": np.zeros(8000)}, 16000)
    assert omni.loads == 1 and list(blob["turns"]) == ["a", "b"] and blob["blocks"] == list(BLOCKS)
    assert omni.model.token2wav.calls[0][0] == 120                 # the throwaway turn
    assert all("estimates" in t and t["estimates"]["codes_per_s"] == 50 for t in blob["turns"].values())
    s = summary(blob)
    assert s["codes"] == {"n": 2, "p50": 120, "p90": 120} and s["est. first audio, block 100 ms"]["n"] == 2
    sh = share(blob)
    assert set(sh) == {"thinker", "talker", "token2wav", "the rest"} and sum(sh.values()) == pytest.approx(1, abs=0.01)
    text = format_profile(blob)
    assert "R4's stages on 2 clips" in text and "| token2wav ms |" in text and "not a streaming rung" in text


def test_a_saved_profile_is_never_overwritten(tmp_path):
    p = save(tmp_path / "p.json", {"turns": {}})
    with pytest.raises(FileExistsError):
        save(p, {"turns": {}})
    assert PROFILE.parts == ("runs", "r4-gpu", "profile-l40s.json")


def test_the_l40s_app_has_the_profile_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.stages, "remote") and hasattr(mod.profile, "info")
    assert mod.PROFILE_PATH == ROOT / PROFILE
