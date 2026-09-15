"""
Tests for s2slab.r0 -- the naive cascade.

The fast tests use fake stages, so they need no model and no download. They
check the wiring, the timing and the text clean-up. The tests marked `slow`
load the real models; the default run skips them. Run them with

    python -m pytest -m slow tests/test_r0.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from s2slab.probes import PROBE_ROOT
from s2slab.r0 import (ASR_SR, ESPEAK_PATH_MAX, KokoroTTS, NaiveCascade, clean_for_speech,
                       espeak_path_problem)

OUT_SR = KokoroTTS.SR
ONE_SECOND = np.zeros(ASR_SR, dtype=np.float32)


class FakeASR:
    def __init__(self, text="i would like to change the delivery address", delay_s=0.0):
        self.text, self.delay_s = text, delay_s
        self.loads = self.calls = 0

    def load(self):
        self.loads += 1

    def transcribe(self, audio):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.text


class FakeLLM:
    def __init__(self, reply="**Sure!** What is the new address?", delay_s=0.0):
        self.reply_text, self.delay_s = reply, delay_s
        self.loads, self.heard = 0, []

    def load(self):
        self.loads += 1

    def reply(self, text):
        self.heard.append(text)
        time.sleep(self.delay_s)
        return self.reply_text


class FakeTTS:
    SR = OUT_SR

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.loads, self.said = 0, []

    def load(self):
        self.loads += 1

    def speak(self, text):
        self.said.append(text)
        time.sleep(self.delay_s)
        t = np.arange(OUT_SR // 2) / OUT_SR
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def fake_r0(asr_s=0.0, llm_s=0.0, tts_s=0.0) -> NaiveCascade:
    return NaiveCascade(FakeASR(delay_s=asr_s), FakeLLM(delay_s=llm_s), FakeTTS(delay_s=tts_s))


# --------------------------------------------------------------------------- #
# Wiring, with fake stages
# --------------------------------------------------------------------------- #


def test_r0_is_registered_and_builds_without_any_model_library():
    """get("r0") must work on a machine with nothing installed. A fresh
    interpreter, because a slow test may already have imported torch here."""
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import sys; from s2slab.rungs import get; r = get('r0'); "
            "print(type(r).__name__, [m for m in ('torch', 'transformers', "
            "'faster_whisper', 'kokoro') if m in sys.modules])")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "NaiveCascade []", proc.stderr


def test_a_turn_flows_from_asr_to_llm_to_tts():
    rung = fake_r0()
    turn = rung.respond(ONE_SECOND, ASR_SR)
    assert turn.transcript == "i would like to change the delivery address"
    assert rung.llm.heard[-1] == turn.transcript            # the LLM got the transcript
    assert turn.reply_text == "**Sure!** What is the new address?"
    assert rung.tts.said[-1] == "Sure! What is the new address?"   # TTS got it cleaned
    assert turn.sr == OUT_SR
    assert list(turn.stages_ms) == ["asr", "llm", "tts"]


def test_each_model_is_timed_in_its_own_stage():
    turn = fake_r0(asr_s=0.03, llm_s=0.02, tts_s=0.01).respond(ONE_SECOND, ASR_SR)
    assert turn.stages_ms["asr"] >= 30
    assert turn.stages_ms["llm"] >= 20
    assert turn.stages_ms["tts"] >= 10
    assert turn.untimed_ms < 5


def test_load_loads_every_model_once_and_warms_up_before_the_clock():
    rung = fake_r0(asr_s=0.05)
    assert rung.load() >= 50                     # the warm-up turn is part of loading
    assert (rung.asr.loads, rung.llm.loads, rung.tts.loads) == (1, 1, 1)
    assert rung.asr.calls == 1                   # the warm-up
    turn = rung.respond(ONE_SECOND, ASR_SR)
    assert rung.asr.calls == 2
    assert turn.total_ms < 100                   # one 50 ms ASR call, not two
    assert rung.load() == 0.0


def test_r0_refuses_audio_at_the_wrong_rate():
    with pytest.raises(ValueError, match="16000"):
        fake_r0().respond(np.zeros(8_000, dtype=np.float32), 8_000)


# --------------------------------------------------------------------------- #
# Text clean-up between the LLM and the TTS
# --------------------------------------------------------------------------- #


def test_markdown_symbols_are_not_read_aloud():
    assert clean_for_speech("**Sure!** Try `reset` or #3") == "Sure! Try reset or 3"


def test_list_markers_are_dropped_and_lines_joined():
    assert clean_for_speech("You need:\n- bread\n- milk\n2. eggs") == "You need: bread milk eggs"


def test_plain_speech_passes_through_unchanged():
    text = "Your appointment is at 2.30 on the 14th, and it costs 12.50."
    assert clean_for_speech(text) == text


def test_a_number_at_the_start_of_a_line_is_not_a_list_marker():
    assert clean_for_speech("2.5 hours is the wait.") == "2.5 hours is the wait."


# --------------------------------------------------------------------------- #
# espeak-ng's path limit: a loud error instead of a silent exit
# --------------------------------------------------------------------------- #


def test_a_short_espeak_path_is_fine():
    assert espeak_path_problem("/x" * 60, None) is None                  # 120 characters


def test_a_path_over_the_limit_is_reported_with_the_fix():
    long_path = "/" + "d" * ESPEAK_PATH_MAX                              # 160 characters
    message = espeak_path_problem(long_path, None)
    assert "160 characters" in message and "export ESPEAK_DATA_PATH" in message


def test_a_short_environment_variable_rescues_a_long_path():
    assert espeak_path_problem("/" + "d" * 200, "/tmp/espeak-ng-data") is None


def test_a_long_environment_variable_does_not():
    assert espeak_path_problem("/" + "d" * 200, "/" + "e" * 200) is not None


# --------------------------------------------------------------------------- #
# The real models: slow, skipped unless asked for with -m slow
# --------------------------------------------------------------------------- #

_F02 = PROBE_ROOT / "audio" / "f02.wav"
needs_models = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("faster_whisper", "transformers", "kokoro")),
    reason="R0's model libraries are not installed")
needs_f02 = pytest.mark.skipif(not _F02.exists(), reason="the probe recordings are not on this machine")


@pytest.fixture(scope="module")
def real_r0():
    """Loaded once for all the slow tests: loading is the slow part."""
    rung = NaiveCascade()
    rung.load()
    return rung


@pytest.fixture(scope="module")
def f02():
    from s2slab.probes import load_wav
    return load_wav(_F02)        # "the appointment is on | the fourteenth at half past two"


@pytest.mark.slow
@needs_models
@needs_f02
def test_real_whisper_hears_the_recording(real_r0, f02):
    assert "appointment" in real_r0.asr.transcribe(f02).lower()


@pytest.mark.slow
@needs_models
@needs_f02
def test_real_cascade_answers_in_speech(real_r0, f02):
    turn = real_r0.respond(f02, ASR_SR)
    assert turn.reply_text
    assert turn.sr == OUT_SR and turn.reply_s > 0.5
    assert np.isfinite(turn.leading_silence_ms)
    assert turn.untimed_ms < 50


@pytest.mark.slow
@needs_models
def test_real_llm_is_greedy_so_replies_repeat(real_r0):
    text = "i would like to change the delivery address for my order"
    assert real_r0.llm.reply(text) == real_r0.llm.reply(text)