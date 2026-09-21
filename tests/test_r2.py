"""
Tests for s2slab.r2 -- the two-tier cascade.

R2 is R1 with the slow tier's recognizer, so the tests are about exactly that:
nothing but the recognizer may differ from R1, and what the slow tier hears is
what the reply is built on. Streaming itself is R1's and is tested in test_r1.
The slow test loads the real small.en (a first run downloads about 0.5 GB):

    python -m pytest -m slow tests/test_r2.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from s2slab.probes import PROBE_ROOT
from s2slab.r0 import ASR_SR, KokoroTTS, WhisperASR
from s2slab.r1 import StreamingQwenLLM
from s2slab.r2 import TIER_A, TIER_B, TIER_B_TURBO, TwoTierCascade
from s2slab.rungs import get

OUT_SR = KokoroTTS.SR
PAIRS = [("r1", "r2", TIER_B), ("r1", "r2-turbo", TIER_B_TURBO),
         ("r1-gpu", "r2-gpu", TIER_B), ("r1-gpu", "r2-gpu-turbo", TIER_B_TURBO)]


class FakeASR:
    def __init__(self, text):
        self.text = text

    def load(self):
        pass

    def transcribe(self, audio):
        return self.text


class FakeStreamingLLM:
    def __init__(self):
        self.heard = []

    def load(self):
        pass

    def reply(self, text):                    # R0's warm-up calls this
        return "Hello."

    def stream(self, text):
        self.heard.append(text)
        yield "It is on the fourteenth,"
        yield " at half past two."


class FakeTTS:
    SR = OUT_SR

    def load(self):
        pass

    def speak(self, text):
        return np.full(len(text.split()) * OUT_SR // 2, 0.1, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Registry: R2 is R1 with another recognizer, and nothing else
# --------------------------------------------------------------------------- #


def test_r2_is_registered_and_builds_without_any_model_library():
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import sys; from s2slab.rungs import get; r = get('r2'); "
            "print(type(r).__name__, type(r.llm).__name__, r.asr.model_name, [m for m in ('torch', "
            "'transformers', 'faster_whisper', 'kokoro') if m in sys.modules])")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "TwoTierCascade StreamingQwenLLM small.en []", proc.stderr


@pytest.mark.parametrize("r1_id, r2_id, tier_b", PAIRS)
def test_only_the_recognizers_size_differs_from_r1(r1_id, r2_id, tier_b):
    """The R2 rows are read against R1's, so a second difference would confound them."""
    r1, r2 = get(r1_id), get(r2_id)
    assert r2.rung_id == r2_id and tier_b in r2.description
    assert r2.asr.model_name == tier_b != r1.asr.model_name
    assert (r2.asr.device, r2.asr.compute_type, r2.asr.beam_size) == \
           (r1.asr.device, r1.asr.compute_type, r1.asr.beam_size)
    assert isinstance(r2.llm, StreamingQwenLLM)
    assert (r2.llm.model_name, r2.llm.device, r2.llm.max_new_tokens) == \
           (r1.llm.model_name, r1.llm.device, r1.llm.max_new_tokens)
    assert (r2.tts.voice, r2.tts.device) == (r1.tts.voice, r1.tts.device)


def test_the_fast_tier_is_the_recognizer_r1_and_the_barge_in_policies_use():
    assert TIER_A == WhisperASR().model_name == get("r1").asr.model_name


# --------------------------------------------------------------------------- #
# The slow tier's words are the ones the reply is built on
# --------------------------------------------------------------------------- #


def test_what_the_slow_tier_heard_is_what_the_llm_is_asked():
    rung = TwoTierCascade(FakeASR("the appointment is on the fourteenth at half past two"),
                          FakeStreamingLLM(), FakeTTS())
    rung.load()
    turn = rung.respond(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert turn.transcript == "the appointment is on the fourteenth at half past two"
    assert rung.llm.heard[-1] == turn.transcript
    assert turn.reply_text == "It is on the fourteenth, at half past two."


def test_r2_streams_as_r1_does_and_times_the_same_stages():
    rung = TwoTierCascade(FakeASR("hello"), FakeStreamingLLM(), FakeTTS())
    rung.load()
    turn = rung.respond(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert [n // (OUT_SR // 2) for _, n in turn.chunks] == [5, 4]            # two clauses, two pieces
    assert set(turn.stages_ms) == {"asr", "llm", "tts"}
    assert turn.first_audio_ms <= turn.total_ms


# --------------------------------------------------------------------------- #
# The real slow tier: skipped unless asked for with -m slow
# --------------------------------------------------------------------------- #

_F02 = PROBE_ROOT / "audio" / "f02.wav"


@pytest.mark.slow
@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("faster_whisper", "transformers", "kokoro")),
                    reason="the model libraries are not installed")
@pytest.mark.skipif(not _F02.exists(), reason="the probe recordings are not on this machine")
def test_real_r2_hears_a_turn_with_the_slow_tier_and_answers_it():
    from s2slab.probes import load_wav
    rung = get("r2")
    rung.load()
    turn = rung.respond(load_wav(_F02), ASR_SR)
    assert rung.asr.model_name == TIER_B
    heard = turn.transcript.lower().replace("14th", "fourteenth")
    assert "appointment" in heard and "fourteenth" in heard
    assert turn.reply_text and turn.reply_s > 0.5 and len(turn.chunks) >= 1