"""
Tests for s2slab.r1 -- the streaming cascade.

The fast tests use fake stages, as test_r0 does; the fake LLM streams its reply
one word at a time with a known delay, so each test knows when every clause
became complete. The streaming LLM adapter is tested against transformers' real
TextIteratorStreamer with a fake model and tokenizer. The slow tests load the
real models and compare R1 with R0 on a probe recording:

    python -m pytest -m slow tests/test_r1.py
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
from s2slab.r0 import ASR_SR, KokoroTTS, NaiveCascade
from s2slab.r1 import StreamingCascade, StreamingQwenLLM
from s2slab.rungs import get

OUT_SR = KokoroTTS.SR
ONE_SECOND = np.zeros(ASR_SR, dtype=np.float32)
REPLY = "Your appointment is on Tuesday, and I have noted the change. Anything else?"


class FakeASR:
    def __init__(self, text="i want to move my appointment"):
        self.text, self.loads = text, 0

    def load(self):
        self.loads += 1

    def transcribe(self, audio):
        return self.text


class FakeStreamingLLM:
    """Streams its reply word by word, each word delay_s after the last. The
    delay runs in the generator, so it lands in whichever stage called next()."""

    def __init__(self, reply=REPLY, delay_s=0.0):
        self.reply_text, self.delay_s = reply, delay_s
        self.loads, self.heard = 0, []

    def load(self):
        self.loads += 1

    def reply(self, text):                    # R0's warm-up calls this
        return self.reply_text

    def stream(self, text):
        self.heard.append(text)
        words = self.reply_text.split(" ")
        for i, word in enumerate(words):
            time.sleep(self.delay_s)
            yield word if i == 0 else " " + word


class FakeTTS:
    """Half a second of tone per word, so a clause's length says which it was."""

    SR = OUT_SR

    def __init__(self, delay_s=0.0):
        self.delay_s, self.loads, self.said = delay_s, 0, []

    def load(self):
        self.loads += 1

    def speak(self, text):
        self.said.append(text)
        time.sleep(self.delay_s)
        n = len(text.split()) * OUT_SR // 2
        t = np.arange(n) / OUT_SR
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def fake_r1(reply=REPLY, llm_s=0.0, tts_s=0.0) -> StreamingCascade:
    return StreamingCascade(FakeASR(), FakeStreamingLLM(reply, llm_s), FakeTTS(tts_s))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_r1_is_registered_and_builds_without_any_model_library():
    root = pathlib.Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    code = ("import sys; from s2slab.rungs import get; r = get('r1'); "
            "print(type(r).__name__, type(r.llm).__name__, [m for m in ('torch', 'transformers', "
            "'faster_whisper', 'kokoro') if m in sys.modules])")
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.stdout.strip() == "StreamingCascade StreamingQwenLLM []", proc.stderr


def test_r1_gpu_uses_exactly_r0_gpus_models():
    """PR-002 compares the two, so only streaming may differ."""
    r0, r1 = get("r0-gpu"), get("r1-gpu")
    assert (r1.asr.model_name, r1.asr.device, r1.asr.compute_type) == \
           (r0.asr.model_name, r0.asr.device, r0.asr.compute_type)
    assert (r1.llm.model_name, r1.llm.device, r1.llm.max_new_tokens) == \
           (r0.llm.model_name, r0.llm.device, r0.llm.max_new_tokens)
    assert (r1.tts.voice, r1.tts.device) == (r0.tts.voice, r0.tts.device)
    assert isinstance(r1.llm, StreamingQwenLLM)



# --------------------------------------------------------------------------- #
# Streaming, with fake stages
# --------------------------------------------------------------------------- #


def test_each_clause_is_spoken_and_emitted_as_its_own_piece():
    rung = fake_r1()
    rung.load()
    rung.tts.said.clear()                             # forget the warm-up turns
    turn = rung.respond(ONE_SECOND, ASR_SR)
    assert rung.tts.said == ["Your appointment is on Tuesday,",
                             "and I have noted the change.", "Anything else?"]
    assert [n // (OUT_SR // 2) for _, n in turn.chunks] == [5, 6, 2]     # words per piece
    assert turn.reply_text == REPLY
    assert turn.transcript == "i want to move my appointment"
    assert rung.llm.heard[-1] == turn.transcript


def test_first_audio_comes_after_the_first_clause_not_the_last_word():
    """13 words at 20 ms each: the whole reply takes 260 ms to write, but the
    first clause is complete after 6 words -- its comma is confirmed by the
    space before the sixth."""
    turn = fake_r1(llm_s=0.02).respond(ONE_SECOND, ASR_SR)
    assert 120 <= turn.first_audio_ms < 200
    assert turn.total_ms >= 260
    assert turn.first_audio_ms < turn.total_ms - 60


def test_first_audio_does_not_grow_with_the_reply():
    """R0's first audio tracks reply length. R1's should not."""
    short = fake_r1("Sure, I have noted the change. Thanks.", llm_s=0.01).respond(ONE_SECOND, ASR_SR)
    long_reply = "Sure, I have noted the change. " + "More words follow here. " * 8
    long = fake_r1(long_reply, llm_s=0.01).respond(ONE_SECOND, ASR_SR)
    assert long.total_ms > short.total_ms + 200
    assert abs(long.first_audio_ms - short.first_audio_ms) < 40


def test_stages_are_asr_llm_tts_and_add_up():
    turn = fake_r1(llm_s=0.01, tts_s=0.02).respond(ONE_SECOND, ASR_SR)
    assert list(turn.stages_ms) == ["asr", "llm", "tts"]
    assert turn.stages_ms["tts"] >= 3 * 20          # three clauses
    assert turn.stages_ms["llm"] >= 13 * 10         # thirteen words
    assert turn.untimed_ms < 10


def test_markdown_is_cleaned_per_clause_and_empty_clauses_are_skipped():
    rung = fake_r1("**Sure.**\n**\nAnything else?")      # the middle line is only markdown
    rung.load()
    rung.tts.said.clear()
    turn = rung.respond(ONE_SECOND, ASR_SR)
    assert rung.tts.said == ["Sure.", "Anything else?"]
    assert len(turn.chunks) == 2


def test_a_reply_with_no_words_has_no_audio():
    turn = fake_r1("").respond(ONE_SECOND, ASR_SR)
    assert len(turn.audio) == 0 and np.isnan(turn.first_audio_ms)


def test_load_warms_up_the_streaming_path_before_the_clock():
    rung = fake_r1()
    rung.load()
    assert rung.llm.heard == ["Hello."]              # the streamed warm-up
    assert (rung.asr.loads, rung.llm.loads, rung.tts.loads) == (1, 1, 1)


def test_r1_refuses_audio_at_the_wrong_rate():
    with pytest.raises(ValueError, match="16000"):
        fake_r1().respond(np.zeros(8_000, dtype=np.float32), 8_000)



# --------------------------------------------------------------------------- #
# The streaming LLM adapter, against transformers' real streamer
# --------------------------------------------------------------------------- #


class _Tokens(dict):
    """What a tokenizer call returns: a dict of tensors with a .to(device)."""

    def to(self, device):
        return self


class FakeTokenizer:
    VOCAB = ["<prompt>", "Your", " appointment", " is", " on", " Tuesday", ",", " thanks", "."]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def __call__(self, prompt, return_tensors):
        import torch
        return _Tokens(input_ids=torch.tensor([[0]]))

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(self.VOCAB[int(i)] for i in ids if int(i) != 0)


class FakeModel:
    """Writes tokens 1..8 into the streamer the way generate() does: the prompt
    first, then one new token at a time, then end()."""

    def __init__(self, fail_after=None, delay_s=0.0):
        self.fail_after, self.delay_s = fail_after, delay_s

    def generate(self, input_ids, streamer=None, **kwargs):
        import torch
        streamer.put(input_ids)
        for k in range(1, len(FakeTokenizer.VOCAB)):
            if self.fail_after is not None and k > self.fail_after:
                raise RuntimeError("CUDA out of memory")
            time.sleep(self.delay_s)
            streamer.put(torch.tensor([k]))
        streamer.end()


def _streaming_llm(model) -> StreamingQwenLLM:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    llm = StreamingQwenLLM()
    llm.tokenizer, llm.model = FakeTokenizer(), model
    return llm


def test_the_streamed_pieces_join_into_the_whole_reply():
    llm = _streaming_llm(FakeModel())
    pieces = list(llm.stream("hi"))
    assert "".join(pieces) == "Your appointment is on Tuesday, thanks."
    assert len(pieces) > 1                            # it really arrived in pieces


def test_pieces_arrive_while_generation_is_still_running():
    """8 tokens at 30 ms: generation takes 240 ms, and the first whole word
    ("Your ") is ready after the second token, at about 60 ms."""
    llm = _streaming_llm(FakeModel(delay_s=0.03))
    from transformers import TextIteratorStreamer  # noqa: F401 -- a first import is slow; keep it off the clock
    t0 = time.perf_counter()
    pieces = llm.stream("hi")          # keep a reference: closing the generator waits for generation
    first = next(pieces)
    elapsed = time.perf_counter() - t0
    assert first == "Your " and elapsed < 0.03 * 8 - 0.05
    assert "".join([first, *pieces]) == "Your appointment is on Tuesday, thanks."


def test_a_failure_inside_generate_is_raised_not_waited_on():
    llm = _streaming_llm(FakeModel(fail_after=3))
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="out of memory"):
        list(llm.stream("hi"))
    assert time.perf_counter() - t0 < 5


# --------------------------------------------------------------------------- #
# The real models: slow, skipped unless asked for with -m slow
# --------------------------------------------------------------------------- #

_F02 = PROBE_ROOT / "audio" / "f02.wav"
needs_models = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("faster_whisper", "transformers", "kokoro")),
    reason="the model libraries are not installed")
needs_f02 = pytest.mark.skipif(not _F02.exists(), reason="the probe recordings are not on this machine")


@pytest.fixture(scope="module")
def real_pair():
    """R0 and R1 sharing one set of loaded models: R1's LLM reuses R0's weights."""
    r0 = NaiveCascade()
    r0.load()
    llm = StreamingQwenLLM()
    llm.tokenizer, llm.model = r0.llm.tokenizer, r0.llm.model
    r1 = StreamingCascade(r0.asr, llm, r0.tts)
    r1._load = lambda: None                            # already loaded; skip reloading
    r1.load()
    return r0, r1


@pytest.fixture(scope="module")
def f02():
    from s2slab.probes import load_wav
    return load_wav(_F02)


@pytest.mark.slow
@needs_models
@needs_f02
def test_real_r1_says_what_r0_says(real_pair, f02):
    """Same prompt, same greedy decoding: streaming changes when, not what."""
    r0, r1 = real_pair
    said_r0 = r0.respond(f02, ASR_SR).reply_text
    said_r1 = r1.respond(f02, ASR_SR).reply_text
    assert said_r1.split() == said_r0.split()


@pytest.mark.slow
@needs_models
@needs_f02
def test_real_r1_streams_a_whole_turn_cleanly(real_pair, f02):
    r0, r1 = real_pair
    r1.respond(f02, ASR_SR)                            # one turn to settle the thread path
    turn = r1.respond(f02, ASR_SR)
    assert turn.chunks and turn.reply_s > 0.5
    if len(turn.chunks) > 1:
        assert turn.first_audio_ms < turn.total_ms
    assert turn.untimed_ms < 50