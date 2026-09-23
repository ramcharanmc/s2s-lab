"""
Tests for s2slab.r3 -- the speech-LLM rung, wired with a fake speech-LLM and a fake TTS.

No model: the fake replies with a fixed sentence and "transcribes" with the
wrapper Qwen2-Audio really uses, so the unwrapping is tested on the real shape.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from s2slab.r0 import ASR_SR, SYSTEM_PROMPT
from s2slab.r3 import MODEL, SpeechLLM, SpeechLLMCascade, unwrap_transcript
from s2slab.r3probe import TRANSCRIBE
from s2slab.rungs import RUNGS, get

OUT_SR = 24_000


class FakeSpeechLLM:
    def __init__(self, delay_s=0.0):
        self.delay_s, self.loads, self.asked = delay_s, 0, []
        self.sr = ASR_SR

    def load(self):
        self.loads += 1

    def reply(self, audio, sr):
        self.asked.append(("reply", len(audio), sr))
        time.sleep(self.delay_s)
        return "**Sure**, I can help with that."

    def hear(self, audio, sr):
        self.asked.append(("hear", len(audio), sr))
        return unwrap_transcript("The transcription of the speech is: 'I need to change my appointment.'")


class FakeTTS:
    SR = OUT_SR

    def __init__(self, delay_s=0.0):
        self.delay_s, self.loads, self.said = delay_s, 0, []

    def load(self):
        self.loads += 1

    def speak(self, text):
        self.said.append(text)
        time.sleep(self.delay_s)
        t = np.arange(OUT_SR // 2) / OUT_SR
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def fake_r3(llm_s=0.0, tts_s=0.0) -> SpeechLLMCascade:
    return SpeechLLMCascade(FakeSpeechLLM(llm_s), FakeTTS(tts_s))


# --------------------------------------------------------------------------- #
# The rung
# --------------------------------------------------------------------------- #


def test_r3_is_registered_and_builds_without_any_model_library():
    assert RUNGS["r3-gpu"] == "s2slab.r3:SpeechLLMCascade"
    rung = get("r3-gpu")
    assert rung.rung_id == "r3-gpu" and rung.llm.model_name == MODEL and rung.llm.device == "cuda"
    assert rung.tts.device == "cuda" and rung.tts.voice == "af_heart" and rung.llm.max_new_tokens == 60


def test_a_turn_goes_from_the_audio_to_the_llm_to_the_tts_with_no_transcript():
    rung = fake_r3()
    audio = np.zeros(2 * ASR_SR, dtype=np.float32)
    turn = rung.respond(audio, ASR_SR)
    assert turn.transcript is None                                          # nothing in the loop ever became text
    assert turn.reply_text == "**Sure**, I can help with that."
    assert rung.tts.said[-1] == "Sure, I can help with that."              # clean_for_speech, as R0 (said[0] is the warm-up)
    assert list(turn.stages_ms) == ["llm", "tts"] and turn.sr == OUT_SR and turn.reply_s == 0.5
    assert rung.llm.asked[-1] == ("reply", 2 * ASR_SR, ASR_SR)
    assert turn.chunks == [(pytest.approx(turn.total_ms), OUT_SR // 2)]       # batch: the whole reply at the end


def test_each_stage_is_timed_on_its_own_and_the_untimed_rest_is_small():
    turn = fake_r3(llm_s=0.05, tts_s=0.03).respond(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert turn.stages_ms["llm"] >= 50 and turn.stages_ms["tts"] >= 30 and turn.untimed_ms < 50


def test_load_loads_both_models_once_and_warms_up_before_the_clock():
    rung = fake_r3()
    ms = rung.load()
    assert rung.llm.loads == 1 and rung.tts.loads == 1 and ms >= 0
    assert rung.llm.asked == [("reply", ASR_SR, ASR_SR)] and rung.tts.said      # the throwaway turn
    assert rung.load() == 0.0 and rung.llm.loads == 1
    rung.respond(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert rung.llm.loads == 1


def test_r3_refuses_audio_at_the_wrong_rate():
    with pytest.raises(ValueError, match="16000 Hz"):
        fake_r3().respond(np.zeros(8000, dtype=np.float32), 8000)


def test_hear_is_outside_the_turn_and_gives_the_words_without_the_wrapper():
    rung = fake_r3()
    assert rung.hear(np.zeros(ASR_SR, dtype=np.float32), ASR_SR) == "I need to change my appointment."
    assert rung.llm.asked == [("hear", ASR_SR, ASR_SR)]                       # no turn was timed


# --------------------------------------------------------------------------- #
# The wrapper Qwen2-Audio puts around a transcript
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw, words", [
    ("The transcription of the speech is: 'I'm fine.'", "I'm fine."),
    ("The transcription of the speech is 'that's great.'", "that's great."),
    ("The transcription of the speech is: 'Sure, whatever you say.", "Sure, whatever you say"),   # no closing quote: the stop reads as the sentence's
    ("The transcription of the audio is: \"It's eleven o'clock.\"", "It's eleven o'clock."),
    ("the transcription is: Don't forget a jacket", "Don't forget a jacket"),
    ("Don't forget a jacket.", "Don't forget a jacket."),                    # no wrapper: unchanged
    ("", ""),
])
def test_the_wrapper_is_removed_and_the_words_kept(raw, words):
    assert unwrap_transcript(raw) == words


# --------------------------------------------------------------------------- #
# The real model's prompts, without loading it
# --------------------------------------------------------------------------- #


def test_the_speech_llm_asks_with_r0s_prompt_and_transcribes_with_the_probes_prompt(monkeypatch):
    llm = SpeechLLM(device="cpu")
    calls = []

    def fake_generate(audio, sr, system, text=""):
        calls.append((system, text, sr))
        return "The transcription of the speech is: 'hello.'" if text else "Hi."

    monkeypatch.setattr(llm, "generate", fake_generate)
    audio = np.zeros(ASR_SR, dtype=np.float32)
    assert llm.reply(audio, ASR_SR) == "Hi." and llm.hear(audio, ASR_SR) == "hello."
    assert calls == [(SYSTEM_PROMPT, "", ASR_SR), ("You are a helpful assistant.", TRANSCRIBE, ASR_SR)]
