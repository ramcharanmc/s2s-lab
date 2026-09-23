"""
Tests for s2slab.r4 -- the S2S rung, wired with a fake model that "speaks".

No model: the fake returns a fixed text and half a second of tone, so the
wiring (one stage, no transcript, the model's own audio at 24 kHz) is checked
without a GPU.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from s2slab.r0 import ASR_SR, SYSTEM_PROMPT
from s2slab.r4 import INSTRUCTION, MODEL, OMNI_SYSTEM, OUT_SR, SPEAKER, OmniS2S, SpeechToSpeech
from s2slab.rungs import RUNGS, get


class FakeOmni:
    def __init__(self, delay_s=0.0):
        self.delay_s, self.loads, self.asked = delay_s, 0, []
        self.sr = ASR_SR

    def load(self):
        self.loads += 1

    def speak(self, audio, sr):
        self.asked.append((len(audio), sr))
        time.sleep(self.delay_s)
        t = np.arange(OUT_SR // 2) / OUT_SR
        return "**Sure**, I can help with that.", (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def fake_r4(delay_s=0.0) -> SpeechToSpeech:
    return SpeechToSpeech(FakeOmni(delay_s))


def test_r4_is_registered_and_builds_without_any_model_library():
    assert RUNGS["r4-gpu"] == "s2slab.r4:SpeechToSpeech"
    rung = get("r4-gpu")
    assert rung.rung_id == "r4-gpu" and rung.model.model_name == MODEL and rung.model.device == "cuda"
    assert rung.model.max_new_tokens == 60 and rung.model.speaker == SPEAKER
    assert not hasattr(rung, "tts") and not hasattr(rung, "asr")            # no Kokoro, no Whisper


def test_a_turn_is_one_stage_with_the_models_own_voice_and_no_transcript():
    rung = fake_r4()
    turn = rung.respond(np.zeros(2 * ASR_SR, dtype=np.float32), ASR_SR)
    assert turn.transcript is None
    assert turn.reply_text == "Sure, I can help with that."                 # clean_for_speech on the reported text
    assert list(turn.stages_ms) == ["s2s"] and turn.sr == OUT_SR and turn.reply_s == 0.5
    assert turn.chunks == [(pytest.approx(turn.total_ms), OUT_SR // 2)]       # batch
    assert rung.model.asked[-1] == (2 * ASR_SR, ASR_SR)
    assert turn.leading_silence_ms == 0.0                                     # a tone from the first sample: no silence to skip


def test_the_stage_is_timed_and_the_rest_is_small():
    turn = fake_r4(0.05).respond(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert turn.stages_ms["s2s"] >= 50 and turn.untimed_ms < 50


def test_load_loads_once_and_warms_up_before_the_clock():
    rung = fake_r4()
    ms = rung.load()
    assert rung.model.loads == 1 and rung.model.asked == [(ASR_SR, ASR_SR)] and ms >= 0
    assert rung.load() == 0.0 and rung.model.loads == 1


def test_r4_refuses_audio_at_the_wrong_rate():
    with pytest.raises(ValueError, match="16000 Hz"):
        fake_r4().respond(np.zeros(8000, dtype=np.float32), 8000)


def test_the_chat_carries_the_models_own_system_line_and_r0s_instruction_beside_the_audio():
    conv = OmniS2S.conversation()
    assert conv[0] == {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM}]}
    assert conv[1]["role"] == "user" and conv[1]["content"][0]["type"] == "audio"
    assert conv[1]["content"][1] == {"type": "text", "text": SYSTEM_PROMPT} and INSTRUCTION == SYSTEM_PROMPT
    assert "generating text and speech" in OMNI_SYSTEM                        # the line the Talker requires


def test_speak_decodes_only_the_new_tokens_and_flattens_the_audio(monkeypatch):
    m = OmniS2S(device="cpu")

    class Ids:
        shape = (1, 7)

    class Inputs(dict):
        def to(self, device):
            return self

    m._inputs = lambda audio, sr: Inputs(input_ids=Ids())
    seen = {}

    class Model:
        def generate(self, **kw):
            seen.update(kw)
            import numpy as _np
            return _np.zeros((1, 9), dtype=int), FakeTensor(_np.ones((1, 240), dtype=np.float32))

    class Processor:
        def batch_decode(self, ids, skip_special_tokens):
            seen["decoded_shape"] = ids.shape
            return ["  Hello there.  "]

    class FakeTensor:
        def __init__(self, a):
            self.a = a

        def reshape(self, *s):
            return FakeTensor(self.a.reshape(*s))

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    m.model, m.processor = Model(), Processor()
    pytest.importorskip("torch")                                              # speak() opens torch.inference_mode()
    text, wav = m.speak(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
    assert text == "Hello there." and wav.shape == (240,) and wav.dtype == np.float32
    assert seen["decoded_shape"] == (1, 2)                                    # 9 tokens minus the 7 of the prompt
    assert seen["return_audio"] is True and seen["speaker"] == SPEAKER
    assert seen["thinker_max_new_tokens"] == 60 and seen["thinker_do_sample"] is False and seen["use_audio_in_video"] is False
