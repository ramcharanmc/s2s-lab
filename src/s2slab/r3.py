"""
R3 -- the speech-LLM rung: the audio goes straight into the LLM. No transcript exists.

    user audio, 16 kHz
      -> [llm]  Qwen2-Audio-7B-Instruct hears the clip, given R0's system prompt    reply text
      -> [tts]  Kokoro-82M, voice af_heart                                            reply audio, 24 kHz

The v2 plan, R3: "An audio encoder emits continuous embeddings; a small
projector maps them into the LLM's embedding space. The LLM never sees a
transcript -- it sees the audio." And what breaks: "most speech-LLMs are not
streaming-native, so you inherit R0 latency on the input side unless you
chunk. Naming that tradeoff honestly is the interesting part." This rung is
built to put a number on it.

WHAT IS THE SAME AS R0/R1, ON PURPOSE. The system prompt (r0.SYSTEM_PROMPT),
greedy decoding, the 60-token cap, clean_for_speech, Kokoro and its voice. The
only differences are the model (7B, hears audio) and the missing ASR stage,
so the row differs from R0's by exactly those two things.

WHAT IS DIFFERENT. Turn.transcript is None -- there is no text of the caller's
words anywhere in the loop, and the eval report says so instead of scoring a
blank. Separately and UNTIMED, hear() asks the same model to transcribe the
clip (r3probe.TRANSCRIBE); evalrun stores that as "heard" and scores it as
"asked to transcribe, outside the loop". It is what the model could have
written down, not what the reply was made from.

Batch, like R0: the reply text is generated whole, then spoken. Streaming the
OUTPUT clause by clause (R1's chunker applies unchanged) is the obvious next
variant; the plan's warning is about the INPUT side, which no output
streaming touches, and that is the number this row is for.

GPU ONLY. Sixteen gigabytes of weights in bf16; there is no CPU variant, and
the tests use a fake speech-LLM. The probe of the same model is r3probe.py /
modal_app/r3_probe.py (PR-006); this file is the rung.

    modal run modal_app/r3_probe.py::evalset --rung r3-gpu --evalset --limit 8     smoke test
    modal run modal_app/r3_probe.py::evalset --rung r0-gpu,r1-gpu,r3-gpu           the row, same container
"""

from __future__ import annotations

import re

import numpy as np

from .r0 import ASR_SR, SYSTEM_PROMPT, KokoroTTS, clean_for_speech
from .r3probe import TRANSCRIBE
from .rungs import Reply, Rung
from .timing import stage

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
TRANSCRIBE_SYSTEM = "You are a helpful assistant."
HEAR_MAX_NEW_TOKENS = 200     # a 15 s clip is 40-odd words; the reply cap (60) would cut it, and its wrapper costs 10
# Qwen2-Audio answers a transcription request as a sentence: "The transcription
# of the speech is: 'I'm fine.'" on all 39 PR-006 clips and the 10 own-voice
# takes; on the eval set's longer clips also "The original content of this
# audio is: '...'" (3 of the 8 smoke-test clips). The words inside are what
# gets scored.
_WRAPPER = re.compile(r"^\s*the (?:original )?(?:transcription|content|text)(?: of (?:the|this) (?:speech|audio))?"
                      r"(?: in this clip)? is:?\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)


def unwrap_transcript(text: str) -> str:
    """The words, out of Qwen2-Audio's "The transcription of the speech is: '...'".

    Takes:   the model's answer to TRANSCRIBE.
    Returns: the quoted words without the wrapper and its quotes; text that
             has no wrapper comes back stripped and otherwise unchanged.
    """
    m = _WRAPPER.match(text or "")
    inner = (m.group(1) if m else (text or "")).strip()
    if m:                                  # the sentence's own full stop, then the quotes around the words
        inner = inner.rstrip(".").strip().strip("'\"‘’“”").strip()
    return inner


class SpeechLLM:
    """Qwen2-Audio-7B-Instruct through transformers: a clip and a prompt in, text out. Greedy."""

    def __init__(self, model_name: str = MODEL, device: str = "cuda", max_new_tokens: int = 60):
        self.model_name, self.device, self.max_new_tokens = model_name, device, max_new_tokens
        self.processor = self.model = None
        self.sr = ASR_SR

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, dtype=torch.bfloat16).to(self.device).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def _inputs(self, audio: np.ndarray, sr: int, system: str, text: str):
        """The chat template with the clip (and a text, if any) as the user's turn."""
        if sr != self.sr:
            raise ValueError(f"the model expects {self.sr} Hz audio, got {sr}")
        content = [{"type": "audio", "audio_url": "turn.wav"}]
        if text:
            content.append({"type": "text", "text": text})
        conversation = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=prompt, audio=[audio], sampling_rate=sr, return_tensors="pt")
        except TypeError:                                             # older processors spell it "audios"
            inputs = self.processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt")
        return inputs.to(self.device)

    def generate(self, audio: np.ndarray, sr: int, system: str, text: str = "",
                 max_new_tokens: int | None = None) -> str:
        """Greedy, capped at max_new_tokens (the reply's cap unless given); only the new tokens are decoded."""
        import torch
        inputs = self._inputs(np.asarray(audio, dtype=np.float32), sr, system, text)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def reply(self, audio: np.ndarray, sr: int) -> str:
        """The agent's reply to what it HEARD: the clip under R0's system prompt, no text."""
        return self.generate(audio, sr, SYSTEM_PROMPT)

    def hear(self, audio: np.ndarray, sr: int) -> str:
        """The same model asked to write the words down (untimed; not in the loop),
        with room for a long clip's words -- the reply's 60-token cap is not a transcript's."""
        return unwrap_transcript(self.generate(audio, sr, TRANSCRIBE_SYSTEM, TRANSCRIBE, HEAR_MAX_NEW_TOKENS))


class SpeechLLMCascade(Rung):
    """R3: the audio into a speech-LLM, its text reply into Kokoro. Two stages, no ASR."""

    rung_id = "r3-gpu"
    description = "R3 speech-LLM: Qwen2-Audio-7B-Instruct hears the turn -> Kokoro-82M; no transcript (bf16, CUDA)"

    def __init__(self, llm=None, tts=None):
        self.llm = llm or SpeechLLM()
        self.tts = tts or KokoroTTS(device="cuda")

    def _load(self) -> None:
        self.llm.load()
        self.tts.load()
        # One throwaway turn, as R0 does: first-call costs belong to start-up.
        said = self.llm.reply(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)
        self.tts.speak(clean_for_speech(said) or "Hello.")

    def _respond(self, audio, sr, sink):
        if sr != ASR_SR:
            raise ValueError(f"R3 expects {ASR_SR} Hz input, got {sr}: resample first (probes.load_wav does)")
        with stage("llm", sink):
            reply_text = self.llm.reply(audio, sr)
        with stage("tts", sink):
            reply = self.tts.speak(clean_for_speech(reply_text))
        return Reply(reply, self.tts.SR, None, reply_text)

    def hear(self, audio, sr) -> str:
        """What the model writes down when asked to transcribe -- outside any turn's clock."""
        return self.llm.hear(np.asarray(audio, dtype=np.float32), sr)
