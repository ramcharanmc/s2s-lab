"""
R4 -- S2S half-duplex: the model hears the turn and SPEAKS the reply itself. No ASR, no TTS.

    user audio, 16 kHz
      -> [s2s]  Qwen2.5-Omni-7B: a "Thinker" LLM writes the reply while a "Talker"
                emits speech tokens for it, decoded to 24 kHz                          reply text AND reply audio

The v2 plan, R4: "No TTS text roundtrip; the model chooses its own delivery.
... Expect a possible negative: speech-token decoders are often weaker than a
dedicated TTS, so R4 can lose to R1 on text quality while winning on latency.
Measure whether output F0 variance actually moves with sentiment. If it
doesn't, the rung bought nothing -- and that is the finding."

WHAT IS THE SAME AS R0-R3. The instruction (r0.SYSTEM_PROMPT, word for word),
greedy decoding, the 60-token cap on the written reply, clean_for_speech on
the text that is reported. WHAT IS NOT: the instruction travels as the user's
text beside the audio, because the Talker only speaks under the model's own
fixed system line (OMNI_SYSTEM) -- a documented difference from R3, where the
same words were the system prompt. And there is no Kokoro: the reply audio is
the model's, voice SPEAKER, at 24 kHz, so its leading silence, level and pitch
are the model's own delivery, which is what PR-008 measures.

Turn.transcript is None (no text of the caller exists anywhere); reply_text
is the Thinker's text, so reply length stays comparable with R0-R3. One timed
stage, s2s: the API returns text and audio together, so the split between
"text ready" and "audio ready" is not observable from outside. Batch.

GPU ONLY, and a bigger one: about 20 GB of bf16 weights plus activations, so
an L40S (modal_app/r4_bench.py), with R3 run in the same container as the
comparator. The tests use a fake model.

    modal run modal_app/r4_bench.py --rung r4-gpu                       30 probe clips: smoke test
    modal run modal_app/r4_bench.py --rung r3-gpu,r4-gpu --evalset      the R4 row, R3 beside it, one L40S container
"""

from __future__ import annotations

import numpy as np

from .r0 import ASR_SR, SYSTEM_PROMPT, clean_for_speech
from .rungs import Reply, Rung
from .timing import stage

MODEL = "Qwen/Qwen2.5-Omni-7B"
# The Talker refuses to speak under any other system line (transformers checks it).
OMNI_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
               "auditory and visual inputs, as well as generating text and speech.")
INSTRUCTION = SYSTEM_PROMPT          # R0's words, as the user's text beside the audio
SPEAKER = "Chelsie"                  # the model's female voice; Kokoro's af_heart is female too
OUT_SR = 24_000                      # the Talker's code2wav output rate


class OmniS2S:
    """Qwen2.5-Omni-7B through transformers: a clip in, the reply's text AND speech out. Greedy."""

    def __init__(self, model_name: str = MODEL, device: str = "cuda", max_new_tokens: int = 60,
                 speaker: str = SPEAKER):
        self.model_name, self.device, self.max_new_tokens, self.speaker = model_name, device, max_new_tokens, speaker
        self.processor = self.model = None
        self.sr = ASR_SR

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_name, dtype=torch.bfloat16).to(self.device).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def conversation(instruction: str = INSTRUCTION) -> list[dict]:
        """The chat: the model's own system line, then the clip and R0's instruction as the user's turn."""
        return [{"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM}]},
                {"role": "user", "content": [{"type": "audio", "audio": "turn.wav"},
                                             {"type": "text", "text": instruction}]}]

    def _inputs(self, audio: np.ndarray, sr: int):
        if sr != self.sr:
            raise ValueError(f"the model expects {self.sr} Hz audio, got {sr}")
        prompt = self.processor.apply_chat_template(self.conversation(), add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=prompt, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
        except TypeError:                                             # older processors spell it "audios"
            inputs = self.processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def speak(self, audio: np.ndarray, sr: int) -> tuple[str, np.ndarray]:
        """Takes: the caller's turn. Returns: (the reply's text, the reply's audio at OUT_SR, float32 mono)."""
        import torch
        inputs = self._inputs(np.asarray(audio, dtype=np.float32), sr)
        with torch.inference_mode():
            text_ids, wav = self.model.generate(**inputs, use_audio_in_video=False, return_audio=True,
                                                speaker=self.speaker, thinker_max_new_tokens=self.max_new_tokens,
                                                thinker_do_sample=False)
        text = self.processor.batch_decode(text_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip(), wav.reshape(-1).float().cpu().numpy().astype(np.float32)


class SpeechToSpeech(Rung):
    """R4: the turn's audio in, the model's own spoken reply out. One stage, no ASR, no TTS."""

    rung_id = "r4-gpu"
    description = "R4 S2S half-duplex: Qwen2.5-Omni-7B hears the turn and speaks the reply itself; no transcript, no TTS (bf16, CUDA)"

    def __init__(self, model=None):
        self.model = model or OmniS2S()

    def _load(self) -> None:
        self.model.load()
        self.model.speak(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)     # one throwaway turn, as R0-R3

    def _respond(self, audio, sr, sink):
        if sr != ASR_SR:
            raise ValueError(f"R4 expects {ASR_SR} Hz input, got {sr}: resample first (probes.load_wav does)")
        with stage("s2s", sink):
            text, wav = self.model.speak(audio, sr)
        return Reply(wav, OUT_SR, None, clean_for_speech(text))
