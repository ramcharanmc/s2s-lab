"""
R0 -- the naive cascade: Whisper -> Qwen -> Kokoro, each waiting for the last.

The v2 plan, days 1-2: "build it in an afternoon, do not skip it." It teaches
the wiring and the audio plumbing, and it lets you feel what three seconds of
latency is like -- which is what motivates every rung after it.

    user turn, 16 kHz
      -> [asr]  faster-whisper base.en, int8    "i would like to change the delivery address"
      -> [llm]  Qwen2.5-0.5B-Instruct, greedy    "Sure, what is the new address?"
      -> [tts]  Kokoro-82M, voice af_heart       reply audio, 24 kHz

NAIVE ON PURPOSE. Every stage waits for the whole output of the one before: the
LLM cannot start until the transcript is final, TTS cannot start until the last
token exists, and nothing plays until every sample is rendered. R1 removes
exactly these waits by streaming, and this stage table is the baseline it has
to beat.

Each stage is a small adapter with one method, so the rung only wires them:

    WhisperASR.transcribe(audio) -> str
    QwenLLM.reply(text)         -> str
    KokoroTTS.speak(text)       -> float32 audio at 24 kHz

Model libraries are imported inside load(), never at the top of this file, so
get("r0") and the fast tests work with nothing installed. Between platforms,
change sizes, not code: NaiveCascade(WhisperASR("small.en", device="cuda"), ...)
on Colab.

One turn, reply written to results/r0/ so you can listen to it:
    PYTHONPATH=src python -m s2slab.r0 probes/endpointing/audio/f02.wav
All 30 probe clips, stage table:
    PYTHONPATH=src python -m s2slab.rungs r0
"""

from __future__ import annotations

import os
import re

import numpy as np

from .rungs import Reply, Rung
from .timing import stage

ASR_SR = 16_000        # Whisper was trained on 16 kHz audio
ESPEAK_PATH_MAX = 159  # espeak-ng keeps its data path in a 160-byte buffer (measured)

# Chosen by trying three on five probe transcripts. An earlier prompt ended
# "everything you write is read aloud", and the 0.5B model answered that
# instruction ("Sure, I'll read it out loud for you") instead of the caller.
SYSTEM_PROMPT = (
    "You are a friendly phone agent helping a caller. Respond to what they "
    "said in one or two short sentences. Use plain spoken English: no lists, "
    "markdown or emoji."
)

def clean_for_speech(text: str) -> str:
    """Strip what a TTS would read aloud literally, or stumble over.

    Takes:   the LLM's reply.
    Returns: the same words without markdown symbols (* _ ` #) or list markers
             at the start of a line, with whitespace collapsed. The prompt asks
             for plain text; a 0.5B model does not always comply, and a voice
             saying "asterisk asterisk" is the bug this prevents.
    """
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def espeak_path_problem(data_path: str, env_path: str | None) -> str | None:
    """Would espeak-ng fail to find its data? If so, say how to fix it.

    Takes:   the data folder espeakng_loader ships, and $ESPEAK_DATA_PATH.
    Returns: None if espeak-ng can use one of them, else a message.

    Kokoro spells out words its dictionary lacks with espeak-ng. espeak-ng
    copies its data path into a 160-byte buffer; a longer path is cut short,
    it falls back to a folder that only existed on the machine that built the
    package, and it EXITS THE PROCESS -- no exception, no traceback. The
    environment variable is read by espeak-ng itself, so a short one rescues a
    long install path.
    """
    if len(data_path) <= ESPEAK_PATH_MAX:
        return None
    if env_path and len(env_path) <= ESPEAK_PATH_MAX:
        return None
    return (f"espeak-ng cannot use its data folder: the path is {len(data_path)} "
            f"characters and espeak-ng handles at most {ESPEAK_PATH_MAX}. It would "
            f"exit this process without an error. Fix:\n"
            f"  ln -sfn '{data_path}' /tmp/espeak-ng-data\n"
            f"  export ESPEAK_DATA_PATH=/tmp/espeak-ng-data")

class WhisperASR:
    """faster-whisper (CTranslate2). base.en in int8 is small enough for a laptop CPU."""

    def __init__(self, model_name: str = "base.en", device: str = "cpu",
                 compute_type: str = "int8", beam_size: int = 1):
        self.model_name, self.device = model_name, device
        self.compute_type, self.beam_size = compute_type, beam_size
        self.model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_name, device=self.device,
                                  compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """Takes: mono float32 at 16 kHz. Returns: the transcript as one line.

        model.transcribe() returns a LAZY generator: the call itself decodes
        nothing, and the work happens while the segments are iterated. Timing
        only the call would report about 0 ms and hide the whole stage, so the
        join below -- which drives the iteration -- is part of the stage.
        """
        segments, _info = self.model.transcribe(
            audio, language="en", beam_size=self.beam_size,
            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

class QwenLLM:
    """Qwen2.5-0.5B-Instruct through transformers. Greedy, so a reply is reproducible."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 device: str = "cpu", max_new_tokens: int = 60):
        self.model_name, self.device = model_name, device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype).to(self.device).eval()

    def reply(self, user_text: str) -> str:
        """Takes: what the user said. Returns: the assistant's reply.

        apply_chat_template wraps the messages in Qwen's own turn markers
        (<|im_start|>system ... <|im_end|>); add_generation_prompt opens the
        assistant's turn, so the model continues as the assistant. generate()
        returns the prompt followed by the new tokens; only the new ones are
        decoded.
        """
        import torch
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None)   # greedy: unset the sampling defaults
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

class KokoroTTS:
    """Kokoro-82M: small enough to speak faster than real time on a CPU."""

    SR = 24_000

    def __init__(self, voice: str = "af_heart", lang_code: str = "a", device: str = "cpu"):
        self.voice, self.lang_code, self.device = voice, lang_code, device   # "a" = American English
        self.pipeline = None

    def load(self) -> None:
        import espeakng_loader
        problem = espeak_path_problem(espeakng_loader.get_data_path(),
                                      os.environ.get("ESPEAK_DATA_PATH"))
        if problem:
            raise RuntimeError(problem)
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code=self.lang_code, repo_id="hexgrad/Kokoro-82M",
                                  device=self.device)
        self.pipeline.load_voice(self.voice)     # fetch the voice now, not in the first turn

    def speak(self, text: str) -> np.ndarray:
        """Takes: text. Returns: float32 mono at 24 kHz, empty if there is nothing to say.

        Kokoro splits long text and yields audio chunk by chunk. R0 waits for
        every chunk and joins them -- the naive part. R1 will play the first
        chunk while the rest are still rendering.
        """
        parts = [np.asarray(r.audio, dtype=np.float32)
                 for r in self.pipeline(text, voice=self.voice) if r.audio is not None]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

class NaiveCascade(Rung):
    """R0: three models in a row, each waiting for the whole output of the last."""

    rung_id = "r0"
    description = "naive cascade: Whisper base.en -> Qwen2.5-0.5B -> Kokoro-82M, batch"

    def __init__(self, asr=None, llm=None, tts=None):
        self.asr = asr or WhisperASR()
        self.llm = llm or QwenLLM()
        self.tts = tts or KokoroTTS()

    def _load(self) -> None:
        for part in (self.asr, self.llm, self.tts):
            part.load()
        # One throwaway turn. The first call through each model pays one-time
        # costs -- memory allocation, caches built on first use, Kokoro's
        # pronunciation dictionary -- that belong to start-up, not to a turn.
        heard = self.asr.transcribe(np.zeros(ASR_SR, dtype=np.float32))
        said = self.llm.reply(heard or "Hello.")
        self.tts.speak(clean_for_speech(said) or "Hello.")

    def _respond(self, audio, sr, sink):
        if sr != ASR_SR:
            raise ValueError(f"R0 expects {ASR_SR} Hz input, got {sr}: "
                             "resample first (probes.load_wav does)")
        with stage("asr", sink):
            transcript = self.asr.transcribe(audio)
        with stage("llm", sink):
            reply_text = self.llm.reply(transcript)
        with stage("tts", sink):
            reply = self.tts.speak(clean_for_speech(reply_text))
        return Reply(reply, self.tts.SR, transcript, reply_text) 
       

class NaiveCascadeGPU(NaiveCascade):
    """R0 with the same three models on a CUDA GPU: the hardware effect alone.

    GPU work is asynchronous -- a CUDA call returns before the GPU finishes.
    The stage times stay honest because every stage hands back host-side data:
    a string, or audio Kokoro has already copied to the CPU. That copy waits
    for the GPU, so each stage's time is spent inside its own stage.
    """

    rung_id = "r0-gpu"
    description = "R0, same models on a CUDA GPU (fp16)"

    def __init__(self):
        super().__init__(WhisperASR(device="cuda", compute_type="float16"),
                         QwenLLM(device="cuda"), KokoroTTS(device="cuda"))


class NaiveCascadeGPUBigger(NaiveCascade):
    """R0 on a GPU with larger ASR and LLM models: what the headroom buys."""

    rung_id = "r0-gpu-bigger"
    description = "R0 on a CUDA GPU: Whisper small.en -> Qwen2.5-1.5B -> Kokoro-82M (fp16)"

    def __init__(self):
        super().__init__(WhisperASR("small.en", device="cuda", compute_type="float16"),
                         QwenLLM("Qwen/Qwen2.5-1.5B-Instruct", device="cuda"),
                         KokoroTTS(device="cuda"))


if __name__ == "__main__":
    import pathlib
    import sys

    import soundfile as sf

    from .probes import load_wav
    from .rungs import get

    if len(sys.argv) not in (2, 3):
        sys.exit("usage: PYTHONPATH=src python -m s2slab.r0 path/to/turn.wav [rung id, default r0]")
    src = pathlib.Path(sys.argv[1])
    rung = get(sys.argv[2] if len(sys.argv) == 3 else "r0")
    print(f"{rung.rung_id}: {rung.description}")
    print("loading models (the first run downloads about 1.5 GB) ...")
    print(f"load: {rung.load():.0f} ms")
    turn = rung.respond(load_wav(src), ASR_SR)
    out = pathlib.Path("results") / rung.rung_id / f"{src.stem}.reply.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), turn.audio, turn.sr)
    print(f"heard: {turn.transcript}")
    print(f"said:  {turn.reply_text}")
    print("   ".join(f"{name} {ms:.0f} ms" for name, ms in turn.stages_ms.items())
          + f"   | total {turn.total_ms:.0f} ms, untimed {turn.untimed_ms:.1f} ms")
    print(f"reply: {turn.reply_s:.1f} s of audio, silent for its first "
          f"{turn.leading_silence_ms:.0f} ms -> {out}")