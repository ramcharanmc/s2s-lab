"""
R1 -- the streaming cascade: speak the first clause while the rest is still being written.

The v2 plan, week 1: "what ~90% of production voice agents are." Same three
models as R0, same prompt, same greedy decoding. Only WHEN audio exists
changes, so any difference from R0 is what streaming alone buys.

    user turn, 16 kHz
      -> [asr]  faster-whisper base.en           the whole turn, as in R0
      -> [llm]  Qwen2.5-0.5B, token by token      "Your appoint" "ment is on Tues" "day," ...
                  -> ClauseChunker                "Your appointment is on Tuesday,"  -> [tts] -> emit()
                  -> ...                          "and I've noted the change."        -> [tts] -> emit()

WHY IT IS FASTER. R0's first audio waits for the last token and the last
sample. R1's waits for the first clause only: prompt processing, a handful of
tokens, and one short synthesis. The eval runs say R0's time tracks reply
length (r = 0.97 on the T4); R1's first audio should not.

WHAT IT COSTS. Generation runs in a background thread while the main thread
synthesizes clauses, so the two share the GPU (or the CPU cores) and Python's
interpreter lock. The total can grow; PR-002's speed guardrail measures by how
much. And each clause is synthesized alone, so its prosody resets at every
seam -- the plan's "sounds like a list". No number here scores that; listen.

WHAT THE STAGES MEAN NOW. "llm" is the time the pipeline WAITED for the next
piece of text -- generation keeps running during "tts", so this is less than
the model's generation time. The stages still add up to the total.

    PYTHONPATH=src python -m s2slab.r0 probes/endpointing/audio/f02.wav r1   one turn, reply saved
    PYTHONPATH=src python -m s2slab.rungs r1                                 30 probe clips
"""

from __future__ import annotations

import threading

from .chunking import ClauseChunker
from .r0 import ASR_SR, KokoroTTS, NaiveCascade, QwenLLM, WhisperASR, clean_for_speech
from .rungs import Reply
from .timing import stage

STREAM_TIMEOUT_S = 120   # a token that takes longer than this means generation is stuck



class StreamingQwenLLM(QwenLLM):
    """QwenLLM that hands its reply out piece by piece as tokens are generated."""

    def stream(self, user_text: str):
        """Takes: what the user said.
        Yields: pieces of the reply as they are decoded -- usually a word or two.
                Joined, they are the reply R0 would give (whitespace aside).

        generate() blocks until the last token, so it runs in a background
        thread and writes into a TextIteratorStreamer, which this generator
        reads from. If generate() fails, the error is re-raised here rather
        than leaving this loop waiting forever.
        """
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.inputs(user_text)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True, timeout=STREAM_TIMEOUT_S)
        failure: list[BaseException] = []

        def generate():
            try:
                with torch.inference_mode():       # per thread: the main thread's mode does not carry over
                    self.model.generate(**inputs, **self.generate_kwargs(), streamer=streamer)
            except BaseException as e:             # noqa: BLE001 -- handed to the main thread
                failure.append(e)
                streamer.end()                     # unblock the reader

        worker = threading.Thread(target=generate, name="qwen-generate", daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            worker.join(timeout=STREAM_TIMEOUT_S)
        if failure:
            raise failure[0]



class StreamingCascade(NaiveCascade):
    """R1: R0's models, with the reply spoken clause by clause as it is written."""

    rung_id = "r1"
    description = "streaming cascade: Whisper base.en -> Qwen2.5-0.5B (streamed) -> Kokoro-82M per clause"

    def __init__(self, asr=None, llm=None, tts=None):
        super().__init__(asr, llm or StreamingQwenLLM(), tts)

    def _load(self) -> None:
        super()._load()                            # load all three, then R0's warm-up turn
        # R1's own path once more, untimed: the first thread, the streamer, and
        # a short synthesis all pay first-use costs that belong to start-up.
        for clause in self._clauses("Hello.", []):
            self.tts.speak(clean_for_speech(clause))

    def _clauses(self, transcript: str, pieces: list, sink: list | None = None):
        """Takes: the transcript; a list to collect every text piece in, so the
                 full reply can be returned; the sink to time the LLM wait in
                 (None: untimed).
        Yields:  complete clauses, the moment each is complete.
        """
        chunker = ClauseChunker()
        tokens = self.llm.stream(transcript)
        while True:
            if sink is None:
                piece = next(tokens, None)
            else:
                with stage("llm", sink):
                    piece = next(tokens, None)
            if piece is None:
                break
            pieces.append(piece)
            yield from chunker.feed(piece)
        yield from chunker.flush()

    def _respond(self, audio, sr, sink):
        if sr != ASR_SR:
            raise ValueError(f"R1 expects {ASR_SR} Hz input, got {sr}: "
                             "resample first (probes.load_wav does)")
        with stage("asr", sink):
            transcript = self.asr.transcribe(audio)
        pieces: list[str] = []
        first = True
        for clause in self._clauses(transcript, pieces, sink):
            text = clean_for_speech(clause)
            if not text:                           # a clause that was only markdown
                continue
            with stage("tts", sink):
                piece = self._shape(self.tts.speak(text), first)
            first = False
            self.emit(piece)
        return Reply(None, self.tts.SR, transcript, "".join(pieces).strip())

    def _shape(self, piece, first: bool):
        """What a piece is before it is handed out: as spoken. r1-trim cuts the first one's silence (M38)."""
        return piece



class StreamingCascadeGPU(StreamingCascade):
    """R1 with the same models as r0-gpu, on a CUDA GPU: the pair PR-002 compares."""

    rung_id = "r1-gpu"
    description = "R1, same models on a CUDA GPU (fp16), reply streamed clause by clause"

    def __init__(self):
        super().__init__(WhisperASR(device="cuda", compute_type="float16"),
                         StreamingQwenLLM(device="cuda"), KokoroTTS(device="cuda"))


class FirstTrimmed:
    """A reply's first piece cut to leadsilence.KEEP_MS before its first frame within 50 dB (M38): the padding
    Kokoro's duration predictor puts before the first phoneme, 320 ms at the median (docs/m38-log.md). Later
    pieces are left alone: their silence is the pause between clauses. turn_extra() says what was cut."""

    last_trim_ms, last_cut_peak_db = 0.0, None

    def _respond(self, audio, sr, sink):
        self.last_trim_ms, self.last_cut_peak_db = 0.0, None
        return super()._respond(audio, sr, sink)

    def _shape(self, piece, first: bool):
        if not first:
            return piece
        import numpy as np

        from .leadsilence import trim
        from .vad import frame_rms
        out, cut_ms = trim(piece, self.tts.SR)
        cut = len(piece) - len(out)
        self.last_trim_ms = round(cut_ms, 1)
        if cut:
            whole, part = frame_rms(piece, self.tts.SR, 5.0), frame_rms(piece[:cut], self.tts.SR, 5.0)
            if part.size and whole.size and whole.max() > 0:
                self.last_cut_peak_db = round(float(20 * np.log10(part.max() / whole.max())), 1)
        return out

    def turn_extra(self) -> dict:
        return {"trim_ms": self.last_trim_ms, "cut_peak_db": self.last_cut_peak_db}


class TrimmedStreamingCascade(FirstTrimmed, StreamingCascade):
    """R1 with its first piece's leading silence trimmed (M38), on the laptop's CPU."""

    rung_id = "r1-trim"
    description = "R1, its reply's first piece cut to 20 ms before its first sound (Kokoro's 320 ms padding)"


class TrimmedStreamingCascadeGPU(FirstTrimmed, StreamingCascadeGPU):
    """R1 on a CUDA GPU with its first piece's leading silence trimmed (M38)."""

    rung_id = "r1-gpu-trim"
    description = "R1 on a CUDA GPU (fp16), its reply's first piece cut to 20 ms before its first sound"
