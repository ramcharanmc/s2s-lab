"""
The rung registry: every speech-to-speech system on the ladder, behind one call.

The v2 plan (§10): "Every rung implements respond(audio) -> Turn, so one
benchmark runs across all of them. If a rung needs a special case in the
benchmark, the interface is wrong." It is the rule policies.py follows with
decide(ctx), one level up.

A rung takes ONE FINISHED user turn -- the endpointer has already decided the
speaker stopped -- and returns the spoken reply plus where the time went:

    turn = get("r0").respond(audio, 16_000)
    turn.transcript      "i would like to change the delivery address"
    turn.reply_text      "Sure -- what is the new address?"
    turn.stages_ms       {"asr": 810.4, "llm": 1432.9, "tts": 655.1}
    turn.total_ms        2901.7

WHAT total_ms IS AND IS NOT. The clock starts when the rung receives the
finished turn and stops when the reply audio exists. What the user waits is
longer on both sides:
  * in front, the endpointer's wait -- 500 ms for T0_fixed_500ms (page 18);
  * behind, any silence at the start of the reply. TTS engines pad their
    output, and no stage timer sees it. Turn.leading_silence_ms measures it.

LOADING IS NOT LATENCY. Models load once per process, in load(), before the
clock starts. Folding that into the first turn would make its number describe
the disk rather than the pipeline. Load time is reported on its own line: it is
the cold-start cost the plan measures in week 8.

FIRST AUDIO. A caller starts hearing the reply when its first piece exists,
not its last. Two ways to hand audio back, and the clock sees both:
  * batch (R0)      return the whole reply in Reply.audio. It exists when
                    _respond() returns, so first audio = total, by construction.
  * streaming (R1)  call self.emit(chunk) the moment each piece exists, and
                    return Reply(None, ...). The base class stamps each chunk
                    and joins them.
Turn.first_audio_ms is when the first piece existed; Turn.first_sound_ms adds
the silence before the first voiced sample -- what the caller waits to HEAR.

respond(audio, sr)
 ├─ load()                            once per rung, BEFORE the clock starts
 ├─ t0 = perf_counter()
 ├─ self._respond(audio, sr, sink)    subclass: each stage inside stage(name, sink);
 │    → Reply(audio, sr, ...)         streaming rungs call self.emit(chunk) as they go
 └─ Turn(..., stages_ms=per_stage_ms(sink), total_ms=perf_counter() - t0,
         chunks=[(ms when it existed, samples), ...])
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .timing import LatencySummary, format_table, per_stage_ms, stage, summarize
from .vad import FRAME_MS, frame_rms

SILENCE_DB = -40.0       # a reply frame this far below the reply's loudest is silence
ENDPOINT_WAIT_S = 0.5    # what T0_fixed_500ms waits after the last word before handing over

def silence_before_speech_ms(audio: np.ndarray, sr: int) -> float:
    """How long a reply stays silent before it says anything, in ms.

    Takes:   the reply audio and its sample rate.
    Calls:   vad.frame_rms() -- one RMS value per 20 ms frame.
    Returns: the start of the first frame within 40 dB of the reply's loudest
             frame. Measured against the reply itself, so the output level does
             not matter. NaN if nothing rises above silence: a reply that never
             speaks has no time-to-speech, and summarize() drops NaN instead of
             counting it as fast.
    """
    rms = frame_rms(np.asarray(audio, dtype=np.float32), sr)
    if rms.size == 0 or rms.max() < 1e-5:          # empty, or digital silence
        return math.nan
    loud = rms >= rms.max() * 10 ** (SILENCE_DB / 20)
    return float(np.argmax(loud)) * FRAME_MS        # argmax: index of the first True

class Reply(NamedTuple):
    """What a rung's _respond() hands back to respond()."""

    audio: np.ndarray | None  # the whole reply; None if it was handed out through emit()    
    sr: int
    transcript: str | None    # what it heard; None if the input never becomes text (R3 up)
    reply_text: str | None    # what it chose to say; None if it speaks without text (R4, R5)


@dataclass
class Turn:
    """One finished user turn in, one spoken reply out, and where the time went."""

    audio: np.ndarray             # the reply, float32 mono
    sr: int                       # its sample rate -- Kokoro speaks at 24 kHz, not 16
    transcript: str | None
    reply_text: str | None
    stages_ms: dict[str, float]   # {"asr": 810.4, "llm": 1432.9, "tts": 655.1}, in run order
    total_ms: float               # the whole turn, on the same clock as the stages
    chunks: list[tuple[float, int]]  # [(ms when it existed, samples)], non-empty pieces only

    @property
    def untimed_ms(self) -> float:
        """Time inside the turn that no stage accounts for.

        Glue code, format conversion, resampling. Near zero is healthy. If it
        grows, work is happening outside every stage, and the stage table no
        longer adds up to what the user waited.
        """
        return self.total_ms - sum(self.stages_ms.values())

    @property
    def reply_s(self) -> float:
        """Length of the spoken reply, in seconds."""
        return len(self.audio) / self.sr

    @property
    def leading_silence_ms(self) -> float:
        return silence_before_speech_ms(self.audio, self.sr)

    @property
    def first_audio_ms(self) -> float:
        """When the first piece of reply audio existed. NaN if there was none."""
        return self.chunks[0][0] if self.chunks else math.nan

    @property
    def first_sound_ms(self) -> float:
        """When the caller would first HEAR the reply, in ms from the turn's start.

        Playback starts at first_audio_ms, so the first voiced sample plays
        leading_silence_ms later -- unless the chunk holding it did not exist
        yet, in which case it plays when that chunk arrives. Batch: total +
        leading silence. NaN if the reply never speaks.
        """
        lead_ms = self.leading_silence_ms
        if math.isnan(lead_ms):
            return math.nan
        voice_at = int(lead_ms / 1000 * self.sr)       # first voiced sample, in the joined reply
        start = 0
        for made_ms, n in self.chunks:
            if voice_at < start + n:
                return max(self.first_audio_ms + lead_ms, made_ms)
            start += n
        return math.nan


    @property
    def stall_ms(self) -> float:
        """How long playback would run dry waiting for the next piece, in ms.

        Play the pieces back to back from the moment the first one exists. If
        the next piece does not exist when the one before finishes playing, the
        caller hears extra silence until it does; this adds those waits up. A
        batch reply is one piece, so 0.0. An early first sound bought with
        stalls later is not a win, so every streaming result reports both.
        """
        stall, cursor = 0.0, None
        for made_ms, n in self.chunks:
            if cursor is not None and made_ms > cursor:
                stall += made_ms - cursor
                cursor = made_ms
            cursor = (made_ms if cursor is None else cursor) + n / self.sr * 1000
        return stall

class Rung:
    """Base class for every rung. Subclasses set rung_id and implement _respond().

    Override _load() to fetch and initialise models; the stub has none.
    """

    rung_id = "?"
    description = ""
    _loaded = False
    _emitted: list | None = None     # [(ms, audio)] during a turn; None outside respond()
    _t0 = 0.0

    def load(self) -> float:
        """Initialise the rung, once.

        Returns: how long it took in ms, or 0.0 if it was already loaded.
        """
        if self._loaded:
            return 0.0
        t0 = time.perf_counter()
        self._load()
        self._loaded = True
        return (time.perf_counter() - t0) * 1000

    def _load(self) -> None:
        pass

    def respond(self, audio: np.ndarray, sr: int) -> Turn:
        """Answer one finished user turn.

        Takes:   mono audio of the whole turn, and its sample rate.
        Calls:   load() if nobody has yet -- before the clock starts;
                 self._respond(), which times each of its stages into `sink`.
        Returns: a Turn: the reply, per-stage times, and the total.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {audio.shape}")
        self.load()
        sink: list = []
        self._emitted = []
        self._t0 = t0 = time.perf_counter()
        try:
            reply = self._respond(audio, sr, sink)
            total_ms = (time.perf_counter() - t0) * 1000
            emitted = self._emitted
        finally:
            self._emitted = None                  # emit() outside a turn is now an error
        if reply.audio is None:                   # streaming: the pieces are the reply
            pieces = emitted
            out = (np.concatenate([a for _, a in pieces]) if pieces
                   else np.zeros(0, dtype=np.float32))
        elif emitted:
            raise ValueError(f"{self.rung_id}: emitted audio AND returned it; "
                             "a streaming rung returns Reply(None, ...)")
        else:                                     # batch: all of it existed at the end
            out = np.asarray(reply.audio, dtype=np.float32)
            pieces = [(total_ms, out)]
        chunks = [(ms, len(a)) for ms, a in pieces if len(a)]
        return Turn(out, reply.sr, reply.transcript, reply.reply_text,
                    per_stage_ms(sink), total_ms, chunks)

    def emit(self, chunk: np.ndarray) -> None:
        """Hand out one piece of reply audio the moment it exists (streaming rungs).

        Takes:   float32 mono audio at the rung's output rate.
        Calls:   nothing -- it stamps the time and keeps the piece.
        Returns: nothing. respond() joins the pieces into Turn.audio.
        """
        if self._emitted is None:
            raise RuntimeError("emit() only works inside respond()")
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {chunk.shape}")
        self._emitted.append(((time.perf_counter() - self._t0) * 1000, chunk))

    def _respond(self, audio: np.ndarray, sr: int, sink: list) -> Reply:
        raise NotImplementedError

class StubRung(Rung):
    """Does nothing, instantly: three empty stages and half a second of silence.

    The v1 plan's Phase 0: if the harness can print a percentile table for a
    pipeline that does nothing, every later change produces a number instead
    of a vibe. Zeros are the correct answer; anything more is harness overhead.
    """

    rung_id = "stub"
    description = "three empty stages, silence out -- tests the harness itself"
    OUT_SR = 24_000

    def _respond(self, audio, sr, sink):
        with stage("asr", sink):
            transcript = ""
        with stage("llm", sink):
            reply_text = ""
        with stage("tts", sink):
            reply = np.zeros(self.OUT_SR // 2, dtype=np.float32)
        return Reply(reply, self.OUT_SR, transcript, reply_text)

# id -> "module:Class". A rung's module is imported only when that rung is
# asked for, so the stub, the tests and the whole CPU track never import torch.
RUNGS = {
    "stub": "s2slab.rungs:StubRung",
    "r0": "s2slab.r0:NaiveCascade",
    "r0-gpu": "s2slab.r0:NaiveCascadeGPU",
    "r0-gpu-bigger": "s2slab.r0:NaiveCascadeGPUBigger",
    "r1": "s2slab.r1:StreamingCascade",
    "r1-gpu": "s2slab.r1:StreamingCascadeGPU",
    "r2": "s2slab.r2:TwoTierCascade",
    "r2-turbo": "s2slab.r2:TwoTierCascadeTurbo",
    "r2-gpu": "s2slab.r2:TwoTierCascadeGPU",
    "r2-gpu-turbo": "s2slab.r2:TwoTierCascadeGPUTurbo",
    "r2-verbatim": "s2slab.r2verbatim:TwoTierCascadeVerbatim",
    "r2-gpu-verbatim": "s2slab.r2verbatim:TwoTierCascadeGPUVerbatim",
    "r3-gpu": "s2slab.r3:SpeechLLMCascade",
    "r4-gpu": "s2slab.r4:SpeechToSpeech",
    "r4-gpu-fast": "s2slab.r4stream:FastSpeechToSpeech",
    "r4-gpu-stream": "s2slab.r4stream:StreamingSpeechToSpeech",
    "r4-gpu-stream-sized": "s2slab.r4sized:SizedStreamingSpeechToSpeech",
    "r4-gpu-stream-timed": "s2slab.r4timed:TimedStreamingSpeechToSpeech",
}

def get(rung_id: str) -> Rung:
    """A fresh, unloaded instance of the named rung."""
    try:
        module_name, class_name = RUNGS[rung_id].split(":")
    except KeyError:
        raise KeyError(f"unknown rung {rung_id!r}; known: {', '.join(RUNGS)}") from None
    return getattr(importlib.import_module(module_name), class_name)()



def run_turns(rung: Rung, clips: dict[str, np.ndarray], sr: int) -> dict[str, Turn]:
    """Every clip through the rung, one turn each, in the order given."""
    return {clip_id: rung.respond(audio, sr) for clip_id, audio in clips.items()}


def summarize_turns(turns: dict[str, Turn]) -> dict[str, LatencySummary]:
    """Percentiles for each stage across turns, plus '(untimed)', 'total',
    'first audio' and 'first sound'. A row with no finite value at all -- first
    sound for a rung that never speaks -- is left out rather than printed as nan."""
    rows: dict[str, list[float]] = {}
    for t in turns.values():
        for name, ms in t.stages_ms.items():
            rows.setdefault(name, []).append(ms)
        rows.setdefault("(untimed)", []).append(t.untimed_ms)
        rows.setdefault("total", []).append(t.total_ms)
        rows.setdefault("first audio", []).append(t.first_audio_ms)
        rows.setdefault("first sound", []).append(t.first_sound_ms)
    summaries = {name: summarize(values) for name, values in rows.items()}
    return {name: s for name, s in summaries.items() if s.n}


def probe_clips(root=None) -> dict[str, np.ndarray]:
    """Each probe recording, as an endpointer would hand it to a rung.

    Takes:   the probe folder (default: probes/endpointing).
    Calls:   probes.load_set() -- already cropped to just before the first word.
    Returns: {"f01": audio, ...} at 16 kHz, each cut ENDPOINT_WAIT_S after the
             true end of turn: what a rung behind T0_fixed_500ms receives when
             the endpointer gets the turn right.
    """
    from .probes import PROBE_ROOT, TARGET_SR, load_set
    items = load_set(root or PROBE_ROOT)
    return {item_id: u.audio[: int((u.true_end_s + ENDPOINT_WAIT_S) * TARGET_SR)]
            for item_id, u in items.items()}


def report(rung_id: str, clips: dict[str, np.ndarray], sr: int) -> str:
    """Load one rung, run every clip through it, and print where the time went.

    Takes:   a registered rung id, clips {id: mono audio}, their sample rate.
    Calls:   get(), load(), run_turns(), summarize_turns(), format_table().
    Returns: the stage table, then load time and reply leading silence --
             kept out of the table because neither is a stage of a turn.
    """
    rung = get(rung_id)
    load_ms = rung.load()
    turns = run_turns(rung, clips, sr)
    return "\n".join([
        format_table(summarize_turns(turns), title=f"{rung_id}: {rung.description}"),
        "",
        f"load: {load_ms:.0f} ms, once, before the first turn (in no row above)",
        f"reply leading silence: {summarize(t.leading_silence_ms for t in turns.values())}",
        f"playback stall: {summarize(t.stall_ms for t in turns.values())}  "
        f"({sum(t.stall_ms > 0 for t in turns.values())} of {len(turns)} turns ran dry)",
    ])



if __name__ == "__main__":
    import sys

    from .probes import TARGET_SR

    print(report(sys.argv[1] if len(sys.argv) > 1 else "stub", probe_clips(), TARGET_SR))