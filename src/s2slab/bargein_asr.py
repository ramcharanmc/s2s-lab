"""
Streaming recognition for barge-in: what had the recognizer said, and BY WHEN?

B1 reads the caller's words while the caller is still speaking. Whisper is not
a streaming model, so this does what production systems do with one: transcribe
the sound so far, again and again, as it grows.

    caller:      |--- "no, wait" --------|
    call 1            [====]  delivered -> Partial(at_s, "No")
    call 2                    [========]  delivered -> Partial(at_s, "No, wait.", final)

THREE THINGS THIS KEEPS HONEST.
  The recognizer takes time.  A transcript of audio up to t is delivered at t
      plus however long the call took -- about 500 ms for Whisper base.en on a
      laptop CPU, about 70 ms on a T4 -- and the policy cannot read it before.
  One call at a time.         A new call starts only when the last has returned,
      so a slow recognizer also gets FEWER looks at the caller.
  No label.                   Calls are triggered by the policy's own causal VAD
      (ctx.sound_ms_at), never by where the label says the caller started.

latency_ms=None charges each call what it really took on this machine;
a number charges that instead, to ask "what would this policy do on a GPU?"
without one. Either way the transcripts themselves come from the real model.
"""

from __future__ import annotations

import json
import pathlib
import time

from .bargein_policies import SOUND_PROB, BargeInContext, Partial

STEP_MS = 200.0          # at most one call per this, however fast the recognizer
MIN_SOUND_MS = 200.0     # no call before the caller has made this much sound
LEAD_S = 0.2             # audio kept before the event's first frame: soft onsets sit under the VAD
SETTLED_MS = 160.0       # this much quiet at the end of the audio: the last word is a whole word
CACHE_ROOT = pathlib.Path("results/bargein")



def stream_partials(ctx: BargeInContext, recognizer, latency_ms: float | None = None) -> list[Partial]:
    """Every transcript the recognizer would deliver while the policy listens to one item.

    Takes:   the item's context; anything with transcribe(audio) -> str; and
             what to charge per call (None: what it really took).
    Returns: Partials in delivery order. `final` means the audio ended in at
             least SETTLED_MS of quiet, so its last word can be trusted.
    """
    out: list[Partial] = []
    free_at, last_call_s, quiet_ms = -1.0, -1e9, 0.0
    event: list[float] | None = None              # [when it began, when its last sound frame ended]
    heard_until, heard_final = -1.0, False        # what the last call covered, and whether it was settled
    owed: tuple[float, float] | None = None       # an event that ended before it was fully heard
    first = max(1, int(ctx.start_s * 1000 / ctx.frame_ms))
    last = min(len(ctx.speech_prob), int(ctx.end_s * 1000 / ctx.frame_ms))

    def call(start_s: float, t_s: float, final: bool) -> None:
        nonlocal free_at, last_call_s, heard_until, heard_final
        audio = ctx.audio[max(0, int((start_s - LEAD_S) * ctx.sr)): int(t_s * ctx.sr)]
        t0 = time.perf_counter()
        text = recognizer.transcribe(audio)
        took_s = (time.perf_counter() - t0) if latency_ms is None else latency_ms / 1000
        free_at, last_call_s, heard_until, heard_final = t_s + took_s, t_s, t_s, final
        out.append(Partial(t_s + took_s, text, final, start_s))

    for k in range(first, last + 1):
        t_s = k * ctx.frame_ms / 1000
        sounding = ctx.speech_prob[k - 1] >= SOUND_PROB
        quiet_ms = 0.0 if sounding else quiet_ms + ctx.frame_ms
        started = ctx.event_start_at(t_s)
        if started is not None:
            if event is None or event[0] != started:
                event, heard_final = [started, t_s], False
            if sounding:
                event[1] = t_s
        elif event is not None:                   # the event has just ended
            if heard_until < event[1] or not heard_final:
                owed = (event[0], t_s)
            event = None
        if t_s < free_at:
            continue                              # the recognizer is still busy with the last call
        if owed is not None:
            call(owed[0], owed[1], final=True)
            owed = None
        elif (event is not None and ctx.sound_ms_at(t_s) >= MIN_SOUND_MS
              and t_s - last_call_s >= STEP_MS / 1000):
            settled = quiet_ms >= SETTLED_MS
            if heard_until < event[1] or (settled and not heard_final):
                call(event[0], t_s, final=settled)
    if owed is not None:                          # the window closed while the recognizer was busy
        call(owed[0], owed[1], final=True)
    return sorted(out)



def partials_for(contexts: dict[str, BargeInContext], recognizer_factory, latency_ms: float | None,
                 cache: pathlib.Path | None = None, refresh: bool = False) -> dict[str, list[Partial]]:
    """stream_partials() for every item, kept in a JSON cache so a policy can be
    re-scored without running the recognizer again.

    Takes:   {item id: context}; a function that returns a LOADED recognizer
             (only called if something is missing from the cache); the latency
             to charge; where the cache lives.
    Returns: {item id: [Partial, ...]}.
    """
    cached = {}
    if cache is not None and cache.exists() and not refresh:
        cached = {i: [Partial(*p) for p in ps] for i, ps in json.loads(cache.read_text(encoding="utf-8")).items()}
    missing = [i for i in contexts if i not in cached]
    if missing:
        recognizer = recognizer_factory()
        for item_id in missing:
            cached[item_id] = stream_partials(contexts[item_id], recognizer, latency_ms)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({i: [list(p) for p in ps] for i, ps in cached.items()}, indent=1)
                             + "\n", encoding="utf-8")
    return {i: cached[i] for i in contexts}


def whisper() -> object:
    """R0's recognizer, loaded and warmed up: the first call pays one-time costs."""
    import numpy as np

    from .r0 import WhisperASR
    asr = WhisperASR()
    asr.load()
    asr.transcribe(np.zeros(16_000, dtype=np.float32))
    return asr


def cache_path(set_name: str, latency_ms: float | None) -> pathlib.Path:
    mode = "measured" if latency_ms is None else f"{latency_ms:.0f}ms"
    return CACHE_ROOT / f"partials-{set_name}-{mode}.json"