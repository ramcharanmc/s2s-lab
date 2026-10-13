"""
M34, step 2: R4 streamed on a right-sized cache, against today's, on development clips, before anything
is registered.

One container, one set of weights, the 30 probe clips (development; the eval set is not touched):
r4-gpu-fast (the codes both streams should write), r4-gpu-stream as frozen for PR-015 (24 + 96 codes), and
r4-gpu-stream-sized (s2slab.r4sized: the same blocks, its codes from our captured loop on a cache sized to
the turn). Both streams run with no start buffer; what a buffer of B seconds would do is computed from each
turn's pieces afterwards (r4streamdev.buffered), as in M31, for every B in BUFFERS_S. For every turn: first
sound, stalls, total, the reply's length, Whisper small.en on the reply against the Thinker's text, whether
its codes are the fast rung's; for the sized stream also the cache it took, and whether its reply was cut.

THE START RULE, written before the run (docs/m34-log.md), as M31 chose PR-015's: for each stream, the
buffer with the lowest first-sound p50 among those with no stalled turn; a tie goes to the smaller buffer.

    modal run modal_app/r4_bench.py::sizeddev       one L40S; saves runs/r4-gpu/sized-dev-l40s.json
"""

from __future__ import annotations

from .r4 import OUT_SR

BUFFERS_S = (0.0, 0.25, 0.5, 1.0)
LISTEN = 3                                       # replies of each rung kept as WAVs
STREAMS = ("r4-gpu-stream", "r4-gpu-stream-sized")


def run(clips: dict, sr: int, asr=None, base=None, make_fast=None, listen: int = LISTEN,
        start_buffer_s: float = 0.0) -> dict:
    """r4-gpu-fast, then both streams, every clip, one set of weights; with no start buffer (development),
    the buffers computed from the pieces afterwards; with one (PR-016: frozen), each stream holds it itself.
    asr and make_fast as in r4streamdev.run."""
    from .r0 import WhisperASR
    from .r4 import OmniS2S
    from .r4sized import SizedStreamingSpeechToSpeech
    from .r4stream import FastOmni, FastSpeechToSpeech, StreamingSpeechToSpeech
    from .r4streamdev import _CodeTap, buffered, to_16k, turn_record, wav_bytes
    asr = asr or WhisperASR("small.en", device="cuda", compute_type="float16")
    asr.load()
    base = base or OmniS2S()
    fast = make_fast(base) if make_fast else FastOmni(base)
    rungs, listened, fast_codes = {}, {}, {}

    def through(name, rung, codes_of, extra=lambda: {}):
        load_ms = rung.load()
        rows = {}
        for i, (clip, audio) in enumerate(clips.items()):
            turn = rung.respond(audio, sr)
            codes = codes_of()
            rows[clip] = dict(turn_record(turn, asr.transcribe(to_16k(turn.audio)), codes), **extra())
            if name == "r4-gpu-fast":
                fast_codes[clip] = codes
            elif codes is not None and clip in fast_codes:
                rows[clip]["codes_as_fast"] = codes == fast_codes[clip]
            if i < listen:
                listened[f"{name}/{clip}.wav"] = wav_bytes(turn.audio)
        rungs[name] = {"load_ms": round(load_ms, 1), "turns": rows}

    fast_rung = FastSpeechToSpeech(fast)
    fast_rung.load()
    tap = _CodeTap(base.model.token2wav)
    try:
        through("r4-gpu-fast", fast_rung, lambda: tap.codes)
    finally:
        tap.close()
    stream = StreamingSpeechToSpeech(fast, start_buffer_s=start_buffer_s)
    through("r4-gpu-stream", stream, lambda: stream.last_codes)
    sized = SizedStreamingSpeechToSpeech(fast, start_buffer_s=start_buffer_s)
    through("r4-gpu-stream-sized", sized, lambda: sized.last_codes,
            lambda: {k: sized.last_turn.get(k) for k in ("places", "prompt", "cut")})
    for name in STREAMS if start_buffer_s == 0 else ():
        rungs[name]["buffers"] = {f"{b:g}": {clip: buffered(t["chunks"], OUT_SR, b, t["leading_silence_ms"])
                                             for clip, t in rungs[name]["turns"].items()} for b in BUFFERS_S}
    return {"clips": list(clips), "buffers_s": list(BUFFERS_S), "rungs": rungs, "listen": listened}


def choose(blob: dict) -> dict:
    """The start rule, per stream: the buffer with the lowest first-sound p50 among those with no stalled
    turn (a tie to the smaller buffer); None when every buffer stalls a turn."""
    from .r4streamdev import summary
    s = summary(blob)
    out = {}
    for name in STREAMS:
        ok = [(v["first_sound_p50"], float(b), b) for b, v in s[name]["buffers"].items()
              if v["stalled"] == 0 and v["first_sound_p50"] is not None]
        best = min(ok) if ok else None
        out[name] = None if best is None else {"buffer_s": best[1], "first_sound_p50": best[0],
                                               "first_sound_p90": s[name]["buffers"][best[2]]["first_sound_p90"]}
    return out


def format_sized_dev(blob: dict) -> str:
    from .r4streamdev import format_dev
    c = choose(blob)
    turns = blob["rungs"]["r4-gpu-stream-sized"]["turns"].values()
    places: dict = {}
    for t in turns:
        places[t["places"]] = places.get(t["places"], 0) + 1
    lines = [format_dev(blob), "",
             "The sized stream's caches: " + ", ".join(f"{n} places on {k} turns" for n, k in sorted(places.items()))
             + f"; replies cut: {sum(bool(t['cut']) for t in turns)}.", "", "The start rule (lowest first-sound p50, no stalled turn):"]
    for name, v in c.items():
        lines.append(f"  {name}: " + ("every buffer stalls a turn" if v is None else
                                      f"{v['buffer_s']:g} s, first sound p50 {v['first_sound_p50']:,.0f} ms, p90 {v['first_sound_p90']:,.0f} ms"))
    return "\n".join(lines)
