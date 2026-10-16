"""
M31, step 2: the streamed R4 on development clips, before anything is registered.

One container, one set of weights, on the 30 probe clips (development data; the eval set is not
touched): R4 as it is, then r4-gpu-fast, then r4-gpu-stream at each of SETTINGS (the first block and
the later blocks, in codes). For every turn: total, first audio, first sound, stalls, the reply's
length; whether the stream's codes are the fast rung's, code for code; and the words: Whisper small.en
on the reply audio against the Thinker's own text (the same text in all three, the Thinker untouched),
so a stream that garbles its seams shows as a higher word error rate than R4's.

The stream runs with no start buffer. What a start buffer of B seconds would do is computed from each
turn's pieces afterwards (buffered): the first piece waits until B seconds of audio exist, then
everything plays on, so every buffer is evaluated on the very same turns.

Three replies of each rung are kept as WAVs for listening (results/, never committed).

    modal run modal_app/r4_bench.py::streamdev      one L40S, about 25 min; saves runs/r4-gpu/stream-dev-l40s.json
"""

from __future__ import annotations

import io
import math

import numpy as np

from .r4 import OUT_SR

SETTINGS = ((24, 96), (24, 192), (48, 96))     # (first block, later blocks), in codes; multiples of the DiT's 12
BUFFERS_S = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
LISTEN = 3                                       # replies of each rung kept as WAVs


def buffered(chunks: list, sr: int, buffer_s: float, lead_ms: float | None) -> dict:
    """What a start buffer would have done to one streamed turn.

    Takes:   its pieces as [(ms when made, samples)], the rate, the buffer, the reply's leading silence.
    Returns: {"first_audio_ms", "first_sound_ms", "stall_ms"} as rungs.Turn would compute them had the
             rung held its first pieces back until buffer_s seconds existed (StartBuffer).
    """
    need, held, out = int(buffer_s * sr), 0, []
    for ms, n in chunks:
        if out and held >= need:
            out.append((ms, n))
            continue
        held += n
        if held >= need:
            out.append((ms, held))
    if not out and chunks:                       # the reply is shorter than the buffer: all of it at the end
        out = [(chunks[-1][0], sum(n for _, n in chunks))]
    return {"first_audio_ms": out[0][0] if out else math.nan, **heard(out, sr, lead_ms)}


def heard(out: list, sr: int, lead_ms: float | None) -> dict:
    """Pieces handed out at [(ms, samples)], played back to back from the first, scored as rungs.Turn scores a
    turn: {"first_sound_ms", "stall_ms"}. buffered's, and r4timed's for a planned start (M37)."""
    first = out[0][0] if out else math.nan
    stall, cursor = 0.0, None
    for ms, n in out:
        if cursor is not None and ms > cursor:
            stall += ms - cursor
            cursor = ms
        cursor = (ms if cursor is None else cursor) + n / sr * 1000
    sound = math.nan
    if lead_ms is not None and not math.isnan(lead_ms) and out:
        voice_at, start = int(lead_ms / 1000 * sr), 0
        for ms, n in out:
            if voice_at < start + n:
                sound = max(first + lead_ms, ms)
                break
            start += n
    return {"first_sound_ms": sound, "stall_ms": stall}


def to_16k(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(audio, dtype=np.float32), 2, 3).astype(np.float32)   # 24 kHz -> 16 kHz


def wav_bytes(audio: np.ndarray, sr: int = OUT_SR) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def turn_record(turn, heard: str, codes: list[int] | None) -> dict:
    from .wer import wer
    e = wer(turn.reply_text or "", heard)
    lead = turn.leading_silence_ms
    return {"total_ms": round(turn.total_ms, 1), "first_audio_ms": round(turn.first_audio_ms, 1),
            "first_sound_ms": None if math.isnan(turn.first_sound_ms) else round(turn.first_sound_ms, 1),
            "stall_ms": round(turn.stall_ms, 1), "reply_s": round(turn.reply_s, 3),
            "leading_silence_ms": None if math.isnan(lead) else round(lead, 1),
            "chunks": [[round(ms, 1), n] for ms, n in turn.chunks],
            "reply_text": turn.reply_text, "heard": heard,
            "words": e.n_words, "word_errors": e.errors, "n_codes": None if codes is None else len(codes)}


class _CodeTap:
    """Keeps the codes the whole-reply token2wav is given (r4-gpu-fast's), to compare with the stream's."""

    def __init__(self, token2wav):
        self.t2w, self.codes = token2wav, None
        inner = token2wav.forward

        def forward(code, *a, **k):
            self.codes = [int(c) for c in np.asarray(code.detach().cpu()).reshape(-1)]
            return inner(code, *a, **k)

        token2wav.forward = forward

    def close(self):
        del self.t2w.forward


def run(clips: dict, sr: int, asr=None, base=None, settings=SETTINGS, make_fast=None,
        start_buffer_s: float = 0.0, listen: int = LISTEN) -> dict:
    """R4, r4-gpu-fast and r4-gpu-stream at each setting, every clip, one set of weights (see the docstring).
    asr: anything with load() and transcribe(audio at 16 kHz) -> str (default Whisper small.en on CUDA);
    make_fast(base) -> the fast model on R4's weights (default FastOmni(base)); start_buffer_s: the
    stream's own start buffer (development: 0, and the buffers computed afterwards; PR-015: frozen);
    listen: replies of each rung kept as WAVs."""
    from .r0 import WhisperASR
    from .r4 import OmniS2S, SpeechToSpeech
    from .r4stream import FastOmni, FastSpeechToSpeech, StreamingSpeechToSpeech, check_blocks
    asr = asr or WhisperASR("small.en", device="cuda", compute_type="float16")
    asr.load()
    base = base or OmniS2S()
    rungs: dict = {}
    listened: dict = {}
    keep = listen
    fast_codes: dict = {}

    def through(name, rung, codes_of):
        load_ms = rung.load()
        rows = {}
        for i, (clip, audio) in enumerate(clips.items()):
            turn = rung.respond(audio, sr)
            codes = codes_of()
            rows[clip] = turn_record(turn, asr.transcribe(to_16k(turn.audio)), codes)
            if name == "r4-gpu-fast":
                fast_codes[clip] = codes
            elif name.startswith("r4-gpu-stream") and codes is not None and clip in fast_codes:
                rows[clip]["codes_as_fast"] = codes == fast_codes[clip]
            if i < keep:
                listened[f"{name}/{clip}.wav"] = wav_bytes(turn.audio)
        rungs[name] = {"load_ms": round(load_ms, 1), "turns": rows}

    r4 = SpeechToSpeech(base)
    r4.load()
    check_blocks(base.model, settings)                 # before any turn: a bad setting costs nothing
    through("r4-gpu", r4, lambda: None)
    fast = make_fast(base) if make_fast else FastOmni(base)
    fast_rung = FastSpeechToSpeech(fast)
    fast_rung.load()
    tap = _CodeTap(base.model.token2wav)
    try:
        through("r4-gpu-fast", fast_rung, lambda: tap.codes)
    finally:
        tap.close()
    for first, block in settings:
        rung = StreamingSpeechToSpeech(fast, first=first, block=block, start_buffer_s=start_buffer_s)
        through(f"r4-gpu-stream/{first}+{block}", rung, lambda r=rung: r.last_codes)
    for name, r in rungs.items():
        if name.startswith("r4-gpu-stream") and start_buffer_s == 0:
            r["buffers"] = {f"{b:g}": {clip: buffered(t["chunks"], OUT_SR, b, t["leading_silence_ms"])
                                       for clip, t in r["turns"].items()} for b in BUFFERS_S}
    return {"clips": list(clips), "settings": [list(s) for s in settings], "buffers_s": list(BUFFERS_S),
            "rungs": rungs, "listen": listened}


def _pct(values, p):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.percentile(v, p)), 1) if v else None


def summary(blob: dict) -> dict:
    """Per rung: first sound p50/p90, stalls (turns with any, p90), total, the word error rate; per stream
    setting and buffer: first sound and stalls as the buffer would make them."""
    out = {}
    for name, r in blob["rungs"].items():
        t = list(r["turns"].values())
        words, errs = sum(x["words"] for x in t), sum(x["word_errors"] for x in t)
        row = {"n": len(t), "first_sound_p50": _pct([x["first_sound_ms"] for x in t], 50),
               "first_sound_p90": _pct([x["first_sound_ms"] for x in t], 90),
               "total_p50": _pct([x["total_ms"] for x in t], 50),
               "stalled": sum(x["stall_ms"] > 0 for x in t), "stall_p90": _pct([x["stall_ms"] for x in t], 90),
               "reply_s_p50": _pct([x["reply_s"] for x in t], 50),
               "wer": round(100 * errs / words, 1) if words else None,
               "codes_as_fast": (sum(bool(x.get("codes_as_fast")) for x in t) if name.startswith("r4-gpu-stream")
                                 else None)}
        if "buffers" in r:
            row["buffers"] = {b: {"first_sound_p50": _pct([v["first_sound_ms"] for v in per.values()], 50),
                                  "first_sound_p90": _pct([v["first_sound_ms"] for v in per.values()], 90),
                                  "stalled": sum(v["stall_ms"] > 0 for v in per.values()),
                                  "stall_p90": _pct([v["stall_ms"] for v in per.values()], 90)}
                              for b, per in r["buffers"].items()}
        out[name] = row
    return out


def format_dev(blob: dict) -> str:
    s = summary(blob)
    f = lambda x: "-" if x is None else f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 10 else str(x)
    n = len(blob["clips"])
    lines = [f"R4, fast and streamed, on {n} probe clips (development)", "",
             "| rung | first sound p50 ms | p90 | stalled turns | stall p90 ms | total p50 ms | reply s p50 | WER % | codes as fast's |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in s.items():
        cf = "-" if r["codes_as_fast"] is None else f"{r['codes_as_fast']} of {r['n']}"
        lines.append(f"| {name} | {f(r['first_sound_p50'])} | {f(r['first_sound_p90'])} | {r['stalled']} of {r['n']} | "
                     f"{f(r['stall_p90'])} | {f(r['total_p50'])} | {r['reply_s_p50']} | {r['wer']} | {cf} |")
    for name, r in s.items():
        if "buffers" not in r:
            continue
        lines += ["", f"{name}, with a start buffer (computed from the same turns):", "",
                  "| buffer s | first sound p50 ms | p90 | stalled turns | stall p90 ms |", "|---:|---:|---:|---:|---:|"]
        for b, v in r["buffers"].items():
            lines.append(f"| {b} | {f(v['first_sound_p50'])} | {f(v['first_sound_p90'])} | {v['stalled']} of {r['n']} | {f(v['stall_p90'])} |")
    lines.append("\nWER: Whisper small.en on the reply audio against the Thinker's text; the same text for every rung")
    return "\n".join(lines)
