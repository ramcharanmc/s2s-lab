"""
M31, step 1: where R4's wait goes, stage by stage, before anything streams.

R4 (s2slab.r4) calls Qwen2.5-Omni's generate once, and inside it three stages run one after
another, each to its end before the next begins:

    thinker     writes the whole text reply (greedy, at most 60 tokens)
    talker      writes every speech code for it (sampled, seeded)
    token2wav   turns all the codes into audio: a DiT makes a mel spectrogram (10 solver steps,
                in fp32), then BigVGAN makes the 24 kHz waveform

So the first sample exists only when the last stage ends: on the eval set R4 waited 0.93 s for
every second of its reply (M28). A streaming R4 (M31's design, S2) would hand the Talker's codes
to token2wav a block at a time. Whether that can keep up with playback, and how soon its first
block could sound, depends on how the time divides between the stages and on how long one block
takes to decode.

This measures that and nothing else: each stage timed (CUDA synchronised) inside R4's own call,
with R4's own settings, on the 30 probe clips; then token2wav alone on the first BLOCKS codes of
each reply, twice (the first call at a new length can include one-off kernel set-up). Nothing
streams yet, and the replies are R4's: the timers only wrap the calls. The estimates are
arithmetic on these times, not a measurement of a streaming rung, and they leave out what a
block's neighbours would add to its decode.

    modal run modal_app/r4_bench.py::profile        one L40S; saves runs/r4-gpu/profile-l40s.json
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from .r4 import OUT_SR

STAGES = ("thinker", "talker", "token2wav", "dit", "bigvgan")   # dit and bigvgan are token2wav's two parts
BLOCKS = (25, 50, 100)        # first-block sizes, in Talker codes, that token2wav is timed on alone
PROFILE = pathlib.Path("runs/r4-gpu/profile-l40s.json")


def _cuda_sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageClock:
    """Wraps the model's stages with timers that synchronise the GPU before and after each call.

    Takes:   the Qwen2.5-Omni model (OmniS2S.model): its thinker, talker and token2wav, and
             token2wav's DiT and BigVGAN. Each wrapper calls the original and returns its output
             unchanged. While a turn is open (start() to stop()), it adds each call's milliseconds
             to that stage, and keeps two small facts: how many text tokens the Thinker wrote, and
             the codes and keyword arguments token2wav was given (to decode a first block later).
    """

    def __init__(self, model, sync=_cuda_sync):
        self.model, self.sync, self.open = model, sync, False
        self.ms: dict[str, float] = {}
        self.text_tokens: int | None = None
        self.t2w_call: tuple | None = None
        self._wrap(model.thinker, "generate", "thinker")
        self._wrap(model.talker, "generate", "talker")
        self._wrap(model.token2wav, "forward", "token2wav")
        self._wrap(model.token2wav.code2wav_dit_model, "sample", "dit")
        self._wrap(model.token2wav.code2wav_bigvgan_model, "forward", "bigvgan")

    def _wrap(self, obj, name: str, label: str) -> None:
        inner = getattr(obj, name)

        def timed(*args, **kwargs):
            if not self.open:
                return inner(*args, **kwargs)
            self.sync()
            t0 = time.perf_counter()
            out = inner(*args, **kwargs)
            self.sync()
            self.ms[label] = self.ms.get(label, 0.0) + (time.perf_counter() - t0) * 1000
            if label == "thinker":
                self.text_tokens = int(out.sequences.shape[1] - kwargs["input_ids"].shape[1])
            elif label == "token2wav":
                self.t2w_call = (args[0] if args else kwargs["code"],
                                 {k: v for k, v in kwargs.items() if k != "code"})
            return out

        setattr(obj, name, timed)          # an instance attribute: the model's own calls find it first

    def start(self) -> None:
        self.ms, self.text_tokens, self.t2w_call, self.open = {}, None, None, True

    def stop(self) -> dict[str, float]:
        self.open = False
        return {k: round(v, 1) for k, v in self.ms.items()}

    def time_call(self, fn, *args, **kwargs) -> tuple[float, object]:
        """One call, synchronised, outside any turn: (milliseconds, its output)."""
        self.sync()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.sync()
        return (time.perf_counter() - t0) * 1000, out


def profile_turn(omni, clock: StageClock, audio: np.ndarray, sr: int, blocks=BLOCKS) -> dict:
    """One clip through R4's own call, timed by stage; then token2wav on its first blocks.

    Returns: {"total_ms", "stages_ms": {stage: ms}, "text_tokens", "codes", "reply_s", "words",
              "block_ms": {"25": {"first", "again"}, or None when the reply has fewer codes, or {"error"}},
              "samples_per_code"}.
    """
    clock.start()
    t0 = time.perf_counter()
    text, wav = omni.speak(audio, sr)
    total_ms = (time.perf_counter() - t0) * 1000
    stages = clock.stop()
    code, kwargs = clock.t2w_call
    n_codes = int(code.shape[-1])
    rec = {"total_ms": round(total_ms, 1), "stages_ms": stages, "text_tokens": clock.text_tokens,
           "codes": n_codes, "reply_s": round(len(wav) / OUT_SR, 3), "words": len(text.split()),
           "block_ms": {}, "samples_per_code": None}
    import torch
    t2w = omni.model.token2wav
    for b in blocks:
        if n_codes < b:
            rec["block_ms"][str(b)] = None
            continue
        try:
            with torch.inference_mode():                         # as inside R4's own call
                first, out = clock.time_call(t2w, code[..., :b], **kwargs)
                again, _ = clock.time_call(t2w, code[..., :b], **kwargs)
        except Exception as e:                                   # recorded, so one block cannot end a paid run
            rec["block_ms"][str(b)] = {"error": f"{type(e).__name__}: {e}"[:300]}
            continue
        rec["block_ms"][str(b)] = {"first": round(first, 1), "again": round(again, 1)}
        if rec["samples_per_code"] is None:
            rec["samples_per_code"] = round(int(np.prod(out.shape)) / b, 2)
    clock.t2w_call = None
    return rec


def estimates(rec: dict, blocks=BLOCKS) -> dict:
    """Arithmetic on one turn's times (not a measurement of a streaming rung).

    codes_per_s        the Talker's codes per second of reply audio
    talker_rtf         the Talker's milliseconds per millisecond of reply
    token2wav_rtf      the same for token2wav decoding the whole reply at once
    for each block size b (when the reply has b codes):
      first_audio_ms   thinker + b codes' share of the Talker + token2wav on the first b codes:
                       when a streaming R4's first block could exist (the Talker's start-up is
                       spread over the codes here, so this is a little early)
      stream_rtf       talker_rtf + token2wav's milliseconds per millisecond of audio on blocks of
                       b codes: below 1, a stream with no neighbours to decode keeps up with playback
    """
    st = rec["stages_ms"]
    reply_ms = rec["reply_s"] * 1000
    if not reply_ms or not rec["codes"]:
        return {"codes_per_s": None, "talker_rtf": None, "token2wav_rtf": None, "blocks": {}}
    codes_per_s = rec["codes"] / rec["reply_s"]
    talker_rtf = st.get("talker", 0.0) / reply_ms
    out = {"codes_per_s": round(codes_per_s, 2), "talker_rtf": round(talker_rtf, 3),
           "token2wav_rtf": round(st.get("token2wav", 0.0) / reply_ms, 3), "blocks": {}}
    per_code = st.get("talker", 0.0) / rec["codes"]
    for b in blocks:
        bm = rec["block_ms"].get(str(b))
        if not bm or "again" not in bm:
            continue
        block_audio_ms = b / codes_per_s * 1000
        out["blocks"][str(b)] = {"first_audio_ms": round(st.get("thinker", 0.0) + b * per_code + bm["again"], 1),
                                 "stream_rtf": round(talker_rtf + bm["again"] / block_audio_ms, 3)}
    return out


def run(omni, clips: dict[str, np.ndarray], sr: int, blocks=BLOCKS) -> dict:
    """Load R4's model, one throwaway turn (as R4's own load does), then every clip.
    Returns {"load_ms", "blocks", "turns": {clip: record with its "estimates"}}."""
    t0 = time.perf_counter()
    omni.load()
    load_ms = (time.perf_counter() - t0) * 1000
    clock = StageClock(omni.model)
    omni.speak(np.zeros(sr, dtype=np.float32), sr)           # the throwaway turn, unwrapped as in R4
    turns = {}
    for clip, audio in clips.items():
        rec = profile_turn(omni, clock, audio, sr, blocks)
        rec["estimates"] = estimates(rec, blocks)
        turns[clip] = rec
    return {"load_ms": round(load_ms, 1), "blocks": list(blocks), "turns": turns}


def _pct(values, p) -> float | None:
    v = [x for x in values if x is not None]
    return round(float(np.percentile(v, p)), 3) if v else None


def summary(blob: dict) -> dict:
    """Median and p90 of every per-turn number, across the clips."""
    turns = list(blob["turns"].values())
    rows: dict[str, list] = {}
    for t in turns:
        rows.setdefault("total ms", []).append(t["total_ms"])
        for s in STAGES:
            rows.setdefault(f"{s} ms", []).append(t["stages_ms"].get(s))
        for k in ("text_tokens", "codes", "reply_s"):
            rows.setdefault(k, []).append(t[k])
        e = t["estimates"]
        for k in ("codes_per_s", "talker_rtf", "token2wav_rtf"):
            rows.setdefault(k, []).append(e[k])
        for b in blob["blocks"]:
            bm = t["block_ms"].get(str(b))
            rows.setdefault(f"token2wav, first {b} codes ms", []).append(bm.get("again") if bm else None)
            eb = e["blocks"].get(str(b))
            rows.setdefault(f"est. first audio, block {b} ms", []).append(eb and eb["first_audio_ms"])
            rows.setdefault(f"est. stream rtf, block {b}", []).append(eb and eb["stream_rtf"])
    return {k: {"n": sum(x is not None for x in v), "p50": _pct(v, 50), "p90": _pct(v, 90)} for k, v in rows.items()}


def share(blob: dict) -> dict[str, float]:
    """Each stage's share of all the turns' time together (thinker, talker, token2wav, the rest)."""
    turns = list(blob["turns"].values())
    total = sum(t["total_ms"] for t in turns)
    out = {s: sum(t["stages_ms"].get(s, 0.0) for t in turns) / total for s in ("thinker", "talker", "token2wav")}
    out["the rest"] = 1 - sum(out.values())
    return {k: round(v, 3) for k, v in out.items()}


def format_profile(blob: dict) -> str:
    s = summary(blob)
    fmt = lambda x: "-" if x is None else (f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}")
    lines = [f"R4's stages on {len(blob['turns'])} clips (load {blob['load_ms'] / 1000:.1f} s)", "",
             "| measure | n | p50 | p90 |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['n']} | {fmt(v['p50'])} | {fmt(v['p90'])} |" for k, v in s.items()]
    errors = sorted({bm["error"] for t in blob["turns"].values() for bm in t["block_ms"].values() if bm and "error" in bm})
    lines += ["", "share of all the time: " + ", ".join(f"{k} {v:.0%}" for k, v in share(blob).items()),
              "estimates are arithmetic on these times, not a streaming rung; stream rtf leaves out a block's neighbours"]
    lines += [f"a first block failed: {e}" for e in errors]
    return "\n".join(lines)


def save(path: pathlib.Path, blob: dict) -> pathlib.Path:
    """The whole profile, no audio. Never overwrites."""
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path
