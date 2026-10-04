"""
M31, step 1b: can R4's two slow parts run faster with standard tools, and do they still make the same thing?

The profile (step 1, runs/r4-gpu/profile-l40s.json) found R4's wait in two places. The Talker takes
19.8 ms a code at 50 codes a second of audio: a real-time factor of 0.99, and 75% of R4's time.
token2wav takes about 890 ms for any block of 25 to 100 codes: its DiT runs 80 times a decode (10
Runge-Kutta steps of 4 calls, guidance in the batch) at about 11 ms whatever the length. A small
model as slow per step as the 7B Thinker, and a cost that does not grow with the work, both point to
time spent launching many small GPU operations rather than computing. The standard remedy is to
compile the step and replay it as a CUDA graph, which needs fixed shapes, so a static KV cache.

TALKER, three ways on the same clips, each through R4's own call with R4's seed:
    eager           R4 as it runs (no change)
    static          a static KV cache for the Talker only (generate's cache_implementation="static",
                    talker_ prefix), transformers' own compile turned off, and the Talker given the
                    plain 2-D attention mask (keep_2d_mask): see THE FIRST RUN below
    static+layers   the same, and the Talker's decoder layers (talker.model) compiled for one-code
                    steps, with no attention mask on those steps (compile_layers); the prefill stays eager
For each: ms per code, the real-time factor, the warm-up (compile) time, and whether its codes are
identical to eager's on the same clip and seed. A sampled Talker can drift to other codes from a
last-bit difference in a logit, so "identical" is reported as found, not assumed.

THE FIRST RUN (f7e8d99, runs/r4-gpu/speed-l40s.json) tried transformers' own route, a static cache
with the Talker's whole call compiled, and failed in both static variants with "TypeError: unsupported
operand type(s) for -: 'int' and 'dict'". Reproduced on the laptop with a tiny random Talker: with a
static cache, generate hands the model a dict of prepared masks, one per attention type, and the
Talker's forward computes its rotary positions from the plain 2-D mask (1 - attention_mask). Giving
it the 2-D mask back runs, with codes identical to eager's. But compiled that way, the decoder
layers saw a mask one longer at every step and recompiled until torch's limit, then ran eager. With
one reply and no padding the mask holds nothing the decoder needs: it builds the causal mask from
the cache position. Without it the tiny Talker compiled to one graph, no breaks, codes identical.

DIT: its network compiled (mode "reduce-overhead", CUDA graphs; "default" if that fails), and
token2wav run on the first 25, 50 and 100 codes of eager's replies, eager and compiled, each with
the same noise seed: milliseconds, and the largest sample difference between the two.

Nothing here changes R4 or its records. A variant that fails is recorded with its error, and the
next one runs. The estimates at the end use the fastest variant that ran.

    modal run modal_app/r4_bench.py::speed               the first run: runs/r4-gpu/speed-l40s.json
    modal run modal_app/r4_bench.py::speed --tag mask    with the mask fixes: runs/r4-gpu/speed-l40s-mask.json
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .r4profile import BLOCKS, StageClock

SPEED = pathlib.Path("runs/r4-gpu/speed-l40s.json")
EVERY = 3                         # every third probe clip: 10 of the 30, spread over the set
WARMUPS = 2                       # turns before timing a compiled variant (compile, then graph capture)
TALKER_VARIANTS = {
    "eager": {},
    "static": {"talker_cache_implementation": "static", "talker_disable_compile": True},
    "static+layers": {"talker_cache_implementation": "static", "talker_disable_compile": True},
}
DIT_MODES = ("reduce-overhead", "default")


def _torch_compile(fn, **kw):
    import torch
    return torch.compile(fn, **kw)


def _mark_step() -> None:
    """Tells CUDA-graph replays that the previous step's outputs are no longer needed."""
    import torch
    if torch.cuda.is_available():
        torch.compiler.cudagraph_mark_step_begin()


def keep_2d_mask(talker):
    """With a static cache, generate prepares the attention masks itself and hands the model a dict of
    them; the Talker's forward needs the plain 2-D mask for its rotary positions. Give it that back.
    Returns the undo. (The wrapper's keyword parameter must be named kwargs: generate checks it.)"""
    prepare = talker.prepare_inputs_for_generation

    def prepare_inputs_for_generation(input_ids, inputs_embeds=None, **kwargs):
        out = prepare(input_ids, inputs_embeds=inputs_embeds, **kwargs)
        if isinstance(out.get("attention_mask"), dict) and kwargs.get("attention_mask") is not None:
            out["attention_mask"] = kwargs["attention_mask"]
        return out

    talker.prepare_inputs_for_generation = prepare_inputs_for_generation
    return lambda: delattr(talker, "prepare_inputs_for_generation")


def compile_layers(talker, compile_fn=_torch_compile):
    """Compile the Talker's decoder layers (talker.model) for one-code steps, with no attention mask on
    them: a 2-D mask grows by one a step and would recompile every step, and with one reply and no
    padding the decoder builds the same causal mask from the cache position. The prefill runs eager,
    and refuses a padded mask. Returns a function that undoes it."""
    m = talker.model
    eager = m.forward
    compiled = compile_fn(eager, mode="reduce-overhead", dynamic=False)

    def forward(*args, **kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is not None and emb.shape[1] == 1:
            kwargs["attention_mask"] = None
            _mark_step()
            return compiled(*args, **kwargs)
        mask = kwargs.get("attention_mask")
        if mask is not None and not isinstance(mask, dict) and not bool(mask.all()):
            raise ValueError("compile_layers drops the mask on one-code steps: one reply, no padding only")
        return eager(*args, **kwargs)

    m.forward = forward
    return lambda: delattr(m, "forward")


def compile_dit(dit, mode: str, compile_fn=_torch_compile):
    """Compile the DiT's network. Each output is copied at once: the Runge-Kutta solver keeps four
    of them alive, and a CUDA-graph replay would overwrite the last. Returns the undo."""
    eager = dit.forward
    compiled = compile_fn(eager, mode=mode, dynamic=False)

    def forward(*args, **kwargs):
        _mark_step()
        return compiled(*args, **kwargs).clone()

    dit.forward = forward
    return lambda: delattr(dit, "forward")


def _graph_breaks() -> int:
    try:
        from torch._dynamo.utils import counters
        return int(sum(counters["graph_break"].values()))
    except Exception:
        return 0


def _seeded(fn, *args, seed: int = 0, **kwargs):
    import torch
    torch.manual_seed(seed)
    return fn(*args, **kwargs)


def talker_turns(omni, clock: StageClock, clips: dict, sr: int, extra: dict) -> dict:
    """Each clip through R4's call with these extra generate arguments: the Talker's milliseconds,
    its codes (on the CPU, to compare), the Thinker's milliseconds and the reply's length."""
    out = {}
    for clip, audio in clips.items():
        clock.start()
        omni.speak(audio, sr, **extra)
        ms = clock.stop()
        code, _ = clock.t2w_call
        clock.t2w_call = None
        codes = [int(c) for c in np.asarray(code.detach().cpu() if hasattr(code, "detach") else code).reshape(-1)]
        out[clip] = {"talker_ms": ms.get("talker"), "thinker_ms": ms.get("thinker"), "codes": codes}
    return out


def talker_variant(omni, clock, clips, sr, name: str, extra: dict, eager: dict | None) -> dict:
    """One Talker variant: warm-up turns (unless eager), then every clip; compared with eager's codes."""
    rec = {"name": name, "extra": {k: v for k, v in extra.items()}, "warmup_ms": [], "turns": {}, "error": None}
    breaks0 = _graph_breaks()
    try:
        first = next(iter(clips.values()))
        if eager is not None:
            for _ in range(WARMUPS):
                t0 = time.perf_counter()
                omni.speak(first, sr, **extra)
                rec["warmup_ms"].append(round((time.perf_counter() - t0) * 1000, 1))
        turns = talker_turns(omni, clock, clips, sr, extra)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        clock.open, clock.t2w_call = False, None
        return rec
    for clip, t in turns.items():
        n = len(t["codes"])
        row = {"talker_ms": t["talker_ms"], "thinker_ms": t["thinker_ms"], "n_codes": n,
               "ms_per_code": round(t["talker_ms"] / n, 2) if n else None}
        if eager is not None:
            e = eager[clip]["codes"]
            same = next((i for i, (a, b) in enumerate(zip(e, t["codes"])) if a != b), None)
            row["identical"] = same is None and len(e) == n
            row["first_difference"] = None if row["identical"] else (same if same is not None else min(len(e), n))
        rec["turns"][clip] = row
        if eager is None:
            row["codes"] = t["codes"]                    # kept for the DiT's blocks and the comparison
    rec["graph_breaks"] = _graph_breaks() - breaks0
    return rec


def dit_blocks(omni, clock, code_lists: dict, compiled: bool, blocks=BLOCKS) -> dict:
    """token2wav on the first b codes of each reply, seeded the same: {b: {clip: (ms, wav as numpy)}}.
    Compiled runs warm up first on the first reply, per block size."""
    import torch
    t2w = omni.model.token2wav
    sp = omni.model.speaker_map[omni.speaker]                 # as generate builds them, for one reply
    dev = torch.device(omni.device)
    kw = {"conditioning": sp["cond"].to(dev).float().expand(1, -1),
          "reference_mel": sp["ref_mel"].to(dev).float().expand(1, -1, -1)}
    out: dict = {}
    with torch.inference_mode():
        for b in blocks:
            out[b] = {}
            have = {c: codes for c, codes in code_lists.items() if len(codes) >= b}
            if compiled and have:
                warm = torch.tensor([next(iter(have.values()))[:b]], device=dev)
                for _ in range(WARMUPS):
                    _seeded(t2w, warm, **kw)
            for clip, codes in have.items():
                x = torch.tensor([codes[:b]], device=dev)
                ms, wav = clock.time_call(_seeded, t2w, x, **kw)
                out[b][clip] = (ms, wav.reshape(-1).float().cpu().numpy())
    return out


def dit_compare(omni, clock, code_lists: dict, blocks=BLOCKS, compile_fn=_torch_compile) -> dict:
    """Eager blocks, then the DiT compiled (each mode in turn until one runs): ms and the difference."""
    eager = dit_blocks(omni, clock, code_lists, compiled=False, blocks=blocks)
    rec = {"mode": None, "errors": [], "blocks": {}}
    dit = omni.model.token2wav.code2wav_dit_model
    fast = None
    for mode in DIT_MODES:
        undo = compile_dit(dit, mode, compile_fn)
        try:
            fast = dit_blocks(omni, clock, code_lists, compiled=True, blocks=blocks)
            rec["mode"] = mode
            break
        except Exception as e:
            rec["errors"].append(f"{mode}: {type(e).__name__}: {e}"[:500])
            undo()
    for b in blocks:
        rows = {}
        for clip, (ms, wav) in eager[b].items():
            row = {"eager_ms": round(ms, 1)}
            if fast is not None and clip in fast[b]:
                fms, fwav = fast[b][clip]
                row["compiled_ms"] = round(fms, 1)
                row["max_abs_diff"] = (round(float(np.max(np.abs(fwav - wav))), 6) if len(fwav) == len(wav)
                                       else "length differs")
            rows[clip] = row
        rec["blocks"][str(b)] = rows
    return rec


def run(omni, clips: dict, sr: int, blocks=BLOCKS, compile_fn=_torch_compile) -> dict:
    """The whole probe on every EVERY-th clip: load, R4's throwaway turn, the three Talker variants,
    then the DiT eager and compiled on eager's codes."""
    chosen = dict(list(clips.items())[::EVERY])
    t0 = time.perf_counter()
    omni.load()
    load_ms = (time.perf_counter() - t0) * 1000
    clock = StageClock(omni.model)
    omni.speak(np.zeros(sr, dtype=np.float32), sr)            # R4's throwaway turn
    variants, eager = {}, None
    for name, extra in TALKER_VARIANTS.items():
        undo = []
        if name != "eager":
            undo.append(keep_2d_mask(omni.model.talker))
        if name == "static+layers":
            undo.append(compile_layers(omni.model.talker, compile_fn))
        try:
            variants[name] = talker_variant(omni, clock, chosen, sr, name, extra, eager)
        finally:
            for u in reversed(undo):
                u()
        if name == "eager":
            if variants[name]["error"]:                     # R4 itself did not run: nothing to compare with
                return {"load_ms": round(load_ms, 1), "clips": list(chosen), "blocks": list(blocks),
                        "talker": variants, "dit": {"mode": None, "errors": [], "blocks": {}}}
            eager = {c: {"codes": t.pop("codes")} for c, t in variants[name]["turns"].items()}
    dit = dit_compare(omni, clock, {c: e["codes"] for c, e in eager.items()}, blocks, compile_fn)
    return {"load_ms": round(load_ms, 1), "clips": list(chosen), "blocks": list(blocks),
            "talker": variants, "dit": dit}


def _median(values) -> float | None:
    v = [x for x in values if isinstance(x, (int, float))]
    return round(float(np.median(v)), 3) if v else None


def _p90(values) -> float | None:
    v = [x for x in values if isinstance(x, (int, float))]
    return round(float(np.percentile(v, 90)), 3) if v else None


def summary(blob: dict) -> dict:
    """Per Talker variant and per block size, the numbers the hand-over reads; then the estimates
    with the fastest Talker variant that ran and the compiled DiT (or eager's, when it did not)."""
    codes_per_s = 50.0                                        # measured in step 1: 50 at p50 and p90
    talker = {}
    for name, v in blob["talker"].items():
        rows = list(v["turns"].values())
        mpc = [r["ms_per_code"] for r in rows]
        talker[name] = {"turns": len(rows), "error": v["error"], "warmup_s": [round(w / 1000, 1) for w in v["warmup_ms"]],
                        "ms_per_code_p50": _median(mpc), "ms_per_code_p90": _p90(mpc),
                        "rtf_p50": None if _median(mpc) is None else round(_median(mpc) * codes_per_s / 1000, 3),
                        "identical": (sum(bool(r.get("identical")) for r in rows) if name != "eager" else None),
                        "graph_breaks": v.get("graph_breaks")}
    dit = {}
    for b, rows in blob["dit"]["blocks"].items():
        r = list(rows.values())
        diffs = [x.get("max_abs_diff") for x in r]
        dit[b] = {"n": len(r), "eager_ms_p50": _median([x["eager_ms"] for x in r]),
                  "compiled_ms_p50": _median([x.get("compiled_ms") for x in r]),
                  "max_abs_diff": max((d for d in diffs if isinstance(d, (int, float))), default=None)}
    ran = {k: t for k, t in talker.items() if t["ms_per_code_p50"] is not None}
    best = min(ran, key=lambda k: ran[k]["ms_per_code_p50"]) if ran else None
    thinker = _median([r["thinker_ms"] for r in blob["talker"]["eager"]["turns"].values()])
    est = {}
    if best and thinker is not None:
        mpc = ran[best]["ms_per_code_p50"]
        for b, d in dit.items():
            block_ms = d["compiled_ms_p50"] if d["compiled_ms_p50"] is not None else d["eager_ms_p50"]
            if block_ms is None:
                continue
            block_audio_ms = int(b) / codes_per_s * 1000
            est[b] = {"first_audio_ms": round(thinker + int(b) * mpc + block_ms, 1),
                      "stream_rtf": round(mpc * codes_per_s / 1000 + block_ms / block_audio_ms, 3)}
    return {"talker": talker, "dit": dit, "dit_mode": blob["dit"]["mode"], "dit_errors": blob["dit"]["errors"],
            "best_talker": best, "thinker_ms_p50": thinker, "estimates": est}


def format_speed(blob: dict) -> str:
    s = summary(blob)
    n = len(blob["clips"])
    f = lambda x: "-" if x is None else f"{x:,.1f}" if isinstance(x, float) else str(x)
    lines = [f"R4's Talker and DiT, eager and compiled, on {n} probe clips (load {blob['load_ms'] / 1000:.1f} s)", "",
             "| Talker | turns | ms per code p50 | p90 | real-time factor | warm-up s | codes identical to eager | graph breaks |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for name, t in s["talker"].items():
        ident = "-" if t["identical"] is None else f"{t['identical']} of {t['turns']}"
        lines.append(f"| {name} | {t['turns']} | {f(t['ms_per_code_p50'])} | {f(t['ms_per_code_p90'])} | "
                     f"{f(t['rtf_p50'])} | {', '.join(map(str, t['warmup_s'])) or '-'} | {ident} | {f(t['graph_breaks'])} |")
    for name, t in s["talker"].items():
        if t["error"]:
            lines.append(f"{name} failed: {t['error']}")
    lines += ["", f"| DiT block | replies | eager ms p50 | compiled ms p50 ({s['dit_mode'] or 'none ran'}) | largest sample difference |",
              "|---:|---:|---:|---:|---:|"]
    for b, d in s["dit"].items():
        diff = "-" if d["max_abs_diff"] is None else f"{d['max_abs_diff']:.4f}"
        lines.append(f"| {b} codes | {d['n']} | {f(d['eager_ms_p50'])} | {f(d['compiled_ms_p50'])} | {diff} |")
    lines += [f"DiT {e}" for e in s["dit_errors"]]
    lines += ["", f"estimates with the fastest Talker ({s['best_talker']}) and the DiT as above; the Thinker {f(s['thinker_ms_p50'])} ms:"]
    for b, e in s["estimates"].items():
        lines.append(f"  block of {b} codes: first audio about {e['first_audio_ms']:,.0f} ms, stream real-time factor {e['stream_rtf']}")
    lines.append("arithmetic on these times, not a streaming rung; below 1 a stream keeps up, before a block's neighbours are counted")
    return "\n".join(lines)
