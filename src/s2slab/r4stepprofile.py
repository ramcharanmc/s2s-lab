"""
M33: where a Talker step's 10 ms goes, and how much of it is the 6144 places of its cache.

M32b found R4's Talker at about 10 ms of GPU work a code, even with the whole step captured as one CUDA
graph (runs/r4-gpu/loop-speed-l40s.json). Its static cache holds 6144 places (r4stream.MAX_CACHE, room for
a 60 s turn and 4096 codes) and every step attends over all of them, masked, while a turn fills far fewer.
Exploratory: no registration; the 10 probe clips M31 and M32b used; nothing of the eval set.

    size        the Talker's configuration, its parameters, the bytes a code reads (its weights, less the
                codec embedding table, of which a step reads one row) and the time an L40S needs to read
                them once (L40S_GBPS); the cache's keys and values at each size, and their read time
    timing      our loop (s2slab.talkerloop) with its step captured as one CUDA graph, the decoder compiled
                by inductor inside it, on a cache of each size in CACHE_SIZES: ms a code, whether the codes
                are the 6144-place cache's, and how many places each turn filled (its prompt and its codes)
    breakdown   one clip's Talker turn, the loop run eagerly (the decoder's compile_layers wrapper set aside,
                so each kernel is seen), under torch.profiler, at 6144 and 2048 places: GPU time a code by
                kind of kernel (attention, matrix products, sampling, the rest), and the ten longest kernels

A cache smaller than a turn needs would cut the reply: each turn's codes are capped at the room its cache
leaves (the places less the prompt), and a turn that reaches that cap is counted, never hidden.

WHAT COUNTS AS WORTH A FOLLOW-UP, written in the design note before this run (docs/m32-log.md): the 2048-
place cache at 8 ms a code or less (p50), no turn cut. Then a design note for a right-sized cache in R4
streamed, and a registration; if not, the latency work on R4 ends here, the breakdown its reason.

    modal run modal_app/r4_bench.py::stepprofile      runs/r4-gpu/step-profile-l40s.json
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .r4speed import _median, _p90, _torch_compile

STEP_PROFILE = pathlib.Path("runs/r4-gpu/step-profile-l40s.json")
EVERY = 3                          # every third probe clip: M31's and M32b's 10
CACHE_SIZES = (6144, 2048, 1024)   # today's, and two right-sized candidates
PROFILED = (6144, 2048)            # the sizes whose kernels are broken down
FOLLOW_UP_MS = 8.0                 # the design note's line, ms a code at p50, 2048 places
L40S_GBPS = 864.0                  # the L40S's memory bandwidth, GB/s (NVIDIA's data sheet)
WARMUPS = 2                        # turns before timing a cache size: the compile, the capture
KINDS = {"attention": ("attention", "fmha", "flash", "sdpa", "scaled_dot_product"),
         "matmul": ("gemm", "gemv", "cublas", "xmma", "splitk", "aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"),
         "sampling": ("sort", "topk", "radix", "cumsum", "scan", "exponential", "argmax", "multinomial")}


def kind(name: str) -> str:
    n = name.lower()
    return next((k for k, words in KINDS.items() if any(w in n for w in words)), "other")


def talker_size(talker, sizes=CACHE_SIZES) -> dict:
    """The Talker's shape and what a code must read, with the read times an L40S sets as a floor."""
    cfg = talker.config
    params = sum(p.numel() for p in talker.parameters())
    emb = talker.get_input_embeddings().weight
    width = next(talker.parameters()).element_size()
    read = (params - emb.numel()) * width                 # a step reads one row of the codec embedding table
    kv = lambda n: cfg.num_hidden_layers * 2 * n * cfg.num_key_value_heads * cfg.head_dim * width   # noqa: E731
    floor = lambda b: round(b / (L40S_GBPS * 1e9) * 1000, 3)                                        # noqa: E731
    return {"layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "intermediate": cfg.intermediate_size,
            "heads": cfg.num_attention_heads, "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
            "vocab": cfg.vocab_size, "embedding": cfg.embedding_size, "params": int(params),
            "bytes_a_code": int(read), "floor_ms": floor(read),
            "kv_bytes": {str(n): int(kv(n)) for n in sizes}, "kv_floor_ms": {str(n): floor(kv(n)) for n in sizes}}


def _capped(loop, size: int, log: list):
    """The loop's generate with each turn's codes capped at the room its cache leaves; each turn logged."""
    inner = loop.generate

    def generate(*args, **kw):
        prompt = int(kw["inputs_embeds"].shape[1])
        room = size - prompt
        if room < 1:
            raise ValueError(f"a prompt of {prompt} places does not fit a cache of {size}")
        asked = int(kw.get("max_new_tokens", 4096))
        kw["max_new_tokens"] = min(asked, room)
        out = inner(*args, **kw)
        log.append({"prompt": prompt, "codes": loop.last_codes, "cut": loop.last_codes >= room and room < asked})
        return out

    loop.generate = generate


def _cache(talker, n):
    from transformers import StaticCache
    return StaticCache(config=talker.config, max_cache_len=n)


def _places(cache) -> int:
    """A static cache's places (transformers 5.17 names it get_max_length; max_cache_len is going)."""
    return int(cache.get_max_length() if hasattr(cache, "get_max_length") else cache.max_cache_len)


def breakdown(omni, talker, cache, size: int, audio, sr: int) -> dict:
    """One Talker turn, the loop eager, under torch.profiler: GPU time a code by kind of kernel (on a machine
    without CUDA, CPU time by operator: the tests' case)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from .talkerloop import TalkerLoop
    cuda = torch.cuda.is_available()
    m = talker.model
    wrapped = vars(m).pop("forward", None)               # compile_layers' wrapper set aside: each kernel seen
    loop = TalkerLoop(talker, cache)
    box: dict = {}
    inner = loop.generate

    def generate(*args, **kw):
        kw["max_new_tokens"] = min(int(kw.get("max_new_tokens", 4096)), size - int(kw["inputs_embeds"].shape[1]))
        with profile(activities=[ProfilerActivity.CUDA] if cuda else [ProfilerActivity.CPU]) as prof:
            out = inner(*args, **kw)
            if cuda:
                torch.cuda.synchronize()
        box["prof"], box["codes"] = prof, loop.last_codes
        return out

    loop.generate = generate
    undo = loop.install()
    home = omni.cache
    try:
        omni.cache = cache
        omni.speak(audio, sr)
    finally:
        undo()
        omni.cache = home
        if wrapped is not None:
            m.forward = wrapped
    rows = {}
    for e in box["prof"].key_averages():
        if cuda and str(getattr(e, "device_type", "")).split(".")[-1] != "CUDA":
            continue
        us = (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)) if cuda \
            else getattr(e, "self_cpu_time_total", 0)
        if us:
            rows[e.key] = rows.get(e.key, 0.0) + float(us)
    n = max(1, box["codes"])
    by_kind: dict = {}
    for name, us in rows.items():
        by_kind[kind(name)] = by_kind.get(kind(name), 0.0) + us
    top = sorted(rows.items(), key=lambda kv: -kv[1])[:10]
    return {"size": size, "codes": box["codes"], "device": "cuda" if cuda else "cpu",
            "ms_a_code": {k: round(v / 1000 / n, 3) for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])},
            "total_ms_a_code": round(sum(rows.values()) / 1000 / n, 3),
            "top": [{"name": k[:120], "kind": kind(k), "ms_a_code": round(v / 1000 / n, 3)} for k, v in top]}


def run(clips: dict, sr: int, omni=None, compile_fn=_torch_compile, sizes=CACHE_SIZES, profiled=PROFILED) -> dict:
    """Load; the Talker's size; the captured loop on each cache size over the chosen clips; then the breakdown."""
    from .r4profile import StageClock
    from .r4speed import talker_variant
    from .r4stream import FastOmni
    from .talkerloop import TalkerLoop
    omni = omni or FastOmni(compile_fn=compile_fn)
    chosen = dict(list(clips.items())[::EVERY])
    first = next(iter(chosen.values()))
    t0 = time.perf_counter()
    omni.load()
    load_ms = (time.perf_counter() - t0) * 1000
    talker, home = omni.model.talker, omni.cache
    m = talker.model
    layers = compile_fn(lambda **kw: type(m).forward(m, **kw), mode="default", dynamic=False)
    caches = {n: home if n == _places(home) else _cache(talker, n) for n in sizes}
    loops, logs, undo = {}, {}, []
    for n in sizes:
        loops[n], logs[n] = TalkerLoop(talker, caches[n], graph=True, layers=layers), []
        loops[n].on = False
        _capped(loops[n], n, logs[n])
        undo.append(loops[n].install())
    out = {"load_ms": round(load_ms, 1), "clips": list(chosen), "sizes": list(sizes), "follow_up_ms": FOLLOW_UP_MS,
           "talker_size": talker_size(talker, sizes), "timing": {}, "turns_filled": {}, "breakdown": {}}
    try:
        for n in sizes:                                    # one turn on each cache, the loop eager and capped: its tensors
            init = TalkerLoop(talker, caches[n])           # made before any capture reads them (TalkerLoop._capture)
            _capped(init, n, [])
            u = init.install()
            try:
                omni.cache = caches[n]
                omni.speak(first, sr)
            finally:
                u()
        clock = StageClock(omni.model)
        reference = None
        for n in sizes:
            for k, loop in loops.items():
                loop.on = k == n
            omni.cache = caches[n]
            warm = []
            if reference is None:                            # talker_variant warms only a compared variant
                for _ in range(WARMUPS):
                    t1 = time.perf_counter()
                    omni.speak(first, sr)
                    warm.append(round((time.perf_counter() - t1) * 1000, 1))
            del logs[n][:]
            rec = talker_variant(omni, clock, chosen, sr, f"graph@{n}", {}, reference)
            rec["warmup_ms"] = warm + rec["warmup_ms"]
            if reference is None and not rec["error"]:
                reference = {c: {"codes": t.pop("codes")} for c, t in rec["turns"].items()}
            out["timing"][str(n)] = rec
            out["turns_filled"][str(n)] = list(logs[n][-len(chosen):])
    finally:
        for u in reversed(undo):
            u()
        omni.cache = home
    for n in profiled:
        try:
            out["breakdown"][str(n)] = breakdown(omni, talker, caches[n], n, first, sr)
        except Exception as e:                             # a failed profile is recorded; the timings stand
            out["breakdown"][str(n)] = {"error": f"{type(e).__name__}: {e}"[:500]}
    return out


def summary(blob: dict) -> dict:
    sizes = {}
    for n, rec in blob["timing"].items():
        rows = list(rec["turns"].values())
        filled = blob["turns_filled"].get(n, [])
        used = [t["prompt"] + t["codes"] for t in filled]
        sizes[n] = {"turns": len(rows), "error": rec["error"], "warmup_s": [round(w / 1000, 1) for w in rec["warmup_ms"]],
                    "ms_per_code_p50": _median([r["ms_per_code"] for r in rows]),
                    "ms_per_code_p90": _p90([r["ms_per_code"] for r in rows]),
                    "identical": None if n == str(blob["sizes"][0]) else sum(bool(r.get("identical")) for r in rows),
                    "used_p50": _median(used), "used_max": max(used) if used else None,
                    "cut": sum(bool(t["cut"]) for t in filled)}
    s2048 = sizes.get("2048")
    readable = s2048 is not None and s2048["ms_per_code_p50"] is not None and s2048["cut"] == 0
    follow = readable and s2048["ms_per_code_p50"] <= blob["follow_up_ms"]
    if not readable:
        verdict = "the 2048-place cache cannot be read (it failed, or cut a turn): no follow-up on this run"
    elif follow:
        verdict = (f"2048 places at {s2048['ms_per_code_p50']} ms a code, at or under {blob['follow_up_ms']:g}: "
                   "a design note for a right-sized cache in R4 streamed follows")
    else:
        verdict = (f"2048 places at {s2048['ms_per_code_p50']} ms a code, over {blob['follow_up_ms']:g}: "
                   "the latency work on R4 ends here")
    return {"sizes": sizes, "follow_up": follow, "verdict": verdict}


def format_step_profile(blob: dict) -> str:
    s, z = summary(blob), blob["talker_size"]
    f = lambda x: "-" if x is None else f"{x:,.2f}" if isinstance(x, float) else str(x)        # noqa: E731
    lines = [f"R4's Talker step, on {len(blob['clips'])} probe clips (load {blob['load_ms'] / 1000:.1f} s)", "",
             f"The Talker: {z['layers']} layers, hidden {z['hidden']}, intermediate {z['intermediate']}, heads {z['heads']} "
             f"({z['kv_heads']} for keys and values, {z['head_dim']} wide), {z['params'] / 1e6:,.0f} M parameters; a code "
             f"reads {z['bytes_a_code'] / 1e9:.2f} GB of weights: {z['floor_ms']} ms on an L40S at best.",
             "The cache's keys and values: " + ", ".join(f"{n} places {z['kv_bytes'][n] / 1e6:,.0f} MB ({z['kv_floor_ms'][n]} ms)"
                                                         for n in z["kv_bytes"]) + ".", "",
             f"| cache places | turns | ms per code p50 | p90 | codes identical to {blob['sizes'][0]}'s | places filled p50 | max "
             "| turns cut | warm-up s |",
             "|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for n, t in s["sizes"].items():
        ident = "(the reference)" if t["identical"] is None else f"{t['identical']} of {t['turns']}"
        lines.append(f"| {n} | {t['turns']} | {f(t['ms_per_code_p50'])} | {f(t['ms_per_code_p90'])} | {ident} | "
                     f"{f(t['used_p50'])} | {f(t['used_max'])} | {t['cut']} | {', '.join(map(str, t['warmup_s'])) or '-'} |")
    for n, t in s["sizes"].items():
        if t["error"]:
            lines.append(f"{n} places failed: {t['error']}")
    for n, b in blob["breakdown"].items():
        lines.append("")
        if "error" in b:
            lines.append(f"Breakdown at {n} places failed: {b['error']}")
            continue
        lines.append(f"Breakdown at {n} places, one turn of {b['codes']} codes, the loop eager ({b['device']} time a code): "
                     + ", ".join(f"{k} {v} ms" for k, v in b["ms_a_code"].items()) + f"; in all {b['total_ms_a_code']} ms.")
        lines += [f"  {r['ms_a_code']:>7} ms  {r['kind']:<9} {r['name']}" for r in b["top"]]
    lines += ["", "Follow-up: " + s["verdict"] + "."]
    return "\n".join(lines)
