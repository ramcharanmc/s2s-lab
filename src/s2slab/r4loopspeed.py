"""
M32b, step 2: does our Talker loop write R4's codes faster than generate, and the same codes?

On every third probe clip (10 of 30, as M31's speed probe), through R4's own call and seed on the fast
route (r4stream.FastOmni: the Talker's static cache, its 2-D mask kept, its layers compiled for one-code
steps), the Talker three ways on one set of weights:

    generate      talker.generate, as r4-gpu-fast and r4-gpu-stream run it today: the reference
    loop          our loop (s2slab.talkerloop), the decoder as FastOmni installs it
    loop+graph    our loop with each code's step captured as one CUDA graph; inside it the decoder
                  compiled by inductor without graphs of its own (mode "default"), as compile_layers
                  compiles it but for the graphs

For each: the Talker's milliseconds a code (its prefill included, as M31 measured it), the warm-up turns'
seconds (the compiles and the capture), and whether its codes are generate's on each clip, and if not,
from which code on. Nothing here changes a rung.

THE STOP RULE, written in M32b's design note before this run (docs/m32-log.md): if neither loop writes a
code in under STOP_MS milliseconds at p50, M32b stops here and is reported as such.

    modal run modal_app/r4_bench.py::loopspeed      runs/r4-gpu/loop-speed-l40s.json
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .r4speed import _median, _p90, _torch_compile

LOOP_SPEED = pathlib.Path("runs/r4-gpu/loop-speed-l40s.json")
EVERY = 3                          # every third probe clip: the same 10 as M31's speed probe
STOP_MS = 10.0                     # the design note's stop rule, ms a code at p50
VARIANTS = ("generate", "loop", "loop+graph")


def loops_for(omni, compile_fn=_torch_compile) -> tuple:
    """The two loops on omni's Talker and cache, installed (generate stays reachable: each has `on`).
    Returns (loops by name, the undo)."""
    from .talkerloop import TalkerLoop
    talker = omni.model.talker
    m = talker.model
    eager_layers = lambda **kw: type(m).forward(m, **kw)                      # noqa: E731
    loops = {"loop": TalkerLoop(talker, omni.cache),
             "loop+graph": TalkerLoop(talker, omni.cache, graph=True,
                                      layers=compile_fn(eager_layers, mode="default", dynamic=False))}
    undo = [loop.install() for loop in loops.values()]
    return loops, lambda: [u() for u in reversed(undo)]


def run(clips: dict, sr: int, omni=None, compile_fn=_torch_compile) -> dict:
    """Load, R4's throwaway turn through generate, then each variant on the chosen clips."""
    from .r4profile import StageClock
    from .r4speed import talker_variant
    from .r4stream import FastOmni
    omni = omni or FastOmni(compile_fn=compile_fn)
    chosen = dict(list(clips.items())[::EVERY])
    t0 = time.perf_counter()
    omni.load()
    load_ms = (time.perf_counter() - t0) * 1000
    loops, undo = loops_for(omni, compile_fn)
    try:
        clock = StageClock(omni.model)                    # it wraps the switch: each variant timed the same way

        def use(name):
            for n, loop in loops.items():
                loop.on = n == name

        use("generate")
        omni.speak(np.zeros(sr, dtype=np.float32), sr)    # R4's throwaway turn: the decoder's compile
        variants, reference = {}, None
        for name in VARIANTS:
            use(name)
            variants[name] = talker_variant(omni, clock, chosen, sr, name, {}, reference)
            if name == "generate":
                if variants[name]["error"]:               # nothing to compare with
                    break
                reference = {c: {"codes": t.pop("codes")} for c, t in variants[name]["turns"].items()}
    finally:
        undo()
    return {"load_ms": round(load_ms, 1), "clips": list(chosen), "stop_ms": STOP_MS, "talker": variants}


def summary(blob: dict) -> dict:
    """Per variant the numbers the hand-over reads, then the stop rule's verdict on the faster loop."""
    out = {}
    for name, v in blob["talker"].items():
        rows = list(v["turns"].values())
        mpc = [r["ms_per_code"] for r in rows]
        out[name] = {"turns": len(rows), "error": v["error"], "warmup_s": [round(w / 1000, 1) for w in v["warmup_ms"]],
                     "ms_per_code_p50": _median(mpc), "ms_per_code_p90": _p90(mpc),
                     "codes_p50": _median([r["n_codes"] for r in rows]),
                     "identical": None if name == "generate" else sum(bool(r.get("identical")) for r in rows),
                     "first_differences": None if name == "generate" else
                     [r.get("first_difference") for r in rows if not r.get("identical")]}
    ran = {k: t for k, t in out.items() if k != "generate" and t["ms_per_code_p50"] is not None}
    best = min(ran, key=lambda k: ran[k]["ms_per_code_p50"]) if ran else None
    go = best is not None and ran[best]["ms_per_code_p50"] < blob["stop_ms"]
    return {"variants": out, "best_loop": best, "go": go,
            "verdict": (f"{best} writes a code in {ran[best]['ms_per_code_p50']} ms at p50, under {blob['stop_ms']:g}: "
                        "M32b goes on to step 3" if go else
                        f"no loop under {blob['stop_ms']:g} ms a code at p50: by the stop rule, M32b stops here")}


def format_loop_speed(blob: dict) -> str:
    s = summary(blob)
    f = lambda x: "-" if x is None else f"{x:,.2f}" if isinstance(x, float) else str(x)       # noqa: E731
    lines = [f"R4's Talker, generate and our loop, on {len(blob['clips'])} probe clips (load {blob['load_ms'] / 1000:.1f} s)", "",
             "| Talker | turns | ms per code p50 | p90 | codes p50 | warm-up s | codes identical to generate |",
             "|---|---:|---:|---:|---:|---|---|"]
    for name, t in s["variants"].items():
        ident = "-" if t["identical"] is None else f"{t['identical']} of {t['turns']}" + (
            f" (others part at code {', '.join(map(str, t['first_differences']))})" if t["first_differences"] else "")
        lines.append(f"| {name} | {t['turns']} | {f(t['ms_per_code_p50'])} | {f(t['ms_per_code_p90'])} | "
                     f"{f(t['codes_p50'])} | {', '.join(map(str, t['warmup_s'])) or '-'} | {ident} |")
    for name, t in s["variants"].items():
        if t["error"]:
            lines.append(f"{name} failed: {t['error']}")
    lines += ["", "Stop rule: " + s["verdict"] + "."]
    return "\n".join(lines)
