"""
M36: what torch.compile compiled, kept between a container's wakes.

M35 found the demo's L40S waking in 186 s, most of it compiling: the DiT's windows 65 s, the sized cache's
decoder 57 s, and most of R4's 61 s the Talker's layers; its weights load in about 4 s. Each Modal
container starts empty, so every wake compiled everything again. PyTorch can save what a process compiled
(torch.compiler.save_cache_artifacts: inductor's graphs, AOTAutograd's, Triton's kernels, the autotuning
results) as one blob, and load it into a fresh process (load_cache_artifacts) before anything compiles, so
that the compiles which follow find their results instead of redoing them. Dynamo still traces the Python,
and CUDA graphs are still captured, at every wake.

    load(path)   before the load: the blob from the volume, if one is there
    save(path)   after the load: the blob of what this process compiled or loaded, written when there is
                 no file, when the file failed to load, or when the blob is larger than the file (new graphs
                 were compiled), so a later change of code adds to the cache rather than being lost

PyTorch keys each entry inside the blob by its version and the graph's source, so a library or code change
compiles afresh rather than reusing something wrong. A file that fails to load is reported, and the wake
compiles as it did before M36.
"""

from __future__ import annotations

import pathlib
import time


def load(path) -> dict:
    """The blob at `path` into this process's compile caches. Returns what happened, for the log."""
    p = pathlib.Path(path)
    if not p.exists():
        return {"loaded": False, "bytes": 0}
    import torch
    data = p.read_bytes()
    t0 = time.perf_counter()
    try:
        torch.compiler.load_cache_artifacts(data)
    except Exception as e:                                        # a stale or broken file: compile as before
        return {"loaded": False, "bytes": len(data), "error": f"{type(e).__name__}: {e}"[:300]}
    return {"loaded": True, "bytes": len(data), "s": round(time.perf_counter() - t0, 1)}


def save(path, loaded: dict | None = None) -> dict:
    """What this process compiled or loaded, written to `path` when it adds to what is there (see the
    docstring). `loaded` is load()'s result. Returns what happened, for the log."""
    import torch
    out = torch.compiler.save_cache_artifacts()
    if not out or not out[0]:
        return {"saved": False, "bytes": 0}
    data = out[0]
    p = pathlib.Path(path)
    had = p.stat().st_size if p.exists() else 0
    if had and (loaded or {}).get("loaded") and len(data) <= had:
        return {"saved": False, "bytes": len(data), "kept": had}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)                                                # never half a file where the next wake reads
    return {"saved": True, "bytes": len(data), "was": had}
