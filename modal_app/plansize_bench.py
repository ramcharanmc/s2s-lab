"""
The cascade at the plans' sizes on Modal (M45; v1 §4, P1 week 1; v2 §3): R1 with large-v3-turbo (int8) and
Qwen2.5-7B-Instruct, beside R1 as built, in one A10 container, so the two differ only in their models.

    modal run modal_app/plansize_bench.py::dev        30 probe clips, both rungs: loads, answers, a stage table each
    modal run modal_app/plansize_bench.py::evalset    the 200 eval clips, both rungs, once; saved as evidence

What happens where: the image is built once on Modal's CPUs with every model's weights (Qwen2.5-7B's 15 GB
included), so no GPU second is spent downloading; the A10 loads and runs the turns; the laptop saves the JSON,
scores it and prints it. The plans put the recognizer on an L4 and the LLM on an A10; here one card holds both,
said wherever the numbers appear.

Cost guardrails (v1 §5.1): explicit timeouts (dev 20 min, the eval set 40, about $0.73 at most at the A10's
rate); no container kept warm; scale down 60 s after the last call; HF_HUB_OFFLINE=1 at run time. The eval-set run
files are written once: runs/r1-gpu-plan/evalset-modal-a10.json and runs/r1-gpu/evalset-modal-a10-plan.json.
"""

import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
A10_DOLLARS_PER_HOUR = 1.10     # Modal's A10 rate, checked 2026-08-19 (v2 §5)
RUNGS = ["r1-gpu-plan", "r1-gpu"]                                  # the plans' sizes first, R1 as built beside it
MODELS = {
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": None,       # the plans' recognizer
    "Qwen/Qwen2.5-7B-Instruct": None,                           # the plans' LLM size, fp16 (docs/m45-log.md: not AWQ)
    "Systran/faster-whisper-base.en": None,                     # R1 as built, beside it
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
}


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))      # transformers 5.17.0, as r0_bench
    .pip_install("accelerate==1.15.0", "numpy==2.2.6", "scipy==1.15.3", "soundfile==0.14.0",
                 "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    .apt_install("gcc")                     # torch 2.14's Triton kernels compile a C helper on first use (r0_bench)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-plansize", image=image)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _free_gpu() -> None:
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


@app.function(gpu="A10G", timeout=20 * 60, scaledown_window=60)
def bench(rung_ids: list[str]) -> dict:
    """Each rung in turn over the 30 probe clips: its stage table, and every reply, to read."""
    import os
    os.chdir("/root")
    from s2slab.probes import TARGET_SR
    from s2slab.rungs import get, probe_clips, run_turns, summarize_turns
    from s2slab.timing import format_table
    t0 = time.perf_counter()
    tables, replies = {}, {}
    for rung_id in rung_ids:
        rung = get(rung_id)
        load_ms = rung.load()
        turns = run_turns(rung, probe_clips(), TARGET_SR)
        tables[rung_id] = (format_table(summarize_turns(turns), title=f"{rung_id}: {rung.description}")
                           + f"\n\nload: {load_ms:.0f} ms, once, before the first turn")
        replies[rung_id] = {clip: t.reply_text for clip, t in turns.items()}
        del rung, turns
        _free_gpu()
    return {"gpu": _gpu_name(), "tables": tables, "replies": replies, "seconds": time.perf_counter() - t0}


@app.function(gpu="A10G", timeout=40 * 60, scaledown_window=60)
def evaluate(rung_ids: list[str], commit: str, limit: int = 0) -> dict:
    """Each rung in turn over the frozen eval set, one container (r3_probe.evaluate's shape)."""
    import os
    os.chdir("/root")
    from s2slab.evalrun import run
    from s2slab.evalset import verify
    problems = verify()
    if problems:
        raise RuntimeError("the eval set on Modal is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    t0 = time.perf_counter()
    machine = f"Modal {_gpu_name()}, one container: {' then '.join(rung_ids)}"
    results = {}
    for rung_id in rung_ids:
        results[rung_id] = run(rung_id, limit=limit or None, machine=machine)
        results[rung_id]["commit"] = commit
        _free_gpu()
    return {"results": results, "seconds": time.perf_counter() - t0}


def results_path(rung_id: str, limit: int = 0) -> pathlib.Path:
    """The plans' rung under its own name; R1 beside it tagged, since R1's A10 run of M19 holds the plain name."""
    name = {"r1-gpu-plan": "evalset-modal-a10", "r1-gpu": "evalset-modal-a10-plan"}[rung_id]
    if limit:
        return REPO / "results" / rung_id / f"{name}-limit{limit}.json"
    return REPO / "runs" / rung_id / f"{name}.json"


def _commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                          capture_output=True, text=True).stdout.strip()


@app.local_entrypoint()
def dev():
    """Both rungs on the 30 probe clips: does the plans' rung load and answer? Every reply printed, to read."""
    t0 = time.time()
    out = bench.remote(RUNGS)
    wall = time.time() - t0
    for rung_id in RUNGS:
        print(out["tables"][rung_id], "\n")
        for clip, reply in out["replies"][rung_id].items():
            print(f"  {rung_id} {clip}: {reply}")
        print()
    print(f"commit {_commit()} · GPU {out['gpu']}")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end: about "
          f"${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def evalset(limit: int = 0):
    """Both rungs over the eval set in one A10 container; refused if a run file exists or the tree is not committed."""
    from s2slab.evalrun import report, save
    from s2slab.plansize import compare, format_report
    from s2slab.sidechannel import refuse_to_overwrite

    commit = _commit()
    if not limit:
        if commit.endswith("-dirty"):
            raise SystemExit(f"{commit}: commit first; the eval-set run is evidence")
        for r in RUNGS:
            refuse_to_overwrite(results_path(r))
    t0 = time.time()
    out = evaluate.remote(RUNGS, commit, limit)
    wall = time.time() - t0
    for rung_id, results in out["results"].items():
        path = save(results, results_path(rung_id, limit))
        print(report(results, root=REPO / "evalset"))
        print(f"\nsaved -> {path.relative_to(REPO)}\n")
    if not limit:
        print(format_report(compare(out["results"]["r1-gpu-plan"], out["results"]["r1-gpu"], root=REPO / "evalset")))
    print(f"\n{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")
