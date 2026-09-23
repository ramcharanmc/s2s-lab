"""
R4 on Modal: the S2S rung on an L40S, with R3 beside it in the same container.

    modal run modal_app/r4_bench.py --rung r4-gpu                            30 probe clips: the smoke test
    modal run modal_app/r4_bench.py --rung r3-gpu,r4-gpu --evalset --limit 8
    modal run modal_app/r4_bench.py --rung r3-gpu,r4-gpu --evalset           the R4 row, saved to runs/<rung>/evalset-modal-l40s.json

What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON and prints the report (scoring needs no GPU)
  image    CUDA 12.6, Python 3.12, torch, transformers, R0's model libraries
           (Kokoro for R3), and the weights: Qwen2.5-Omni-7B (about 22 GB),
           Qwen2-Audio-7B-Instruct and Kokoro for the R3 comparator -- all
           fetched on CPU at build time
  L40S     48 GB: Omni's ~20 GB of bf16 weights plus the activations of a
           15 s clip do not fit the A10, which is why R4 lives on this card and
           R3 is re-measured here rather than read off its A10 row (M13's rule)

The same guardrails as r0_bench: explicit timeout, no container kept warm,
scale down 60 s after the call, HF_HUB_OFFLINE=1, one container per run, the
models of one rung dropped before the next loads, evidence never overwritten.
"""

import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
L40S_DOLLARS_PER_HOUR = 1.95   # Modal's L40S rate, checked 2026-08-19 (v2 §5)

# Every Hugging Face repo the rungs this file runs will load; tests/test_modal_app.py checks it.
MODELS = {
    "Qwen/Qwen2.5-Omni-7B": None,                               # R4
    "Qwen/Qwen2-Audio-7B-Instruct": None,                       # R3, the comparator
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],   # R3's voice
}


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.17.0", "accelerate==1.15.0", "soundfile==0.14.0", "numpy==2.2.6",
                 "huggingface_hub==1.31.0")
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
    .pip_install("scipy==1.15.3", "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    .apt_install("gcc")                     # torch 2.14's Triton helpers compile a C stub on first use
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r4", image=image)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _free_gpu() -> None:
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


@app.function(gpu="L40S", timeout=60 * 60, scaledown_window=60)
def bench(rung_ids: list[str]) -> dict:
    """Each rung over the 30 probe clips, one container: the stage table from rungs.report()."""
    import os
    os.chdir("/root")
    from s2slab.probes import TARGET_SR
    from s2slab.rungs import probe_clips, report
    t0 = time.perf_counter()
    tables = {}
    for rung_id in rung_ids:
        tables[rung_id] = report(rung_id, probe_clips(), TARGET_SR)
        _free_gpu()
    return {"gpu": _gpu_name(), "tables": tables, "seconds": time.perf_counter() - t0}


@app.function(gpu="L40S", timeout=60 * 60, scaledown_window=60)
def evaluate(rung_ids: list[str], commit: str, limit: int = 0) -> dict:
    """Each rung over the frozen eval set, one container (r0_bench.evaluate, on the L40S)."""
    import os
    os.chdir("/root")
    from s2slab.evalrun import run
    from s2slab.evalset import verify
    problems = verify()
    if problems:
        raise RuntimeError("the eval set on Modal is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    t0 = time.perf_counter()
    machine = f"Modal {_gpu_name()}"
    if len(rung_ids) > 1:
        machine += f", one container: {' then '.join(rung_ids)}"
    results = {}
    for rung_id in rung_ids:
        results[rung_id] = run(rung_id, limit=limit or None, machine=machine)
        results[rung_id]["commit"] = commit
        _free_gpu()
    return {"results": results, "seconds": time.perf_counter() - t0}


def results_path(rung_id: str, limit: int = 0, tag: str = "") -> pathlib.Path:
    """runs/<rung>/evalset-modal-l40s[-tag].json for the real run (committed evidence);
    results/<rung>/...-limitN.json for a smoke test (ignored)."""
    suffix = f"-{tag}" if tag else ""
    if limit:
        return REPO / "results" / rung_id / f"evalset-modal-l40s{suffix}-limit{limit}.json"
    return REPO / "runs" / rung_id / f"evalset-modal-l40s{suffix}.json"


@app.local_entrypoint()
def main(rung: str = "r4-gpu", evalset: bool = False, limit: int = 0, tag: str = ""):
    from s2slab.evalrun import report, save
    from s2slab.sidechannel import refuse_to_overwrite

    rung_ids = [r.strip() for r in rung.split(",") if r.strip()]
    if evalset and not limit:
        for r in rung_ids:
            refuse_to_overwrite(results_path(r))
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = evaluate.remote(rung_ids, commit, limit) if evalset else bench.remote(rung_ids)
    wall = time.time() - t0
    if evalset:
        for rung_id, results in out["results"].items():
            path = save(results, results_path(rung_id, limit, tag))
            print(report(results, root=REPO / "evalset"))
            print(f"\nsaved -> {path.relative_to(REPO)}\n")
    else:
        for table in out["tables"].values():
            print(table, "\n")
        print(f"commit {commit} · GPU {out['gpu']}")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")
