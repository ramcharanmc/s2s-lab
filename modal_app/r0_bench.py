"""
R0 on Modal: the reportable GPU row (v2 plan §5).

Colab gave development numbers on a shared T4. This runs the same rungs and the
same code on a Modal T4 billed per second, so the number can be reported
together with what it cost.

    modal run modal_app/r0_bench.py                             # 30 probe clips, r0-gpu
    modal run modal_app/r0_bench.py --rung r0-gpu-bigger
    modal run modal_app/r0_bench.py --evalset --limit 8         # eval-set smoke test
    modal run modal_app/r0_bench.py --evalset                   # all 200 clips: p90 and WER

What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON and prints the report (scoring needs no GPU)
  image    built once on Modal's CPUs and cached: CUDA 12.6 + cuDNN 9, Python
           3.12, the pinned packages, and every model's weights
  T4       loads the models from the image and runs the turns

The full eval-set run is evidence, so its JSON goes to runs/<rung>/, which is
committed. A --limit smoke test goes to results/, which is not.

Cost guardrails (v1 §5.1): an explicit timeout; no container kept warm
(min_containers stays 0); scale down 60 s after the last call; weights fetched
during the image build on CPU, never on the GPU's meter. HF_HUB_OFFLINE=1 at
run time turns a model missing from MODELS into an error instead of a silent
download billed at GPU rates.
"""

import json
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
T4_DOLLARS_PER_HOUR = 0.59     # Modal's T4 rate, checked 2026-08-19 (v2 §5)

# Every Hugging Face repo the GPU rungs load, and which files (None = all).
# tests/test_modal_app.py checks this against the rungs themselves.
MODELS = {
    "Systran/faster-whisper-base.en": None,
    "Systran/faster-whisper-small.en": None,
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    "Qwen/Qwen2.5-1.5B-Instruct": None,
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
}


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04",
                              add_python="3.12")
    # torch 2.14.0 is built for CUDA 12.6 and 13.0, not 12.8 (checked on the
    # index). 12.6 matches the base image. faster-whisper's engine,
    # CTranslate2, needs CUDA 12 and cuDNN 9, which the base provides.
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "soundfile==0.14.0",
                 "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    # torch 2.14 runs some small GPU ops through Triton kernels, which compile
    # a C helper on first use; the CUDA runtime image has no C compiler. Colab
    # has one, so the same kernels ran there. Added after the weights so every
    # layer above stays cached. (Kill switch instead: TORCH_DISABLE_NATIVE_JIT=1.)
    .apt_install("gcc")
    .env({"HF_HUB_OFFLINE": "1"})
    # Added last and at container start, so editing code never rebuilds the image.
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    # The manifest and the 200 WAVs (44 MB). Uploaded once; Modal re-sends only
    # files whose contents change, and these are frozen.
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r0", image=image)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                           "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.function(gpu="T4", timeout=20 * 60, scaledown_window=60)
def bench(rung_id: str) -> dict:
    """Load one rung and run the 30 probe clips through it, on the T4.

    Takes:   a registered rung id.
    Returns: {"gpu": name and driver, "table": the stage table from
             rungs.report(), "seconds": time spent inside this function}.
    """
    import os
    os.chdir("/root")                       # probes.PROBE_ROOT is relative to here
    from s2slab.probes import TARGET_SR
    from s2slab.rungs import probe_clips, report
    t0 = time.perf_counter()
    table = report(rung_id, probe_clips(), TARGET_SR)
    return {"gpu": _gpu_name(), "table": table, "seconds": time.perf_counter() - t0}


@app.function(gpu="T4", timeout=20 * 60, scaledown_window=60)
def evaluate(rung_id: str, commit: str, limit: int = 0) -> dict:
    """Run one rung over the frozen eval set, on the T4.

    Takes:   a registered rung id; the laptop's commit (the container has no
             git, so the commit that sent the code has to travel with it); an
             optional limit for a smoke test (0 = all 200).
    Calls:   evalset.verify() first -- a WAV that differs from the manifest
             means this is not the frozen set, and the run stops before any
             GPU time is spent on it; then evalrun.run().
    Returns: {"results": the evalrun results dict, "seconds": time inside}.
    """
    import os
    os.chdir("/root")                       # evalset.EVAL_ROOT is relative to here
    from s2slab.evalrun import run
    from s2slab.evalset import verify
    problems = verify()
    if problems:
        raise RuntimeError("the eval set on Modal is not the frozen set:\n  "
                           + "\n  ".join(problems[:10]))
    t0 = time.perf_counter()
    results = run(rung_id, limit=limit or None, machine=f"Modal {_gpu_name()}")
    results["commit"] = commit
    return {"results": results, "seconds": time.perf_counter() - t0}


def results_path(rung_id: str, limit: int = 0) -> pathlib.Path:
    """Where a Modal eval-set run is saved on the laptop.

    The full run is evidence for the one table, so it goes to runs/, which is
    committed. A --limit smoke test proves nothing, so it goes to the ignored
    results/ folder and can never be mistaken for the real run.
    """
    if limit:
        return REPO / "results" / rung_id / f"evalset-modal-t4-limit{limit}.json"
    return REPO / "runs" / rung_id / "evalset-modal-t4.json"


@app.local_entrypoint()
def main(rung: str = "r0-gpu", evalset: bool = False, limit: int = 0):
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    if evalset:
        out = evaluate.remote(rung, commit, limit)
    else:
        out = bench.remote(rung)
    wall = time.time() - t0

    if evalset:
        # Scoring happens here, from the JSON alone -- the same code that scores
        # the laptop run, so both rows of the table are scored identically.
        sys.path.insert(0, str(REPO / "src"))
        from s2slab.evalrun import report, save
        path = save(out["results"], results_path(rung, limit))
        print(report(out["results"], root=REPO / "evalset"))
        print(f"\nsaved -> {path.relative_to(REPO)}")
    else:
        print(out["table"])
        print(f"\ncommit {commit} · GPU {out['gpu']}")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end "
          f"(queue, cold start and run): about ${wall / 3600 * T4_DOLLARS_PER_HOUR:.3f} "
          f"at the T4 rate. Modal's dashboard has the exact bill.")
