"""
R0 on Modal: the reportable GPU row (v2 plan §5).

Colab gave development numbers on a shared T4. This runs the same rungs and the
same code on a Modal T4 billed per second, so the number can be reported
together with what it cost.

    modal run modal_app/r0_bench.py                             # 30 probe clips, r0-gpu
    modal run modal_app/r0_bench.py --rung r0-gpu-bigger
    modal run modal_app/r0_bench.py --evalset --limit 8         # eval-set smoke test
    modal run modal_app/r0_bench.py --evalset                   # all 200 clips: p90 and WER
    modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu        # two rungs, ONE container: probe clips
    modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu --evalset --tag pr002
                                                                # the PR-002 run: same container, same GPU
    modal run modal_app/r0_bench.py --rung r1-gpu,r2-gpu,r2-gpu-turbo
                                                                # R2's slow tier beside R1: probe clips                                                                
    
What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON and prints the report (scoring needs no GPU)
  image    built once on Modal's CPUs and cached: CUDA 12.6 + cuDNN 9, Python
           3.12, the pinned packages, and every model's weights
  T4       loads the models from the image and runs the turns

The full eval-set run is evidence, so its JSON goes to runs/<rung>/, which is
committed. A --limit smoke test goes to results/, which is not. Evidence is never
overwritten: if the file already exists, the run stops before any GPU time is
spent, and --tag gives the new run its own name.

Several rungs (comma-separated) run one after another inside one container
call, so they share the GPU, the image and the cold start, and their numbers
can be compared turn by turn (v1 rule 7; PR-002's V1).

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
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": None,       # R2's larger slow tier, 1.6 GB
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    "Qwen/Qwen2.5-1.5B-Instruct": None,
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
}

# faster-whisper resolves a model name to a Hugging Face repo. Most are
# Systran's conversions; the turbo model was converted by someone else.
WHISPER_REPOS = {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}


def whisper_repo(model_name: str) -> str:
    """The repo faster-whisper will look for, offline, when a rung asks for model_name."""
    return WHISPER_REPOS.get(model_name, f"Systran/faster-whisper-{model_name}")


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


def _free_gpu() -> None:
    """Drop the last rung's models before loading the next, so the second rung
    does not run in whatever memory the first left over."""
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


@app.function(gpu="T4", timeout=20 * 60, scaledown_window=60)
def bench(rung_ids: list[str]) -> dict:
    """Load each rung in turn and run the 30 probe clips through it, on the T4.

    Takes:   registered rung ids, run in the order given, one container.
    Returns: {"gpu": name and driver, "tables": {rung id: the stage table from
             rungs.report()}, "seconds": time spent inside this function}.
    """
    import os
    os.chdir("/root")                       # probes.PROBE_ROOT is relative to here
    from s2slab.probes import TARGET_SR
    from s2slab.rungs import probe_clips, report
    t0 = time.perf_counter()
    tables = {}
    for rung_id in rung_ids:
        tables[rung_id] = report(rung_id, probe_clips(), TARGET_SR)
        _free_gpu()
    return {"gpu": _gpu_name(), "tables": tables, "seconds": time.perf_counter() - t0}


@app.function(gpu="T4", timeout=20 * 60, scaledown_window=60)
def evaluate(rung_ids: list[str], commit: str, limit: int = 0) -> dict:
    """Run each rung in turn over the frozen eval set, on the T4.

    Takes:   registered rung ids, run in the order given, one container; the
             laptop's commit (the container has no git, so the commit that sent
             the code has to travel with it); an optional limit for a smoke
             test (0 = all 200).
    Calls:   evalset.verify() first -- a WAV that differs from the manifest
             means this is not the frozen set, and the run stops before any
             GPU time is spent on it; then evalrun.run() per rung.
    Returns: {"results": {rung id: the evalrun results dict}, "seconds": time inside}.
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
    """Where a Modal eval-set run is saved on the laptop.

    The full run is evidence for the one table, so it goes to runs/, which is
    committed. A --limit smoke test proves nothing, so it goes to the ignored
    results/ folder and can never be mistaken for the real run. A tag names a
    run after what it is for: evalset-modal-t4-pr002.json.
    """
    suffix = f"-{tag}" if tag else ""
    if limit:
        return REPO / "results" / rung_id / f"evalset-modal-t4{suffix}-limit{limit}.json"
    return REPO / "runs" / rung_id / f"evalset-modal-t4{suffix}.json"


def refuse_to_overwrite(paths: list[pathlib.Path]) -> None:
    """Evidence is written once. Checked BEFORE the remote call, so a name clash
    costs nothing; results/ is scratch and may be overwritten freely."""
    taken = [p for p in paths if p.exists() and (REPO / "runs") in p.parents]
    if taken:
        raise SystemExit("already recorded, and evidence is never overwritten:\n  "
                         + "\n  ".join(str(p.relative_to(REPO)) for p in taken)
                         + "\nGive this run its own name with --tag.")


@app.local_entrypoint()
def main(rung: str = "r0-gpu", evalset: bool = False, limit: int = 0, tag: str = ""):
    rung_ids = [r.strip() for r in rung.split(",") if r.strip()]
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    if evalset:
        refuse_to_overwrite([results_path(r, limit, tag) for r in rung_ids])
    t0 = time.time()
    if evalset:
        out = evaluate.remote(rung_ids, commit, limit)
    else:
        out = bench.remote(rung_ids)
    wall = time.time() - t0

    if evalset:
        # Scoring happens here, from the JSON alone -- the same code that scores
        # the laptop run, so every row of the table is scored identically.
        sys.path.insert(0, str(REPO / "src"))
        from s2slab.evalrun import report, save
        for rung_id, results in out["results"].items():
            path = save(results, results_path(rung_id, limit, tag))
            print(report(results, root=REPO / "evalset"))
            print(f"\nsaved -> {path.relative_to(REPO)}\n")
    else:
        for table in out["tables"].values():
            print(table, "\n")
        print(f"commit {commit} · GPU {out['gpu']}")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end "
          f"(queue, cold start and run): about ${wall / 3600 * T4_DOLLARS_PER_HOUR:.3f} "
          f"at the T4 rate. Modal's dashboard has the exact bill.")
