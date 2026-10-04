"""
R4 on Modal: the S2S rung on an L40S, with R3 beside it in the same container.

    modal run modal_app/r4_bench.py::main --rung r4-gpu                      30 probe clips: the smoke test
    modal run modal_app/r4_bench.py::main --rung r3-gpu,r4-gpu --evalset --limit 8
    modal run modal_app/r4_bench.py::main --rung r3-gpu,r4-gpu --evalset     the R4 row, saved to runs/<rung>/evalset-modal-l40s.json
    modal run modal_app/r4_bench.py::pairs --split dev                       PR-008's development: both rungs answer the 21 dev prosody pairs
    modal run modal_app/r4_bench.py::main --rung r0-gpu,r1-gpu,r2-gpu,r2-gpu-turbo,r3-gpu,r4-gpu --evalset --tag ladder
                                                                             M28: the ladder, R0 to R4, in ONE L40S container
    modal run modal_app/r4_bench.py::profile                                 M31, step 1: R4's stages timed on the 30 probe clips
    modal run modal_app/r4_bench.py::speed                                   M31, step 1b: the Talker and the DiT, eager and compiled
The file has four entry points (main, pairs, profile, speed), so Modal needs the one named: ::main, ::pairs,
::profile or ::speed.

What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON and prints the report (scoring needs no GPU)
  image    CUDA 12.6, Python 3.12, torch, transformers, R0's model libraries
           (Kokoro for R3), and the weights: Qwen2.5-Omni-7B (about 22 GB),
           Qwen2-Audio-7B-Instruct and Kokoro for the R3 comparator, and R0 to
           R2's recognizers and LLM (M28) -- all fetched on CPU at build time
  L40S     48 GB: Omni's ~20 GB of bf16 weights plus the activations of a
           15 s clip do not fit the A10, which is why R4 lives on this card and
           R3 is re-measured here rather than read off its A10 row (M13's rule).
           It is the one card that holds every rung, so M28 runs the whole
           ladder here: R0 to R4 in one container, R5 in r5_bench --gpu l40s

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
PROFILE_PATH = REPO / "runs" / "r4-gpu" / "profile-l40s.json"   # M31, step 1 (s2slab.r4profile)
SPEED_PATH = REPO / "runs" / "r4-gpu" / "speed-l40s.json"       # M31, step 1b (s2slab.r4speed)

# Every Hugging Face repo the rungs this file runs will load; tests/test_modal_app.py checks it.
MODELS = {
    "Qwen/Qwen2.5-Omni-7B": None,                               # R4
    "Qwen/Qwen2-Audio-7B-Instruct": None,                       # R3, the comparator
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],   # R3's voice
}

# M28: R0 to R2 in the same container as R3 and R4. Their voice is Kokoro's af_heart, above.
# A layer of its own, after the others, so the cached layers above are not rebuilt.
LADDER_MODELS = {
    "Systran/faster-whisper-base.en": None,                     # R0, R1, R2's live tier
    "Systran/faster-whisper-small.en": None,                    # R2's slow tier
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": None,       # R2 turbo's slow tier
    "Qwen/Qwen2.5-0.5B-Instruct": None,                         # R0 to R2's LLM
}


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


def download_ladder() -> None:
    """R0 to R2's weights (M28), once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in LADDER_MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.17.0", "accelerate==1.15.0", "soundfile==0.14.0", "numpy==2.2.6",
                 "huggingface_hub==1.31.0")
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
    .pip_install("scipy==1.15.3", "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    # Qwen2.5-Omni's processor loads an image and a video processor even for audio-only
    # input, and they import Pillow and torchvision (the first smoke test failed on it).
    # After the weights, so that layer stays cached. 0.29.0 is what pip resolved for
    # torch 2.14.0 on the first build (smoke test at 34a4209), now pinned.
    .pip_install("torch==2.14.0", "torchvision==0.29.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("pillow==11.3.0")
    .apt_install("gcc")                     # torch 2.14's Triton helpers compile a C stub on first use
    .run_function(download_ladder)
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r4", image=image)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _versions() -> str:
    """The library versions the run actually used (torchvision is resolved, not pinned)."""
    import torch
    import torchvision
    import transformers
    return f"torch {torch.__version__}, torchvision {torchvision.__version__}, transformers {transformers.__version__}"


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
    return {"gpu": _gpu_name(), "versions": _versions(), "tables": tables, "seconds": time.perf_counter() - t0}


# R4 at ~8 s a turn: 200 turns ~30 min, R3 ~4; with R0 to R2 before them (M28) about 50 min in all.
@app.function(gpu="L40S", timeout=120 * 60, scaledown_window=60)
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
    return {"results": results, "versions": _versions(), "seconds": time.perf_counter() - t0}


@app.function(gpu="L40S", timeout=45 * 60, scaledown_window=60)
def answer(clips: dict[str, bytes], rung_ids: list[str], repeat: int = 2) -> dict:
    """PR-008: each rung answers every prosody clip (r4probe.answer_clips), one container.
    Returns {"gpu", "versions", "answers": {rung: {"replies", "repeat"}}, "seconds"}."""
    import os
    os.chdir("/root")
    from s2slab.r4probe import answer_clips
    from s2slab.rungs import get
    t0 = time.perf_counter()
    answers = {}
    for rung_id in rung_ids:
        rung = get(rung_id)
        answers[rung_id] = answer_clips(rung, clips, repeat)
        del rung
        _free_gpu()
    return {"gpu": _gpu_name(), "versions": _versions(), "answers": answers, "seconds": time.perf_counter() - t0}


# R4 at ~8 s a turn on 30 clips, plus three first blocks decoded twice each: about 6 min after the load.
@app.function(gpu="L40S", timeout=30 * 60, scaledown_window=60)
def stages() -> dict:
    """M31, step 1: R4's three stages timed inside its own call, on the 30 probe clips (r4profile.run)."""
    import os
    os.chdir("/root")
    from s2slab.probes import TARGET_SR
    from s2slab.r4 import OmniS2S
    from s2slab.r4profile import run
    from s2slab.rungs import probe_clips
    t0 = time.perf_counter()
    blob = run(OmniS2S(), probe_clips(), TARGET_SR)
    return {"gpu": _gpu_name(), "versions": _versions(), "profile": blob, "seconds": time.perf_counter() - t0}


# Ten clips three ways (R4 ~8.6 s a turn eager), two warm-up turns per compiled variant, then the DiT eager and
# compiled on three block sizes: about 10 min after the load, most of it compiling. The timeout leaves room.
@app.function(gpu="L40S", timeout=45 * 60, scaledown_window=60)
def speed_probe() -> dict:
    """M31, step 1b: the Talker and the DiT, eager and compiled, on every third probe clip (r4speed.run)."""
    import os
    os.chdir("/root")
    from s2slab.probes import TARGET_SR
    from s2slab.r4 import OmniS2S
    from s2slab.r4speed import run
    from s2slab.rungs import probe_clips
    t0 = time.perf_counter()
    blob = run(OmniS2S(), probe_clips(), TARGET_SR)
    return {"gpu": _gpu_name(), "versions": _versions(), "speed": blob, "seconds": time.perf_counter() - t0}


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
            refuse_to_overwrite(results_path(r, tag=tag))      # the name this run will save under
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
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def pairs(split: str = "dev", confirm: str = "", save: str = ""):
    """Both rungs answer the prosody pairs of one side (the test side only with a
    registration); the replies' delivery is measured and reported on the laptop."""
    from s2slab.r3probe import clips_for
    from s2slab.r4probe import RUNGS, analyse, compare, format_criteria, format_report, write_replies
    from s2slab.r4probe import save as save_run
    from s2slab.sidechannel import refuse_to_overwrite

    if save:
        refuse_to_overwrite(pathlib.Path(save))
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    prs, clips = clips_for(split, confirm)
    t0 = time.time()
    out = answer.remote(clips, list(RUNGS))
    wall = time.time() - t0
    measures, deltas = analyse(prs, out["answers"])
    write_replies(out["answers"], REPO / "results" / "r4-gpu" / "replies" / split)
    head = (f"{len(prs)} {split} prosody pairs" + (f" · confirms {confirm}" if confirm else "")
            + f" · {' and '.join(RUNGS)} on {out['gpu']} · commit {commit}")
    print(head, "\n")
    print(format_report(prs, deltas, {r: a["repeat"] for r, a in out["answers"].items()}))
    if split == "test":
        print("\n" + format_criteria(compare(deltas), confirm))
    print(f"\nreplies written to results/r4-gpu/replies/{split}/ for listening")
    if save:
        save_run(pathlib.Path(save), head, prs, out["answers"], measures, deltas)
        print(f"saved to {save}")
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def profile():
    """Where R4's wait goes, before anything streams (M31, step 1). Saves runs/r4-gpu/profile-l40s.json,
    never overwritten, and prints the stage table."""
    from s2slab.r4profile import format_profile, save
    from s2slab.sidechannel import refuse_to_overwrite

    refuse_to_overwrite(PROFILE_PATH)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = stages.remote()
    wall = time.time() - t0
    blob = {"run": "M31, step 1: R4's stages timed on the 30 probe clips", "commit": commit,
            "machine": f"Modal {out['gpu']}", "versions": out["versions"], **out["profile"]}
    path = save(PROFILE_PATH, blob)
    print(format_profile(blob))
    print(f"\ncommit {commit} · GPU {out['gpu']}\nsaved -> {path.relative_to(REPO)}")
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def speed():
    """Can R4's Talker and DiT run faster, unchanged in what they make (M31, step 1b)? Saves
    runs/r4-gpu/speed-l40s.json, never overwritten, and prints the two tables and the estimates."""
    from s2slab.r4profile import save
    from s2slab.r4speed import format_speed
    from s2slab.sidechannel import refuse_to_overwrite

    refuse_to_overwrite(SPEED_PATH)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = speed_probe.remote()
    wall = time.time() - t0
    blob = {"run": "M31, step 1b: the Talker and the DiT, eager and compiled, on 10 probe clips", "commit": commit,
            "machine": f"Modal {out['gpu']}", "versions": out["versions"], **out["speed"]}
    path = save(SPEED_PATH, blob)
    print(format_speed(blob))
    print(f"\ncommit {commit} · GPU {out['gpu']}\nsaved -> {path.relative_to(REPO)}")
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")
