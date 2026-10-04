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
    modal run modal_app/r4_bench.py::speed --tag mask                        the same with the Talker's mask fixed (second run)
    modal run modal_app/r4_bench.py::streamdev                               M31, step 2: R4, r4-gpu-fast and the stream's settings, probe clips
    modal run modal_app/r4_bench.py::pr015 --confirm PR-015                  PR-015, once: the three on the eval set
The file has six entry points (main, pairs, profile, speed, streamdev, pr015), so Modal needs the one named:
::main, ::pairs, ::profile, ::speed, ::streamdev or ::pr015.

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
STREAM_DEV_PATH = REPO / "runs" / "r4-gpu" / "stream-dev-l40s.json"   # M31, step 2 (s2slab.r4streamdev)


def speed_path(tag: str = "") -> pathlib.Path:
    """runs/r4-gpu/speed-l40s[-tag].json: each run of the speed probe under its own name."""
    return SPEED_PATH.with_name(f"speed-l40s-{tag}.json") if tag else SPEED_PATH


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


# 30 clips through five rungs, R4 ~8.6 s a turn, the others faster, each reply heard by Whisper, and the compiles:
# about 25 min after the load. The timeout leaves room.
@app.function(gpu="L40S", timeout=75 * 60, scaledown_window=60)
def stream_dev() -> dict:
    """M31, step 2: R4, r4-gpu-fast and r4-gpu-stream at each setting, the 30 probe clips (r4streamdev.run)."""
    import os
    os.chdir("/root")
    from s2slab.probes import TARGET_SR
    from s2slab.r4streamdev import run
    from s2slab.rungs import probe_clips
    t0 = time.perf_counter()
    blob = run(probe_clips(), TARGET_SR)
    return {"gpu": _gpu_name(), "versions": _versions(), "dev": blob, "seconds": time.perf_counter() - t0}


# 200 clips through three rungs (R4 ~7 s a turn on the eval set, the others about as long), each reply heard
# by Whisper: 75 to 120 min after the load, by the container's speed. The timeout leaves room.
@app.function(gpu="L40S", timeout=180 * 60, scaledown_window=60)
def pr015_run(confirm: str) -> dict:
    """PR-015's one run: R4, r4-gpu-fast and r4-gpu-stream as frozen, on the 200 eval clips (pr015.run)."""
    import os
    os.chdir("/root")
    from s2slab.evalset import TARGET_SR
    from s2slab.pr015 import eval_clips, run
    clips, cohorts = eval_clips(confirm)
    t0 = time.perf_counter()
    blob = run(clips, TARGET_SR, cohorts)
    return {"gpu": _gpu_name(), "versions": _versions(), "run": blob, "seconds": time.perf_counter() - t0}


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
def speed(tag: str = ""):
    """Can R4's Talker and DiT run faster, unchanged in what they make (M31, step 1b)? Saves
    runs/r4-gpu/speed-l40s[-tag].json, never overwritten, and prints the two tables and the estimates."""
    from s2slab.r4profile import save
    from s2slab.r4speed import format_speed
    from s2slab.sidechannel import refuse_to_overwrite

    out_path = speed_path(tag)
    refuse_to_overwrite(out_path)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = speed_probe.remote()
    wall = time.time() - t0
    blob = {"run": "M31, step 1b: the Talker and the DiT, eager and compiled, on 10 probe clips"
                   + (f" ({tag})" if tag else ""), "commit": commit,
            "machine": f"Modal {out['gpu']}", "versions": out["versions"], **out["speed"]}
    path = save(out_path, blob)
    print(format_speed(blob))
    print(f"\ncommit {commit} · GPU {out['gpu']}\nsaved -> {path.relative_to(REPO)}")
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def streamdev():
    """The streamed R4 on development clips (M31, step 2): saves runs/r4-gpu/stream-dev-l40s.json, never
    overwritten; three replies of each rung to results/r4-gpu/streamdev/ for listening (not committed)."""
    from s2slab.r4profile import save
    from s2slab.r4streamdev import format_dev
    from s2slab.sidechannel import refuse_to_overwrite

    refuse_to_overwrite(STREAM_DEV_PATH)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = stream_dev.remote()
    wall = time.time() - t0
    listen = out["dev"].pop("listen")
    blob = {"run": "M31, step 2: R4, r4-gpu-fast and r4-gpu-stream on the 30 probe clips (development)",
            "commit": commit, "machine": f"Modal {out['gpu']}", "versions": out["versions"], **out["dev"]}
    path = save(STREAM_DEV_PATH, blob)
    folder = REPO / "results" / "r4-gpu" / "streamdev"
    for name, data in listen.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    print(format_dev(blob))
    print(f"\ncommit {commit} · GPU {out['gpu']}\nsaved -> {path.relative_to(REPO)}; "
          f"{len(listen)} replies for listening -> {folder.relative_to(REPO)}/")
    print(out["versions"])
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def pr015(confirm: str = ""):
    """PR-015, once (docs/preregistration.md): refused without --confirm PR-015, if its run file exists, or from
    a tree with uncommitted changes. Saves runs/r4-gpu/pr015.json and prints the criteria."""
    from s2slab.pr015 import REGISTRATION, RUN, criteria, format_pr015, save
    from s2slab.sidechannel import refuse_to_overwrite

    if confirm != REGISTRATION:
        raise SystemExit(f"add --confirm {REGISTRATION}: the eval set is its confirmation's, and it runs once")
    refuse_to_overwrite(REPO / RUN)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    if commit.endswith("-dirty"):
        raise SystemExit(f"{commit}: commit first; {REGISTRATION} runs from a clean commit")
    t0 = time.time()
    out = pr015_run.remote(confirm)
    wall = time.time() - t0
    blob = {"registration": REGISTRATION, "commit": commit, "machine": f"Modal {out['gpu']}",
            "versions": out["versions"], **out["run"]}
    path = save(blob, REPO / RUN)
    print(format_pr015(blob, criteria(blob)))
    print(f"\nsaved -> {path.relative_to(REPO)}")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")
