"""
R5 on Modal: full-duplex Moshi on an A10, driven by s2slab.duplex.

    modal run modal_app/r5_bench.py                                  30 probe clips: the smoke test (dev)
    modal run modal_app/r5_bench.py --evalset --limit 8              eval-set smoke test -> results/
    modal run modal_app/r5_bench.py --evalset                        the R5 row -> runs/r5-moshi/evalset-modal-a10.json

What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON, writes the listening files, prints the report
  image    its OWN image: the `moshi` package pins torch below 2.10, so this is
           torch 2.8 (CUDA 12.6), not the 2.14 of the R0-R4 images; Moshi's
           weights (kyutai/moshiko-pytorch-bf16, CC-BY-4.0, not gated) baked
           in on CPU at build time
  A10      loads Moshi once, then one conversation per clip: the clip, then
           duplex.TAIL_S of silence, one 80 ms frame at a time

The smoke test also sends back the first LISTEN conversations as stereo WAVs
(caller left, Moshi right, on one timeline) under results/r5-moshi/listen/, so
the voiced threshold and the gaps can be checked by ear before the real run.

The same guardrails as the other benches: explicit timeout, no container kept
warm, scale down 60 s after the call, HF_HUB_OFFLINE=1, evidence never overwritten.
"""

import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
A10_DOLLARS_PER_HOUR = 1.10     # Modal's A10 rate, checked 2026-08-19 (v2 §5)
MODELS = {"kyutai/moshiko-pytorch-bf16": None}
LISTEN = 3


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    # moshi 0.2.13 requires torch >=2.2,<2.10 and numpy <2.3 (its PyPI metadata, read 2026-09-24).
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("moshi==0.2.13", "numpy==2.2.6", "scipy==1.15.3", "soundfile==0.14.0")
    .run_function(download_weights)
    # Praat, because the probe loader counts syllable nuclei (prosody.count_syllable_nuclei) as it
    # loads -- the first smoke test died there. After the weights, so that layer stays cached.
    .pip_install("praat-parselmouth==0.4.7")
    # gcc for torch's Triton helpers (as in every GPU image here); PortAudio because the
    # package depends on sounddevice, which looks for the library when imported.
    .apt_install("gcc", "libportaudio2")
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r5", image=image)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _versions() -> str:
    import importlib.metadata as md

    import torch
    return f"torch {torch.__version__}, moshi {md.version('moshi')}"


def _drive(model_id: str, items: list, listen: int = 0) -> tuple[dict, dict]:
    """Load one model, run every item (duplex.run_items); return (results, {id: stereo WAV bytes})."""
    import io

    import soundfile as sf
    import torch

    from s2slab import duplex, r5
    torch.cuda.reset_peak_memory_stats()
    model = r5.get(model_id)
    t0 = time.perf_counter()
    model.load()
    load_s = time.perf_counter() - t0
    got = duplex.run_items(model, items)
    wavs = {}
    for it in items[:listen]:
        buf = io.BytesIO()
        sf.write(buf, duplex.stereo(it.audio, duplex.converse(model, it.audio)), duplex.SR,
                 format="WAV", subtype="PCM_16")
        wavs[it.id] = buf.getvalue()
    results = {"rung": model.rung_id, "description": model.description, "load_s": round(load_s, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20), **got}
    return results, wavs


@app.function(gpu="A10G", timeout=30 * 60, scaledown_window=60)
def smoke(model_id: str, commit: str) -> dict:
    """The 30 probe clips (dev), with LISTEN stereo files for listening."""
    import os
    os.chdir("/root")
    from s2slab.rungs import probe_clips

    class Item:
        def __init__(self, i, a):
            self.id, self.cohort, self.audio = i, "probe", a

    items = [Item(i, a) for i, a in probe_clips().items()]
    t0 = time.perf_counter()
    results, wavs = _drive(model_id, items, LISTEN)
    results.update(machine=f"Modal {_gpu_name()}", commit=commit, versions=_versions())
    return {"results": results, "listen": wavs, "seconds": time.perf_counter() - t0}


@app.function(gpu="A10G", timeout=60 * 60, scaledown_window=60)
def evaluate(model_id: str, commit: str, limit: int = 0) -> dict:
    """The frozen eval set, verified first."""
    import os
    os.chdir("/root")
    from s2slab.evalset import load, verify
    problems = verify()
    if problems:
        raise RuntimeError("the eval set on Modal is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    items = load()
    items = items[:limit] if limit else items
    t0 = time.perf_counter()
    results, _ = _drive(model_id, items)
    results.update(machine=f"Modal {_gpu_name()}", commit=commit, versions=_versions())
    return {"results": results, "seconds": time.perf_counter() - t0}


def results_path(model_id: str, evalset: bool, limit: int = 0, tag: str = "") -> pathlib.Path:
    """runs/<model>/evalset-modal-a10[-tag].json for the full eval set (committed evidence);
    results/<model>/... for a smoke test or a --limit run (ignored)."""
    suffix = f"-{tag}" if tag else ""
    if not evalset:
        return REPO / "results" / model_id / f"probes-modal-a10{suffix}.json"
    if limit:
        return REPO / "results" / model_id / f"evalset-modal-a10{suffix}-limit{limit}.json"
    return REPO / "runs" / model_id / f"evalset-modal-a10{suffix}.json"


@app.local_entrypoint()
def main(model: str = "r5-moshi", evalset: bool = False, limit: int = 0, tag: str = ""):
    from s2slab.duplex import report, save
    from s2slab.sidechannel import refuse_to_overwrite

    path = results_path(model, evalset, limit, tag)
    if (REPO / "runs") in path.parents:
        refuse_to_overwrite(path)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = evaluate.remote(model, commit, limit) if evalset else smoke.remote(model, commit)
    wall = time.time() - t0
    results = out["results"]
    if path.exists():
        path.unlink()                                   # results/ is scratch: a smoke test may be redone
    save(results, path)
    for clip_id, data in out.get("listen", {}).items():
        d = REPO / "results" / model / "listen"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{clip_id}.wav").write_bytes(data)
    print(report(results))
    print(f"\nsaved -> {path.relative_to(REPO)}")
    if out.get("listen"):
        print(f"listen: results/{model}/listen/ -- caller on the left, {model} on the right")
    print(results.get("versions", ""))
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")
