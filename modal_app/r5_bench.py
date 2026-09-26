"""
R5 on Modal: full-duplex models on an A10, driven by s2slab.duplex.

    modal run modal_app/r5_bench.py                                  30 probe clips: the smoke test (dev)
    modal run modal_app/r5_bench.py --evalset --limit 8              eval-set smoke test -> results/
    modal run modal_app/r5_bench.py --evalset                        the R5 row -> runs/r5-moshi/evalset-modal-a10.json
    modal run modal_app/r5_bench.py --model r5-personaplex           the same three, for PersonaPlex
    modal run modal_app/r5_bench.py::dialogues --limit 50            FT-1 dev: Moshi answering the user
                                                                     channel of DailyTalk dev dialogues
    modal run modal_app/r5_bench.py::pr010 --confirm PR-010          FT-1's registered test: base Moshi,
                                                                     then Moshi + LoRA, one L40S container

What happens where:
  laptop   sends src/s2slab, the probe folder and the eval set; afterwards
           saves the run's JSON, writes the listening files, prints the report
  image    ONE PER MODEL, because each brings its own package named `moshi`:
             Moshi       Kyutai's `moshi` 0.2.13 pins torch below 2.10, so torch
                         2.8 (CUDA 12.6), not the 2.14 of the R0-R4 images;
                         kyutai/moshiko-pytorch-bf16 (CC-BY-4.0, not gated)
             PersonaPlex NVIDIA's fork (github NVIDIA/personaplex, MIT, pinned
                         by commit) pins torch below 2.5 and numpy below 2.2, so
                         torch 2.4.1 (CUDA 12.4); nvidia/personaplex-7b-v1
                         (NVIDIA Open Model License, gated: the build reads the
                         HF token from the Modal secret HF_SECRET, as HF_TOKEN)
           weights baked in on CPU at build time, pinned where the repo allows
  A10      loads the model once, then one conversation per clip: duplex.LEAD_S of
           silence (the model opens the call), the clip, then duplex.TAIL_S of
           silence, one 80 ms frame at a time

The smoke test also sends back the first LISTEN conversations as stereo WAVs
(caller left, the model right, on one timeline) under results/<model>/listen/,
so the voiced threshold and the gaps can be checked by ear before the real run.

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
L40S_DOLLARS_PER_HOUR = 1.95    # Modal's L40S rate, checked 2026-08-19 (v2 §5)
MODELS = {"kyutai/moshiko-pytorch-bf16": None}
LISTEN = 3

# PersonaPlex: the same values as s2slab.r5 (a test checks), written out here because
# the build step runs before src/s2slab is added to the image.
PERSONAPLEX_REPO = "nvidia/personaplex-7b-v1"
PERSONAPLEX_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"      # the weights on 2026-09-24
PERSONAPLEX_FILES = ["config.json", "model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
                     "tokenizer_spm_32k_3.model", "voices.tgz"]
PERSONAPLEX_CODE = ("git+https://github.com/NVIDIA/personaplex.git"
                    "@3428dfd95309a7f3c84fd93259ded0f810d1ff91#subdirectory=moshi")  # main on 2026-09-24
HF_SECRET = "huggingface-secret"


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


def download_personaplex() -> None:
    """Runs once, inside the PersonaPlex image build, on CPU, with the HF token.
    The voices are unpacked beside the weights, where the package's offline.py looks for them."""
    import tarfile

    from huggingface_hub import snapshot_download
    d = pathlib.Path(snapshot_download(PERSONAPLEX_REPO, revision=PERSONAPLEX_REVISION,
                                       allow_patterns=PERSONAPLEX_FILES))
    with tarfile.open(d / "voices.tgz", "r:gz") as tar:
        tar.extractall(path=d)
    print("voices:", sorted(p.name for p in (d / "voices").iterdir()))


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

personaplex_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    # git to install the package from GitHub; the rest as the repo's Dockerfile (Opus) and
    # the Moshi image (gcc, PortAudio) have them.
    .apt_install("git", "gcc", "pkg-config", "libopus-dev", "libportaudio2")
    # The package requires torch >=2.2,<2.5 and numpy >=1.26,<2.2 (its pyproject.toml at the pinned
    # commit). It imports tqdm without declaring it; the pins below all have Python 3.12 wheels.
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(f"moshi-personaplex @ {PERSONAPLEX_CODE}", "numpy==1.26.4", "scipy==1.13.1",
                 "soundfile==0.14.0", "tqdm==4.67.1", "praat-parselmouth==0.4.7")
    .run_function(download_personaplex, secrets=[modal.Secret.from_name(HF_SECRET)])
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "probes" / "endpointing", "/root/probes/endpointing")
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r5", image=image)
# FT-1's adapter, copied off the Vast machine and put here with `modal volume put ft1 ...` (vast/README.md).
ft1_volume = modal.Volume.from_name("ft1", create_if_missing=True)


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def _versions() -> str:
    import importlib.metadata as md

    import torch
    got = []
    for dist in ("moshi", "moshi-personaplex"):
        try:
            got.append(f"{dist} {md.version(dist)}")
        except md.PackageNotFoundError:
            pass
    return ", ".join([f"torch {torch.__version__}", *got])


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


def _smoke(model_id: str, commit: str) -> dict:
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


def _evaluate(model_id: str, commit: str, limit: int = 0) -> dict:
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


@app.function(gpu="A10G", timeout=30 * 60, scaledown_window=60)
def smoke(model_id: str, commit: str) -> dict:
    return _smoke(model_id, commit)


# Two hours: 200 conversations of LEAD_S + clip + TAIL_S at RTF 0.7-0.8 are about 50 minutes of
# stream work, and PersonaPlex adds about 5 s of prompts per call (its probe smoke test, 2026-09-24).
@app.function(gpu="A10G", timeout=120 * 60, scaledown_window=60)
def evaluate(model_id: str, commit: str, limit: int = 0) -> dict:
    return _evaluate(model_id, commit, limit)


@app.function(gpu="A10G", timeout=30 * 60, scaledown_window=60, image=personaplex_image)
def smoke_personaplex(model_id: str, commit: str) -> dict:
    return _smoke(model_id, commit)


@app.function(gpu="A10G", timeout=120 * 60, scaledown_window=60, image=personaplex_image)
def evaluate_personaplex(model_id: str, commit: str, limit: int = 0) -> dict:
    return _evaluate(model_id, commit, limit)


# --------------------------------------------------------------------------- #
# FT-1: Moshi answering the user channel of DailyTalk dialogues (dev fold only)
# --------------------------------------------------------------------------- #

DIALOGUE_TAIL_S = 4.0      # silence after the dialogue: time for the model's last answer


def _dialogues(model_id: str, commit: str, users: dict, listen: int = 0) -> dict:
    """users: {dialogue number: the user (right) channel as a 24 kHz 16-bit WAV}. Each is streamed
    through the model from its first frame -- no lead-in: the dialogue's own opening silence on the
    user's side is where the answerer speaks first -- then DIALOGUE_TAIL_S of silence. Returns
    turngaps.summary(user, model) per dialogue, the run's measures, and the first `listen`
    conversations as stereo WAVs (the user left, the model right)."""
    import io
    import os

    import soundfile as sf
    import torch
    os.chdir("/root")
    from s2slab import duplex, r5
    from s2slab.turngaps import ipus, summary
    torch.cuda.reset_peak_memory_stats()
    model = r5.get(model_id)
    t0 = time.perf_counter()
    model.load()
    load_s = time.perf_counter() - t0
    rows, wavs = {}, {}
    for k, n in enumerate(sorted(users)):
        user, sr = sf.read(io.BytesIO(users[n]), dtype="float32")
        conv = duplex.converse(model, user, sr=sr, lead_s=0.0, tail_s=DIALOGUE_TAIL_S)
        both = duplex.stereo(user, conv, sr)
        rows[n] = {**summary(ipus(both[:, 0], sr), ipus(both[:, 1], sr)),
                   "rtf": round(conv.wall_s / conv.stream_s, 3), "setup_s": round(conv.begin_s, 3),
                   "text": "".join(t for t in conv.text if t).strip()}
        if k < listen:
            buf = io.BytesIO()
            sf.write(buf, both, sr, format="WAV", subtype="PCM_16")
            wavs[n] = buf.getvalue()
    results = {"rung": model.rung_id, "description": model.description, "load_s": round(load_s, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20),
               "machine": f"Modal {_gpu_name()}", "commit": commit, "versions": _versions(),
               "tail_s": DIALOGUE_TAIL_S, "rows": rows}
    return {"results": results, "listen": wavs, "seconds": time.perf_counter() - t0}


@app.function(gpu="A10G", timeout=60 * 60, scaledown_window=60)
def dialogues_remote(model_id: str, commit: str, users: dict, listen: int = 0) -> dict:
    return _dialogues(model_id, commit, users, listen)


@app.local_entrypoint()
def dialogues(model: str = "r5-moshi", limit: int = 0, listen: int = 3):
    """FT-1 dev: the model answers the user channel of the DailyTalk dev dialogues on this machine;
    the human reference is measured here, on the laptop, from the same files."""
    import io
    import json

    import soundfile as sf

    from s2slab import dailytalk
    if model != "r5-moshi":
        raise SystemExit("the dialogues run is Moshi's (FT-1); PersonaPlex has its own image and is not part of FT-1")
    os_root = REPO / dailytalk.ROOT
    ids = dailytalk.dialogue_ids("dev", os_root)[: limit or None]
    if not ids:
        raise SystemExit(f"no dev dialogues under {os_root}: download them first (see FT-1 part 1)")
    users, ref = {}, {}
    for n in ids:
        left, right = dailytalk.load(n, os_root)
        ref[n] = dailytalk.reference(left, right)
        buf = io.BytesIO()
        sf.write(buf, right, dailytalk.SR, format="WAV", subtype="PCM_16")
        users[n] = buf.getvalue()
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = dialogues_remote.remote(model, commit, users, listen)
    wall = time.time() - t0
    results = out["results"]
    results["reference"] = ref
    path = REPO / "results" / model / f"dailytalk-dev-{len(ids)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")   # results/ is scratch
    for n, data in out["listen"].items():
        d = REPO / "results" / model / "dailytalk-listen"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"dev-{n}.wav").write_bytes(data)
    print(f"{results['rung']}: {results['description']}")
    print(f"{results['machine']} · commit {commit} · load {results['load_s']:.0f} s · peak GPU memory {results['peak_vram_mb']:,} MB\n")
    print(dailytalk.report(results["rows"], ref, model))
    print(f"\nsaved -> {path.relative_to(REPO)}" + (f"; listen: results/{model}/dailytalk-listen/" if out["listen"] else ""))
    print(results.get("versions", ""))
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


# An L40S, not the registered A10 (PR-010, Deviations): on the A10 (22 GB) the first run ran out of
# memory fusing the LoRA into the second model. Moshi + LoRA alone peaks at 29.3 GiB (the repeat).
@app.function(gpu="L40S", timeout=150 * 60, scaledown_window=60, volumes={"/ft1": ft1_volume})
def pr010_remote(commit: str, users: dict, listen: int = 0) -> dict:
    """PR-010: base Moshi, then Moshi + FT-1's LoRA, in ONE container, on the same dialogues and seeds.
    Records the GPU memory the base pass leaves allocated, so the first run's crash can be understood."""
    import gc
    import hashlib
    import json as _json

    import torch
    adapter = pathlib.Path("/ft1/lora.safetensors")
    if not adapter.exists() or not pathlib.Path("/ft1/config.json").exists():
        raise RuntimeError("the ft1 volume has no lora.safetensors and config.json: see vast/README.md, step 6")
    base = _dialogues("r5-moshi", commit, users, listen)
    gc.collect()
    torch.cuda.empty_cache()
    left_mb = round(torch.cuda.memory_allocated() / 2 ** 20)
    lora = _dialogues("r5-moshi-lora", commit, users, listen)
    return {"base": base, "lora": lora, "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            "adapter_config": _json.loads(pathlib.Path("/ft1/config.json").read_text()),
            "gpu_mb_left_after_base": left_mb}


@app.local_entrypoint()
def pr010(confirm: str = "", listen: int = 3):
    """PR-010's one registered run: the 100 test dialogues, base then LoRA, the criterion, the evidence."""
    import io
    import json

    import soundfile as sf

    from s2slab import dailytalk
    from s2slab.sidechannel import refuse_to_overwrite
    if confirm != dailytalk.REGISTRATION:
        raise SystemExit(f"the test dialogues are locked: run with --confirm {dailytalk.REGISTRATION}, once")
    path = REPO / "runs" / "ft1" / "pr010.json"
    refuse_to_overwrite(path)
    root = REPO / dailytalk.ROOT
    missing = [n for n in dailytalk.TEST_IDS if not (root / "data_stereo" / f"{n}.wav").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test dialogues are not downloaded (first: {missing[0]}): see vast/README.md, step 1")
    users, ref = {}, {}
    for n in dailytalk.TEST_IDS:
        left, right = dailytalk.load(n, root, confirm=confirm)
        ref[n] = dailytalk.reference(left, right)
        buf = io.BytesIO()
        sf.write(buf, right, dailytalk.SR, format="WAV", subtype="PCM_16")
        users[n] = buf.getvalue()
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = pr010_remote.remote(commit, users, listen)
    wall = time.time() - t0
    base, lora = out["base"]["results"], out["lora"]["results"]
    results = {"registration": dailytalk.REGISTRATION, "commit": commit, "ids": list(dailytalk.TEST_IDS),
               "adapter_sha256": out["adapter_sha256"], "adapter_config": out["adapter_config"],
               "machine": base["machine"], "versions": base["versions"], "tail_s": base["tail_s"],
               "gpu_mb_left_after_base": out["gpu_mb_left_after_base"],
               "base": base, "lora": lora, "reference": ref}
    results["criteria"] = dailytalk.pr010(base["rows"], lora["rows"], ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    for tag in ("base", "lora"):
        for n, data in out[tag]["listen"].items():
            d = REPO / "results" / "ft1" / "listen"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"test-{n}-{tag}.wav").write_bytes(data)
    print(f"{base['machine']} · commit {commit} · adapter sha256 {out['adapter_sha256'][:12]}")
    print(f"peak GPU memory: base {base['peak_vram_mb']:,} MB, LoRA {lora['peak_vram_mb']:,} MB; "
          f"left allocated after the base pass: {out['gpu_mb_left_after_base']:,} MB\n")
    print("## base Moshi\n\n" + dailytalk.report(base["rows"], ref, "base Moshi"))
    print("\n## Moshi + FT-1's LoRA\n\n" + dailytalk.report(lora["rows"], ref, "Moshi + LoRA"))
    print("\n" + dailytalk.format_pr010(results))
    print(f"\nsaved -> {path.relative_to(REPO)}; listen: results/ft1/listen/")
    print(f"{out['base']['seconds'] + out['lora']['seconds']:.0f} s inside the function, {wall:.0f} s end to end: "
          f"about ${wall / 3600 * L40S_DOLLARS_PER_HOUR:.3f} at the L40S rate. Modal's dashboard has the exact bill.")


def functions(model_id: str):
    """(smoke, evaluate) on the image that carries this model's package."""
    return {"r5-moshi": (smoke, evaluate), "r5-personaplex": (smoke_personaplex, evaluate_personaplex)}[model_id]


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

    run_smoke, run_evaluate = functions(model)
    path = results_path(model, evalset, limit, tag)
    if (REPO / "runs") in path.parents:
        refuse_to_overwrite(path)
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = run_evaluate.remote(model, commit, limit) if evalset else run_smoke.remote(model, commit)
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
