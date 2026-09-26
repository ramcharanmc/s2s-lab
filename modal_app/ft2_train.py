"""
FT-2 on Modal: the Whisper LoRA trained on an A10, with the ft/ project run exactly as on the laptop.

    modal run modal_app/ft2_train.py::dev                              30 steps on the real data: does the
                                                                       whole loop run there (about $0.10)
    modal run modal_app/ft2_train.py::train                            the experiment, 900 steps (~$0.40)
    modal run modal_app/ft2_train.py::evaluate                         the dev look: base whisper-small.en on
                                                                       all 250 validation clips
    modal run modal_app/ft2_train.py::evaluate --adapter RUN/adapters/step-000900      the LoRA, same clips
    modal run modal_app/ft2_train.py::export --adapter RUN/adapters/step-000900        merged + CTranslate2, CPU

What happens where:
  laptop  sends ft/src, ft/configs and src/s2slab; afterwards saves the run's summary under
          results/ft2/ and prints the validations and the cost line
  volume  `ft2`, persistent:
            /ft2/data/disfluencyspeech   the train and validation parquet files (1.4 GB), fetched ONCE
                                         by a CPU function -- never on the GPU's meter (v2 §6.1)
            /ft2/runs/<name>             a training run: hydra config, csv log, every adapter
                                         (adapters/step-NNNNNN), every val-*.jsonl, the checkpoints
            /ft2/evals/<name>            a dev look: metrics.json and its val-*.jsonl
            /ft2/export/<name>           hf/ (merged transformers model) and ct2/ (faster-whisper's)
          Bring anything back with:  modal volume get ft2 runs/<name> results/ft2/
  image   CUDA torch 2.14 and the exact versions ft/uv.lock resolved for Python 3.12; the base
          model baked in at build time (openai/whisper-small.en, MIT); HF_HUB_OFFLINE=1 at run time
  A10     `python src/train.py experiment=ft2_whisper_verbatim ...` as a subprocess, cwd /root/ft:
          the same command, configs and code as the laptop, so a run here is reproducible there.
          The paths are the only overrides added (data on the volume, the run directory).

The cap. Every function's timeout comes from a dollar figure at the A10 rate (CAP_DOLLARS for a
training run, DEV_CAP_DOLLARS for the smoke run), and the trainer gets `max_time` a little inside
it, so a run that overruns stops itself and keeps its adapters instead of being killed. The other
guardrails as everywhere: no container kept warm, scale down 60 s after the call, a run directory
is never reused.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
A10_DOLLARS_PER_HOUR = 1.10     # Modal's A10 rate, checked 2026-08-19 (v2 §5)
CAP_DOLLARS = 1.50              # one training run; 900 steps should cost about a quarter of this
DEV_CAP_DOLLARS = 0.30          # the 30-step smoke run
EVAL_CAP_DOLLARS = 0.50         # 250 clips generated twice over at most
EXPERIMENT = "ft2_whisper_verbatim"
MODELS = {"openai/whisper-small.en": None}          # baked into the image; ft/configs/model/whisper_lora.yaml's model
# The versions ft/uv.lock resolved for Python 3.12 (a test reads the lock and checks). torch is
# the CUDA build of the same version.
TORCH = "torch==2.14.0"
PINS = ["pytorch-lightning==2.6.6", "torchmetrics==1.9.0", "transformers==5.17.0", "peft==0.21.0",
        "safetensors==0.8.0", "huggingface-hub==1.33.0", "pyarrow==25.0.1", "soundfile==0.14.0",
        "numpy==2.5.3", "scipy==1.18.1", "hydra-core==1.3.7", "hydra-colorlog==1.2.0", "rich==15.0.0",
        "ctranslate2==4.8.2", "tokenizers==0.23.2"]
FT = "/root/ft"
VOLUME = "/ft2"
DEV_OVERRIDES = ["trainer.max_steps=30", "trainer.val_check_interval=15", "data.val_limit=8",
                 "trainer.log_every_n_steps=5", "trainer.num_sanity_val_steps=0"]


def seconds_for(dollars: float, rate: float = A10_DOLLARS_PER_HOUR) -> int:
    """How long a cap buys at the card's rate."""
    return int(dollars / rate * 3600)


def max_time(seconds: int, margin_s: int = 300) -> str:
    """Lightning's max_time, `DD:HH:MM:SS`, a margin inside the container's timeout."""
    s = max(seconds - margin_s, 60)
    return f"00:{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install(TORCH, index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(*PINS)
    .run_function(download_weights)
    .apt_install("gcc")                                     # torch's Triton helpers, as in every GPU image here
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": f"{FT}:/root", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "ft" / "src", f"{FT}/src", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "ft" / "configs", f"{FT}/configs")
)

app = modal.App("s2s-lab-ft2", image=image)
volume = modal.Volume.from_name("ft2", create_if_missing=True)


def _run(cmd: list[str], log_path: pathlib.Path | None = None) -> None:
    """The command as a subprocess in ft/, its output streamed and (if asked) kept."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=FT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log:
                log.write(line)
    finally:
        if log:
            log.close()
    if proc.wait() != 0:
        raise RuntimeError(f"exit code {proc.returncode}: {' '.join(cmd)}")


def _machine() -> str:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return f"Modal {out}" if out else "Modal (no GPU)"


def _validations(run_dir: pathlib.Path, root: str = VOLUME) -> list[dict]:
    """Every validation's metrics, from the val-step-*.jsonl the module writes (first line)."""
    out = []
    for p in sorted(run_dir.rglob("val-step-*.jsonl")):
        with open(p, encoding="utf-8") as f:
            first = json.loads(f.readline())
        out.append({"step": int(p.stem.split("-")[-1]), **first["metrics"], "file": str(p.relative_to(root))})
    return out


@app.function(timeout=30 * 60, volumes={VOLUME: volume})
def fetch_data() -> dict:
    """The dataset onto the volume, on a CPU: HF_HUB_OFFLINE lifted for this one function."""
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    from src.data.disfluencyspeech_datamodule import FILES, REVISION, DisfluencySpeechDataModule
    dm = DisfluencySpeechDataModule(data_dir=f"{VOLUME}/data/disfluencyspeech")
    t0 = time.time()
    dm.prepare_data()
    volume.commit()
    sizes = {f: (dm.root / f).stat().st_size for s in FILES for f in FILES[s]}
    return {"revision": REVISION, "files": sizes, "mb": round(sum(sizes.values()) / 1e6), "seconds": round(time.time() - t0)}


@app.function(gpu="A10G", timeout=seconds_for(CAP_DOLLARS), scaledown_window=60, volumes={VOLUME: volume}, cpu=4)
def train_remote(name: str, commit: str, overrides: list[str], cap_seconds: int) -> dict:
    """One training run of the experiment, into /ft2/runs/<name>."""
    run_dir = pathlib.Path(VOLUME) / "runs" / name
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} exists: a run directory is never reused")
    run_dir.mkdir(parents=True)
    t0 = time.time()
    _run([sys.executable, "src/train.py", f"experiment={EXPERIMENT}", f"paths.data_dir={VOLUME}/data",
          f"paths.log_dir={VOLUME}/runs", f"hydra.run.dir={run_dir}", f"trainer.max_time={max_time(cap_seconds)}",
          f"tags=[ft2,{commit}]", *overrides], run_dir / "console.log")
    volume.commit()
    return {"name": name, "run_dir": str(run_dir), "machine": _machine(), "seconds": time.time() - t0,
            "validations": _validations(run_dir),
            "adapters": sorted(str(p.relative_to(VOLUME)) for p in run_dir.rglob("adapters/step-*"))}


@app.function(gpu="A10G", timeout=seconds_for(EVAL_CAP_DOLLARS), scaledown_window=60, volumes={VOLUME: volume}, cpu=4)
def evaluate_remote(name: str, adapter: str | None) -> dict:
    """The dev look: eval.py over the whole validation split, into /ft2/evals/<name>."""
    out_dir = pathlib.Path(VOLUME) / "evals" / name
    if out_dir.exists():
        raise RuntimeError(f"{out_dir} exists: a run directory is never reused")
    out_dir.mkdir(parents=True)
    if adapter and not (pathlib.Path(VOLUME) / adapter / "adapter_model.safetensors").exists():
        raise RuntimeError(f"no adapter at {VOLUME}/{adapter}")
    t0 = time.time()
    _run([sys.executable, "src/eval.py", f"adapter_dir={VOLUME}/{adapter}" if adapter else "adapter_dir=null",
          f"paths.data_dir={VOLUME}/data", f"paths.log_dir={VOLUME}/evals", f"hydra.run.dir={out_dir}"],
         out_dir / "console.log")
    volume.commit()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"name": name, "adapter": adapter, "machine": _machine(), "seconds": time.time() - t0, **metrics,
            "validations": _validations(out_dir)}


@app.function(timeout=30 * 60, volumes={VOLUME: volume}, memory=8192)
def export_remote(name: str, adapter: str, quantization: str) -> dict:
    """Merge + CTranslate2 on a CPU, into /ft2/export/<name>."""
    import hashlib
    out_dir = pathlib.Path(VOLUME) / "export" / name
    if out_dir.exists():
        raise RuntimeError(f"{out_dir} exists: a run directory is never reused")
    t0 = time.time()
    _run([sys.executable, "src/export.py", "--adapter", f"{VOLUME}/{adapter}", "--out", str(out_dir),
          "--quantization", quantization])
    volume.commit()
    files = {str(p.relative_to(out_dir)): p.stat().st_size for p in sorted(out_dir.rglob("*")) if p.is_file()}
    return {"name": name, "adapter": adapter, "seconds": time.time() - t0, "files": files,
            "ct2_model_sha256": hashlib.sha256((out_dir / "ct2" / "model.bin").read_bytes()).hexdigest()}


# --------------------------------------------------------------------------- #
# local entry points
# --------------------------------------------------------------------------- #
def _commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                          capture_output=True, text=True).stdout.strip()


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _save(kind: str, name: str, out: dict) -> pathlib.Path:
    path = REPO / "results" / "ft2" / f"{kind}-{name}.json"          # results/ is scratch, ignored by git
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return path


def _cost_line(inside: float, wall: float) -> str:
    return (f"{inside:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
            f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


def _print_validations(rows: list[dict]) -> None:
    print(f"{'step':>6} {'loss':>7} {'WER raw':>8} {'WER norm':>9} {'fillers kept':>13} {'clips':>6}")
    for r in rows:
        print(f"{r['step']:>6} {r.get('val/loss', float('nan')):>7.3f} {100 * r['wer_raw']:>7.1f}% {100 * r['wer_normalized']:>8.1f}% "
              f"{100 * r['filler_recall']:>12.1f}% {int(r['n_clips']):>6}")


def _train(kind: str, cap: float, overrides: list[str]) -> None:
    commit = _commit()
    if commit.endswith("-dirty"):
        print("NOTE: uncommitted changes; the run is stamped -dirty")
    data = fetch_data.remote()
    print(f"data on the volume: {data['mb']} MB, revision {data['revision'][:7]}, {data['seconds']} s")
    name = f"{kind}-{commit}-{_stamp()}"
    t0 = time.time()
    out = train_remote.remote(name, commit, overrides, seconds_for(cap))
    wall = time.time() - t0
    out.update({"commit": commit, "kind": kind, "overrides": overrides, "wall_s": wall, "data": data})
    path = _save(kind, name, out)
    print(f"\n{out['machine']} · commit {commit} · run {name}")
    _print_validations(out["validations"])
    print(f"adapters: {len(out['adapters'])} under {out['run_dir']}  (modal volume get ft2 runs/{name} results/ft2/)")
    print(f"saved -> {path.relative_to(REPO)}")
    print(_cost_line(out["seconds"], wall))


@app.local_entrypoint()
def dev(overrides: str = ""):
    """The smoke run: 30 steps of the real experiment on the real data, validation on 8 clips twice."""
    _train("dev", DEV_CAP_DOLLARS, DEV_OVERRIDES + overrides.split())


@app.local_entrypoint()
def train(overrides: str = ""):
    """The experiment as configured (900 steps); `--overrides "a=b c=d"` for anything else, recorded."""
    _train("train", CAP_DOLLARS, overrides.split())


@app.local_entrypoint()
def evaluate(adapter: str = ""):
    """The dev look on all 250 validation clips: the base model (no --adapter), or an adapter on the volume,
    given as its path under /ft2, e.g. runs/train-<commit>-<stamp>/adapters/step-000900."""
    commit = _commit()
    name = f"eval-{'base' if not adapter else adapter.replace('/', '_')}-{_stamp()}"
    t0 = time.time()
    out = evaluate_remote.remote(name, adapter or None)
    wall = time.time() - t0
    out.update({"commit": commit, "wall_s": wall})
    path = _save("eval", name, out)
    print(f"\n{out['machine']} · commit {commit} · {'base model' if not adapter else adapter}")
    print(out["data"])
    _print_validations(out["validations"])
    print(f"saved -> {path.relative_to(REPO)}")
    print(_cost_line(out["seconds"], wall))


@app.local_entrypoint()
def export(adapter: str, quantization: str = "float16"):
    """Merge the adapter into whisper-small.en and convert to CTranslate2, on a CPU; the result stays on the
    volume under export/<name>, where the bench (next commit) reads it."""
    commit = _commit()
    name = f"export-{adapter.replace('/', '_')}-{quantization}"
    t0 = time.time()
    out = export_remote.remote(name, adapter, quantization)
    wall = time.time() - t0
    out.update({"commit": commit, "wall_s": wall})
    path = _save("export", name, out)
    print(f"\nexport/{name}: {len(out['files'])} files; ct2/model.bin sha256 {out['ct2_model_sha256'][:12]}")
    print(f"saved -> {path.relative_to(REPO)}")
    print(f"{out['seconds']:.0f} s on a CPU container, {wall:.0f} s end to end (CPU rate, not in the A10 estimate).")
