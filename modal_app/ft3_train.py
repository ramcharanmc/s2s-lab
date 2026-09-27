"""
FT-3 on Modal: NeMo's own FastPitch and HiFi-GAN fine-tuning recipe, on one A10, in three stages.

    modal run modal_app/ft3_train.py::data        the 5-minute Hi-Fi TTS speaker (9017, the tutorial's
                                                  tarball), manifests split, pitch statistics -- CPU
    modal run modal_app/ft3_train.py::train       FastPitch: tts_en_fastpitch -> speaker 9017, 1,000 steps
    modal run modal_app/ft3_train.py::vocoder     mels from the fine-tuned FastPitch, then HiFi-GAN
                                                  fine-tuned on them, 1,000 steps
    mkdir -p data/ft3/models && for m in fastpitch hifigan; do \
        modal volume get ft3 models/${m}_9017.nemo data/ft3/models/; done
                                                  the two .nemo files, back to the laptop, ONE FILE A CALL into a
                                                  folder that exists (the folder form wrote both into one file):
                                                  the dev sweep and PR-012 run there on a CPU (ft/src/ft3/), $0

What happens where:
  image   nvidia/cuda 12.6 + torch 2.8 (cu126) + nemo-toolkit[tts] 3.0.0 (the pins of
          ft/requirements-ft3.txt, the CUDA build of the same torch), the Speech repo cloned at v3.0.0
          for the recipe's scripts and configs (they are not in the wheel), the two pretrained
          checkpoints (tts_en_fastpitch, tts_en_hifigan; LJSpeech; CC-BY-4.0) baked in on a CPU
  volume  `ft3`: data/9017 (audio, manifests, sup_data, pitch stats), runs/<stage>-<name>, models/
          (fastpitch_9017.nemo, hifigan_9017.nemo)
  A10     the recipe as the tutorial gives it (tutorials/tts/FastPitch_Finetuning.ipynb, quoted in
          docs/preregistration.md PR-012): fastpitch_finetune.py with fastpitch_align_v1.05.yaml, batch 24,
          Adam 2e-4, 1,000 steps, pitch stats from extract_sup_data.py; then mels with ground-truth
          alignment and hifigan_finetune.py, batch 32, lr 1e-5, 1,000 steps

Caps as in ft2_train.py: a dollar figure per function turned into its timeout; run directories never
reused; the recipe's own numbers, nothing tuned.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
A10_DOLLARS_PER_HOUR = 1.10     # Modal's A10 rate, checked 2026-08-19 (v2 §5)
TRAIN_CAP_DOLLARS = 1.00        # FastPitch, 1,000 steps: the tutorial says 50 min on a K80; an A10 is several times faster
VOCODER_CAP_DOLLARS = 1.00      # 60 mels, then HiFi-GAN 1,000 steps
NEMO = "nemo-toolkit[tts]==3.0.0"            # ft/requirements-ft3.txt's pin
TORCH = "torch==2.8.0"                        # the CUDA build of the laptop env's torch
SPEECH_TAG = "v3.0.0"                         # NVIDIA-NeMo/Speech: the scripts and configs, at the wheel's version
SPEECH_REPO = "https://github.com/NVIDIA-NeMo/Speech.git"
PRETRAINED = {"fastpitch": "tts_en_fastpitch", "hifigan": "tts_en_hifigan"}
DATA_URL = "https://multilangaudiosamples.s3.us-east-2.amazonaws.com/9017_5_mins.tar.gz"   # the tutorial's cut of Hi-Fi TTS 9017
SPEAKER = "9017"
SRC = "/root/speech"                          # the clone
VOLUME = "/ft3"
# The tutorial's pitch range for this speaker (librosa.pyin's fmin and fmax). The training LOADS the pitch
# files the data stage caches, so the data stage must extract with this range too: the first extract
# used the config's default, 65-2093 Hz, and came back with a maximum of 2022 Hz and a spread of 90.9 Hz
# against the tutorial's 64.0 -- octave errors, which the training would have read from the cache whatever
# model.pitch_fmax said. That extract stays on the volume (sup_data/, stats.json); nothing reads it.
PITCH_FMIN, PITCH_FMAX = 30, 512
PITCH_TAG = f"f{PITCH_FMIN}_{PITCH_FMAX}"
FIRST_EXTRACT = {"pitch_fmin": 65.40639132514966, "pitch_fmax": 2093.004522404789, "pitch_mean": 158.11416625976562,
                 "pitch_std": 90.88006591796875, "pitch_max": 2021.7091064453125}
# The tutorial's fine-tune settings, verbatim (FastPitch_Finetuning.ipynb at the pinned tag)
FASTPITCH_OVERRIDES = ["+trainer.max_steps=1000", "~trainer.max_epochs", "trainer.check_val_every_n_epoch=25",
                       "model.train_ds.dataloader_params.batch_size=24", "model.validation_ds.dataloader_params.batch_size=24",
                       "model.n_speakers=1", f"model.pitch_fmin={PITCH_FMIN}", f"model.pitch_fmax={PITCH_FMAX}", "model.optim.lr=2e-4",
                       "~model.optim.sched", "model.optim.name=adam", "trainer.devices=1", "trainer.strategy=auto",
                       "+model.text_tokenizer.add_blank_at=true"]
HIFIGAN_OVERRIDES = ["model.train_ds.dataloader_params.batch_size=32", "model.max_steps=1000", "model.optim.lr=0.00001",
                     "~model.optim.sched", "trainer.check_val_every_n_epoch=10", "model/train_ds=train_ds_finetune",
                     "model/validation_ds=val_ds_finetune"]


GET_MODELS = "mkdir -p data/ft3/models && for m in fastpitch hifigan; do modal volume get ft3 models/${m}_9017.nemo data/ft3/models/; done"


def seconds_for(dollars: float, rate: float = A10_DOLLARS_PER_HOUR) -> int:
    return int(dollars / rate * 3600)


def download_pretrained() -> None:
    """Runs once, inside the image build, on CPU: both checkpoints into NeMo's cache, and copied to
    /root/pretrained as .nemo files for init_from_nemo_model."""
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel
    FastPitchModel.from_pretrained(PRETRAINED["fastpitch"])
    HifiGanModel.from_pretrained(PRETRAINED["hifigan"])
    out = pathlib.Path("/root/pretrained")
    out.mkdir(exist_ok=True)
    for p in pathlib.Path("/root/.cache/torch/NeMo").rglob("*.nemo"):
        (out / p.name).write_bytes(p.read_bytes())
    print("pretrained:", sorted(p.name for p in out.iterdir()))


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.12")
    # clang, cmake, build-essential: two of NeMo's TTS extras (pyopenjtalk, cdifflib) have no wheel for
    # Python 3.12 and compile here, and Modal's Python calls clang -- the first build died on its absence
    .apt_install("git", "gcc", "g++", "clang", "cmake", "build-essential", "libsndfile1", "ffmpeg", "sox")
    .pip_install(TORCH, "torchaudio==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(NEMO, "praat-parselmouth==0.4.7", "soundfile==0.14.0")
    .run_commands(f"git clone --depth 1 --branch {SPEECH_TAG} {SPEECH_REPO} {SRC}")
    .run_function(download_pretrained)
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/root/ft:/root", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "ft" / "src", "/root/ft/src", ignore=["**/__pycache__"])
)

app = modal.App("s2s-lab-ft3", image=image)
volume = modal.Volume.from_name("ft3", create_if_missing=True)


def _run(cmd: list[str], cwd: str, log_path: pathlib.Path | None = None) -> str:
    """The command as a subprocess, its output streamed, kept, and returned."""
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if log:
                log.write(line)
    finally:
        if log:
            log.close()
    if proc.wait() != 0:
        raise RuntimeError(f"exit code {proc.returncode}: {' '.join(cmd)}")
    return "".join(lines)


def _machine() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:                   # a CPU container has no nvidia-smi: ::package died on it, after saving
        out = ""
    return f"Modal {out}" if out else "Modal (no GPU)"


def _fresh(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        raise RuntimeError(f"{path} exists: a run directory is never reused")
    path.mkdir(parents=True)
    return path


DATA = pathlib.Path(VOLUME) / "data" / SPEAKER
DICT = f"{SRC}/scripts/tts_dataset_files/cmudict-0.7b_nv22.10"
HETERONYMS = f"{SRC}/scripts/tts_dataset_files/heteronyms-052722"
SUP = DATA / f"sup_data_{PITCH_TAG}"          # the cached pitch files, one directory per pitch range
STATS = DATA / f"stats_{PITCH_TAG}.json"


def extract_cmd() -> list[str]:
    """extract_sup_data.py on the training manifest, with the training's pitch range."""
    return [sys.executable, f"{SRC}/scripts/dataset_processing/tts/extract_sup_data.py",
            f"manifest_filepath={DATA / 'train.json'}", f"sup_data_path={SUP}",
            f"phoneme_dict_path={DICT}", f"heteronyms_path={HETERONYMS}",
            f"dataset.pitch_fmin={PITCH_FMIN}", f"dataset.pitch_fmax={PITCH_FMAX}"]


@app.function(timeout=40 * 60, volumes={VOLUME: volume}, cpu=4, memory=8192)
def data_remote() -> dict:
    """The tutorial's data steps on a CPU: the tarball, the last two lines as validation, the rest as
    training (audio paths made absolute), then extract_sup_data.py for the pitch statistics."""
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    import tarfile
    import urllib.request
    if STATS.exists():
        return json.loads(STATS.read_text())
    if SUP.exists():
        raise RuntimeError(f"{SUP} exists without {STATS.name}: a half-finished extract, its cache is not reused")
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tgz = DATA / "9017_5_mins.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(DATA_URL, tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(DATA)
    src = DATA / "9017_5_mins"
    records = [json.loads(l) for l in (src / "manifest.json").read_text().splitlines() if l.strip()]
    for r in records:
        r["audio_filepath"] = str(src / r["audio_filepath"])
    train, val = records[:-2], records[-2:]                                 # the tutorial's split: the last two lines
    (DATA / "train.json").write_text("".join(json.dumps(r) + "\n" for r in train))
    (DATA / "val.json").write_text("".join(json.dumps(r) + "\n" for r in val))
    out = _run(extract_cmd(), cwd=SRC, log_path=DATA / f"extract_sup_data_{PITCH_TAG}.log")
    m = re.search(r"PITCH_MEAN=([\d.]+), PITCH_STD=([\d.]+)", out)
    lim = re.search(r"PITCH_MIN=([\d.]+), PITCH_MAX=([\d.]+)", out)
    if not m or not lim:
        raise RuntimeError("extract_sup_data.py did not print PITCH_MEAN/PITCH_STD and PITCH_MIN/PITCH_MAX")
    stats = {"speaker": SPEAKER, "clips_train": len(train), "clips_val": len(val),
             "seconds_train": round(sum(r["duration"] for r in train), 1), "pitch_mean": float(m.group(1)),
             "pitch_std": float(m.group(2)), "pitch_min": float(lim.group(1)), "pitch_max": float(lim.group(2)),
             "pitch_fmin": PITCH_FMIN, "pitch_fmax": PITCH_FMAX, "sup_data": SUP.name,
             "tutorial_stats": {"pitch_mean": 152.3, "pitch_std": 64.0}, "first_extract": FIRST_EXTRACT,
             "source": DATA_URL, "seconds": round(time.time() - t0)}
    STATS.write_text(json.dumps(stats, indent=1))
    volume.commit()
    return stats


HIFIGAN_MIN_S = {"train": 0.75, "val": 3.0}     # hifigan.yaml's train_min_duration and val_min_duration: shorter clips are pruned


def short_clips(splits: dict) -> dict:
    """How many of each split's clips hifigan.yaml would prune."""
    return {s: sum(r["duration"] < HIFIGAN_MIN_S[s] for r in records) for s, records in splits.items()}


def _last_ckpt(exp_dir: pathlib.Path, model_dir: str) -> pathlib.Path:
    """The run's last checkpoint. NeMo's exp_manager writes <exp_dir>/<model>/<date_time>/checkpoints/:
    the first training run finished its 1,000 steps and failed here, on a path without the date."""
    root = pathlib.Path(exp_dir) / model_dir
    cks = sorted(root.rglob("*last.ckpt"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not cks:
        found = sorted(str(p.relative_to(root)) for p in root.rglob("*.ckpt")) if root.exists() else []
        raise RuntimeError(f"no *last.ckpt under {root}; checkpoints there: {found}")
    return cks[-1]


def _package(run_dir: pathlib.Path, model_dir: str, nemo_name: str) -> pathlib.Path:
    """The run's last checkpoint as models/<nemo_name>; a model file is never overwritten."""
    from nemo.collections.tts import models
    out = pathlib.Path(VOLUME) / "models" / nemo_name
    if out.exists():
        raise RuntimeError(f"{out} exists: a model file is never overwritten")
    ckpt = _last_ckpt(run_dir, model_dir)
    model = getattr(models, MODEL_CLASSES[model_dir]).load_from_checkpoint(str(ckpt), map_location="cpu")
    out.parent.mkdir(exist_ok=True)
    model.save_to(str(out))
    return ckpt


MODEL_CLASSES = {"FastPitch": "FastPitchModel", "HifiGan": "HifiGanModel"}
NEMO_NAMES = {"FastPitch": "fastpitch_9017.nemo", "HifiGan": "hifigan_9017.nemo"}


@app.function(timeout=20 * 60, volumes={VOLUME: volume}, cpu=4, memory=8192)
def package_remote(name: str) -> dict:
    """A finished run on the volume, packaged on a CPU: for a run whose training completed and whose
    packaging did not."""
    run_dir = pathlib.Path(VOLUME) / "runs" / name
    found = [m for m in MODEL_CLASSES if (run_dir / m).exists()]
    if len(found) != 1:
        raise RuntimeError(f"{run_dir}: expected one of {sorted(MODEL_CLASSES)} in it, found {found}")
    t0 = time.time()
    ckpt = _package(run_dir, found[0], NEMO_NAMES[found[0]])
    volume.commit()
    return {"name": name, "machine": _machine(), "seconds": time.time() - t0, "checkpoint": str(ckpt),
            "nemo": f"models/{NEMO_NAMES[found[0]]}", "packaged_after_the_run": True}


@app.function(gpu="A10G", timeout=seconds_for(TRAIN_CAP_DOLLARS), scaledown_window=60, volumes={VOLUME: volume}, cpu=4)
def train_remote(name: str, commit: str) -> dict:
    """fastpitch_finetune.py as the tutorial runs it, then the last checkpoint saved as models/fastpitch_9017.nemo."""
    stats = json.loads(STATS.read_text())
    run_dir = _fresh(pathlib.Path(VOLUME) / "runs" / name)
    t0 = time.time()
    cmd = [sys.executable, f"{SRC}/examples/tts/fastpitch_finetune.py", "--config-name=fastpitch_align_v1.05.yaml",
           f"train_dataset={DATA / 'train.json'}", f"validation_datasets={DATA / 'val.json'}",
           f"sup_data_path={SUP}", f"phoneme_dict_path={DICT}", f"heteronyms_path={HETERONYMS}",
           f"exp_manager.exp_dir={run_dir}", "+init_from_nemo_model=/root/pretrained/tts_en_fastpitch_align.nemo",
           f"model.pitch_mean={stats['pitch_mean']}", f"model.pitch_std={stats['pitch_std']}", *FASTPITCH_OVERRIDES]
    _run(cmd, cwd=f"{SRC}/examples/tts", log_path=run_dir / "console.log")
    volume.commit()                                                          # the checkpoints first: packaging can fail
    ckpt = _package(run_dir, "FastPitch", NEMO_NAMES["FastPitch"])
    volume.commit()
    return {"name": name, "commit": commit, "machine": _machine(), "seconds": time.time() - t0, "checkpoint": str(ckpt),
            "nemo": "models/fastpitch_9017.nemo",
            "pitch_stats": {k: stats[k] for k in ("pitch_mean", "pitch_std", "pitch_min", "pitch_max", "pitch_fmin", "pitch_fmax")},
            "overrides": FASTPITCH_OVERRIDES}


@app.function(gpu="A10G", timeout=seconds_for(VOCODER_CAP_DOLLARS), scaledown_window=60, volumes={VOLUME: volume}, cpu=4)
def vocoder_remote(name: str, commit: str) -> dict:
    """The tutorial's HiFi-GAN step: mels of every training and validation clip from the fine-tuned
    FastPitch with ground-truth alignment, then hifigan_finetune.py from tts_en_hifigan."""
    import torch
    from nemo.collections.tts.models import FastPitchModel
    from src.ft3 import mels
    splits = {s: [json.loads(l) for l in (DATA / f"{s}.json").read_text().splitlines() if l.strip()] for s in ("train", "val")}
    short = short_clips(splits)
    if short["val"] == len(splits["val"]) or short["train"] == len(splits["train"]):
        raise RuntimeError(f"hifigan.yaml would prune a whole split before training: {short} of "
                           f"{ {s: len(r) for s, r in splits.items()} } clips are under its minimum durations {HIFIGAN_MIN_S}")
    run_dir = _fresh(pathlib.Path(VOLUME) / "runs" / name)
    t0 = time.time()
    spec_model = FastPitchModel.restore_from(f"{VOLUME}/models/fastpitch_9017.nemo").cuda().eval()
    manifests = {}
    for split, records in splits.items():
        records = mels.write(spec_model, records, run_dir / f"mels_{split}")
        manifests[split] = run_dir / f"hifigan_{split}.json"
        manifests[split].write_text("".join(json.dumps(r) + "\n" for r in records))
    volume.commit()                                                          # the mels, before the training starts
    del spec_model
    torch.cuda.empty_cache()
    cmd = [sys.executable, f"{SRC}/examples/tts/hifigan_finetune.py", "--config-name=hifigan.yaml",
           f"train_dataset={manifests['train']}", f"validation_datasets={manifests['val']}",
           f"exp_manager.exp_dir={run_dir}", "+init_from_nemo_model=/root/pretrained/tts_hifigan.nemo", *HIFIGAN_OVERRIDES]
    _run(cmd, cwd=f"{SRC}/examples/tts", log_path=run_dir / "console.log")
    volume.commit()
    ckpt = _package(run_dir, "HifiGan", NEMO_NAMES["HifiGan"])
    volume.commit()
    return {"name": name, "commit": commit, "machine": _machine(), "seconds": time.time() - t0, "checkpoint": str(ckpt),
            "nemo": "models/hifigan_9017.nemo", "overrides": HIFIGAN_OVERRIDES, "pruned_by_duration": short}


# --------------------------------------------------------------------------- #
# local entry points
# --------------------------------------------------------------------------- #
def _commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _save(kind: str, out: dict) -> pathlib.Path:
    path = REPO / "results" / "ft3" / f"modal-{kind}-{_stamp()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return path


def _cost_line(inside: float, wall: float) -> str:
    return (f"{inside:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
            f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def data():
    """Stage 1, CPU: the speaker's five minutes onto the volume, with the pitch statistics."""
    t0 = time.time()
    out = data_remote.remote()
    print(json.dumps(out, indent=1))
    print(f"saved -> {_save('data', out).relative_to(REPO)}; {time.time() - t0:.0f} s end to end, CPU rate")


@app.local_entrypoint()
def train():
    """Stage 2, A10: FastPitch fine-tuned to speaker 9017."""
    commit = _commit()
    name = f"fastpitch-{commit}-{_stamp()}"
    t0 = time.time()
    out = train_remote.remote(name, commit)
    wall = time.time() - t0
    out["wall_s"] = wall
    print(f"\n{out['machine']} · commit {commit} · {out['nemo']} from {out['checkpoint']}")
    print(f"saved -> {_save('train', out).relative_to(REPO)}")
    print(_cost_line(out["seconds"], wall))


@app.local_entrypoint()
def package(name: str):
    """A finished run's last checkpoint as its .nemo file, on a CPU:  ::package --name fastpitch-<commit>-<stamp>"""
    t0 = time.time()
    out = package_remote.remote(name)
    out["commit"] = _commit()
    print(f"\n{out['nemo']} from {out['checkpoint']}")
    print(f"saved -> {_save('package', out).relative_to(REPO)}; {time.time() - t0:.0f} s end to end, CPU rate")


@app.local_entrypoint()
def vocoder():
    """Stage 3, A10: HiFi-GAN fine-tuned on the fine-tuned FastPitch's mels."""
    commit = _commit()
    name = f"hifigan-{commit}-{_stamp()}"
    t0 = time.time()
    out = vocoder_remote.remote(name, commit)
    wall = time.time() - t0
    out["wall_s"] = wall
    print(f"\n{out['machine']} · commit {commit} · {out['nemo']} from {out['checkpoint']}")
    print(f"saved -> {_save('vocoder', out).relative_to(REPO)}")
    print(_cost_line(out["seconds"], wall))
    print("bring both voices back:  " + GET_MODELS)
