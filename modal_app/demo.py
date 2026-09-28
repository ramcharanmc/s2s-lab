"""
The demo on Modal: the page and the measured part on a CPU, the live turn on a T4 that sleeps.

    modal serve modal_app/demo.py       a temporary URL, gone at Ctrl+C: the first run, and every try
    modal deploy modal_app/demo.py      the URL that stays: the author's call

Before either, once, by the author (the code is typed unseen and never written down here; ten
characters or more, or the live part stays closed; --force replaces a code already there):

    read -rs CODE && modal secret create s2s-lab-demo DEMO_ACCESS_CODE="$CODE" && unset CODE

What happens where:
  laptop   sends src/s2slab, the run files the measured part reads, the manifest and six clips
  web      a CPU container: the page, the measured data, the clips, the status, and the door of the
           live part: the access code, the day's budget, the decoding, and the measures of the
           visitor's voice (Praat). It holds no model
  rungs    a T4: the four rungs, loaded when the container starts, one turn through each. Woken
           only by a request that passed the door; asleep 60 s after the last one
  volume   `ft2`, read-only: PR-011's export, for the rung r2-gpu-verbatim
  dict     `s2s-lab-demo-spend`: the day's spend and the last measured wake, kept across restarts

WHY TWO CONTAINERS. One web app on the T4 would wake the GPU for every visit to the page, and for
every crawler: a cost nobody decided on, outside the access code and outside the cap. Here a
request reaches the GPU only with the code and inside the budget.

Cost guardrails (v1 5.1): nothing kept warm; one container of each at most; a timeout on both; the
weights fetched in the image build on a CPU; HF_HUB_OFFLINE=1 at run time. The cap counts the T4
at its list rate for the time it is up (s2slab.demo.Budget): the wall time of a request that woke
it holds the wake, so no estimate of the cold start is added. Not counted: the CPUs and memory of
both containers, which Modal bills beside the GPU. Modal's dashboard is the bill.

A visitor's audio: decoded in the web container's memory, sent to the T4 as samples, answered, and
dropped. Nothing here writes it to a disk, a volume, the dict or a log.
"""

import pathlib
import subprocess

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
T4_DOLLARS_PER_HOUR = 0.59          # Modal's T4 rate, checked 2026-08-19 (v2 §5)
CAP_DOLLARS = 1.0                   # a day
WINDOW_S = 60                       # the T4 sleeps this long after its last turn
SECRET = "s2s-lab-demo"             # holds DEMO_ACCESS_CODE
SPEND = "s2s-lab-demo-spend"
VOLUME = "/ft2"
EXPORT_NAME = "export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16"   # == s2slab.r2verbatim.EXPORT_NAME
EXPORT = f"{VOLUME}/export/{EXPORT_NAME}/ct2"

# What a visitor sees -> the registered rung. R2 and R2 verbatim differ in the recognizer's LoRA alone.
OFFERED = {"R0": "r0-gpu", "R1": "r1-gpu", "R2": "r2-gpu", "R2 verbatim": "r2-gpu-verbatim"}
ABOUT = {"R2": "slow tier small.en here; the measured R2 row is large-v3-turbo",
         "R2 verbatim": "R2 with FT-2's LoRA on its recognizer (PR-011): one speaker's disfluency, not a visitor's"}

# What the web container is sent, and nothing else of runs/ and evalset/. tests/test_modal_demo.py
# holds both to s2slab.demo.
RUN_FILES = ("runs/r0-gpu/evalset-modal-t4-pr002.json", "runs/r1-gpu/evalset-modal-t4-pr002.json",
             "runs/r2-gpu-turbo/evalset-modal-t4-pr004.json", "runs/r3-gpu/evalset-modal-a10.json",
             "runs/r4-gpu/evalset-modal-l40s.json", "runs/r5-moshi/evalset-modal-a10.json",
             "runs/r5-personaplex/evalset-modal-a10.json", "runs/ft2/pr011.json", "runs/prosody/pr006.json")
CLIPS = ("clean-07", "clean-11", "clean-23", "clean-37", "disfluent-29", "disfluent-42")

# Every Hugging Face repo the four rungs load, and which files (None = all). The verbatim rung's
# recognizer is not here: it is on the volume.
MODELS = {
    "Systran/faster-whisper-base.en": None,
    "Systran/faster-whisper-small.en": None,
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
}


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")      # as modal_app/r0_bench.py
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "soundfile==0.14.0", "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    .apt_install("gcc")                                     # torch's Triton helpers, as in every GPU image here
    .env({"HF_HUB_OFFLINE": "1", "S2SLAB_FT2_CT2": EXPORT})
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REPO / "requirements" / "demo.txt"))
    .pip_install("numpy==2.2.6", "soundfile==0.14.0", "av==17.1.0")      # av decodes what a browser records
    .pip_install("praat-parselmouth==0.4.7")                             # the visitor's pitch, measured beside the door
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_file(REPO / "evalset" / "manifest.json", "/root/evalset/manifest.json")
)
for _rel in RUN_FILES:
    web_image = web_image.add_local_file(REPO / _rel, f"/root/{_rel}")
for _clip in CLIPS:
    web_image = web_image.add_local_file(REPO / "evalset" / "audio" / f"{_clip}.wav", f"/root/evalset/audio/{_clip}.wav")

app = modal.App("s2s-lab-demo")
ft2 = modal.Volume.from_name("ft2")


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.cls(image=gpu_image, gpu="T4", volumes={VOLUME: ft2.read_only()}, max_containers=1, scaledown_window=WINDOW_S,
         timeout=5 * 60, startup_timeout=10 * 60)
class Rungs:
    """The four rungs on one T4, loaded once when the container starts. A rung that does not load
    is answered for with a sentence; the others still answer."""

    @modal.enter()
    def load(self):
        import os
        os.chdir("/root")
        from s2slab.demo import Live
        self.live = Live(OFFERED)
        self.load_s = self.live.load_all()
        self.machine = f"Modal {_gpu_name()}"
        print(f"loaded {[r for r in OFFERED if r not in self.live.unavailable]} in {self.load_s:.1f} s on {self.machine}; "
              f"not loaded: {list(self.live.unavailable)}", flush=True)

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        from s2slab.demo import answer
        out = answer(self.live, labels, raw)
        out.update(machine=self.machine, load_s=round(self.load_s, 1))
        return out


def make_app(call, store, root="/root"):
    """The web app with the rungs behind `call` and the spend in `store`: what web() serves, and
    what the tests run with a function and a dict in their place."""
    from s2slab.demo import Budget, Remote
    from s2slab.demoweb import create_app
    budget = Budget(store, CAP_DOLLARS, T4_DOLLARS_PER_HOUR, window_s=WINDOW_S, cold_start_s=0.0)
    return create_app(Remote(OFFERED, call), budget, root=root, machine="Modal T4", about=ABOUT)


@app.function(image=web_image, secrets=[modal.Secret.from_name(SECRET, required_keys=["DEMO_ACCESS_CODE"])],
              max_containers=1, scaledown_window=WINDOW_S, timeout=10 * 60)
@modal.concurrent(max_inputs=16)                # the page is still served while a turn waits for the T4
@modal.asgi_app()
def web():
    return make_app(Rungs().turns.remote, modal.Dict.from_name(SPEND, create_if_missing=True))
