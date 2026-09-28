"""
The demo on Modal: the page and the measured part on a CPU, the live turn on a T4 that sleeps,
and R3 and R4 on an A10 and an L40S that wake only when a visitor asks for them.

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
           only by a request that passed the door; asleep 300 s after the last one
  hears    an A10: R3 (Qwen2-Audio hears the recording; 17 GB). Woken only when R3 is asked for
  speaks   an L40S: R4 (Qwen2.5-Omni speaks the reply itself; 22.5 GB, more than an A10 holds).
           Woken only when R4 is asked for. Both asleep 120 s after their last turn
  volume   `ft2`, read-only: PR-011's export, for the rung r2-gpu-verbatim
  dict     `s2s-lab-demo-spend`: the day's spend and the last measured wake, kept across restarts

WHY TWO CONTAINERS. One web app on the T4 would wake the GPU for every visit to the page, and for
every crawler: a cost nobody decided on, outside the access code and outside the cap. Here a
request reaches the GPU only with the code and inside the budget.

THE WINDOW IS 300 S, NOT THE 60 S OF EVERY BATCH RUN HERE (v2 5.1, rule 3). A batch run's next call
comes at once; a visitor's comes after listening and recording. What it costs: a burst of use
ends with 300 s of an idle T4, about five cents, and the cap counts them.

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
WINDOW_S = 300                      # the T4 sleeps this long after its last turn: a visitor listens to four replies and
                                    # records again in less; 60 s, the batch runs' window, made every turn a wake
A10_DOLLARS_PER_HOUR, L40S_DOLLARS_PER_HOUR = 1.10, 1.95      # checked 2026-08-19 (v2 §5)
WINDOW_BIG_S = 120                  # the A10 and the L40S: their idle time costs two and three times the T4's
SECRET = "s2s-lab-demo"             # holds DEMO_ACCESS_CODE
SPEND = "s2s-lab-demo-spend"
VOLUME = "/ft2"
EXPORT_NAME = "export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16"   # == s2slab.r2verbatim.EXPORT_NAME
EXPORT = f"{VOLUME}/export/{EXPORT_NAME}/ct2"

# What a visitor sees -> the registered rung. R2 and R2 verbatim differ in the recognizer's LoRA alone.
OFFERED = {"R0": "r0-gpu", "R1": "r1-gpu", "R2": "r2-gpu", "R2 verbatim": "r2-gpu-verbatim"}
ABOUT = {"R2": "slow tier small.en here; the measured R2 row is large-v3-turbo",
         "R2 verbatim": "R2 with FT-2's LoRA on its recognizer (PR-011): one speaker's disfluency, not a visitor's",
         "R3": "speech-LLM: Qwen2-Audio hears the recording itself; no transcript reaches it",
         "R4": "speech-to-speech: Qwen2.5-Omni hears the recording and speaks the reply itself; no recognizer, no TTS"}
# The machines beside the T4, each woken only by a visitor who asks for its rung.
OTHERS = {"a10": {"offered": {"R3": "r3-gpu"}, "says": "an A10", "rate": A10_DOLLARS_PER_HOUR},
          "l40s": {"offered": {"R4": "r4-gpu"}, "says": "an L40S", "rate": L40S_DOLLARS_PER_HOUR}}

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


MODELS_A10 = {
    "Qwen/Qwen2-Audio-7B-Instruct": None,                                             # R3
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],   # its voice
}
MODELS_L40S = {"Qwen/Qwen2.5-Omni-7B": None}                                          # R4, about 22 GB


def _download(models: dict) -> None:
    from huggingface_hub import snapshot_download
    for repo, files in models.items():
        snapshot_download(repo, allow_patterns=files)


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    _download(MODELS)


def download_weights_a10() -> None:
    _download(MODELS_A10)


def download_weights_l40s() -> None:
    _download(MODELS_L40S)


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

def _speech_llm_image(download):
    """R3's and R4's libraries, as modal_app/r3_probe.py and r4_bench.py pin them."""
    return (
        modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
        .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
        .pip_install("transformers==5.17.0", "accelerate==1.15.0", "soundfile==0.14.0", "numpy==2.2.6", "huggingface_hub==1.31.0")
        .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
        .pip_install("scipy==1.15.3", "praat-parselmouth==0.4.7")
        .run_function(download)
        # Qwen2.5-Omni's processor imports Pillow and torchvision even for audio (r4_bench.py); harmless for R3
        .pip_install("torch==2.14.0", "torchvision==0.29.0", index_url="https://download.pytorch.org/whl/cu126")
        .pip_install("pillow==11.3.0")
        .apt_install("gcc")
        .env({"HF_HUB_OFFLINE": "1"})
        .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    )


a10_image = _speech_llm_image(download_weights_a10)
l40s_image = _speech_llm_image(download_weights_l40s)

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
        _load_rungs(self, OFFERED)

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        return _turns(self, labels, raw)


def _load_rungs(self, offered: dict) -> None:
    import os
    os.chdir("/root")
    from s2slab.demo import Live
    self.live = Live(offered)
    self.load_s = self.live.load_all()
    self.machine = f"Modal {_gpu_name()}"
    print(f"loaded {[r for r in offered if r not in self.live.unavailable]} in {self.load_s:.1f} s on {self.machine}; "
          f"not loaded: {list(self.live.unavailable)}", flush=True)


def _turns(self, labels: list, raw: bytes) -> dict:
    from s2slab.demo import answer
    out = answer(self.live, labels, raw)
    out.update(machine=self.machine, load_s=round(self.load_s, 1))
    return out


@app.cls(image=a10_image, gpu="A10G", max_containers=1, scaledown_window=WINDOW_BIG_S, timeout=5 * 60, startup_timeout=15 * 60)
class Hears:
    """R3 on an A10, loaded when the container starts."""

    @modal.enter()
    def load(self):
        _load_rungs(self, OTHERS["a10"]["offered"])

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        return _turns(self, labels, raw)


@app.cls(image=l40s_image, gpu="L40S", max_containers=1, scaledown_window=WINDOW_BIG_S, timeout=5 * 60, startup_timeout=15 * 60)
class Speaks:
    """R4 on an L40S, loaded when the container starts."""

    @modal.enter()
    def load(self):
        _load_rungs(self, OTHERS["l40s"]["offered"])

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        return _turns(self, labels, raw)


def make_app(call, store, root="/root", others=None):
    """The web app with the T4's rungs behind `call`, the other machines' behind `others` (by
    name, as in OTHERS) and the spend in `store`: what web() serves, and what the tests run with
    functions and a dict in their place."""
    from s2slab.demo import Budget, Remote, Several
    from s2slab.demoweb import create_app
    others = dict(others or {})
    budget = Budget(store, CAP_DOLLARS, T4_DOLLARS_PER_HOUR, window_s=WINDOW_S, cold_start_s=0.0,
                    machines={name: (OTHERS[name]["rate"], WINDOW_BIG_S) for name in others})
    live = Several(Remote(OFFERED, call), {name: Remote(OTHERS[name]["offered"], c) for name, c in others.items()},
                   says={name: OTHERS[name]["says"] for name in others})
    return create_app(live, budget, root=root, machine="Modal T4", about=ABOUT)


@app.function(image=web_image, secrets=[modal.Secret.from_name(SECRET, required_keys=["DEMO_ACCESS_CODE"])],
              max_containers=1, scaledown_window=WINDOW_S, timeout=10 * 60)
@modal.concurrent(max_inputs=16)                # the page is still served while a turn waits for the T4
@modal.asgi_app()
def web():
    return make_app(Rungs().turns.remote, modal.Dict.from_name(SPEND, create_if_missing=True),
                    others={"a10": Hears().turns.remote, "l40s": Speaks().turns.remote})
