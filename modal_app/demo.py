"""
The demo on Modal: the page and the measured part on a CPU, the live turn on a T4 that sleeps,
and R3, R4 and R5 on machines of their own that wake only when a visitor asks for them.

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
           only by a request that passed the door; asleep 300 s after the last one. It also
           renders a severity score onto Kokoro's speech (PR-009's voices, sentences and scores)
  hears    an A10: R3 (Qwen2-Audio hears the recording; 17 GB). Woken only when R3 is asked for
  speaks   an L40S: R4 and R4 streamed (Qwen2.5-Omni speaks the reply itself; 22.5 GB, more than an
           A10 holds), both on one set of its weights (r4stream.pair); R4 streamed is r4-gpu-stream-timed
           (M37, PR-017: M34's sized stream, its start timed from the turn), its caches prepared at load.
           Woken only when one is asked for
  talks    a second A10: R5 (Moshi, full duplex; 16.6 GB, so not beside R3's 17). The recording
           is one side of a call, streamed as the eval clips were. Woken only when R5 is asked
           for. All three asleep 120 s after their last turn
  volume   `ft2`, read-only: PR-011's export, for the rung r2-gpu-verbatim
  volume   `s2s-lab-compile-cache`: what the L40S compiled, loaded before its next load so that a wake
           after the first finds most compiles done (M36, s2slab.compilecache)
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
CAP_DOLLARS = 3.0                   # a day (1.0 until M32a, the author's choice)
WINDOW_S = 300                      # the T4 sleeps this long after its last turn: a visitor listens to four replies and
                                    # records again in less; 60 s, the batch runs' window, made every turn a wake
A10_DOLLARS_PER_HOUR, L40S_DOLLARS_PER_HOUR = 1.10, 1.95      # checked 2026-08-19 (v2 §5)
WINDOW_BIG_S = 120                  # the A10 and the L40S: their idle time costs two and three times the T4's
SECRET = "s2s-lab-demo"             # holds DEMO_ACCESS_CODE
SPEND = "s2s-lab-demo-spend"
VOLUME = "/ft2"
COMPILE_CACHE = "/compile-cache"      # the volume s2s-lab-compile-cache, on the L40S (M36)
COMPILE_FILE = f"{COMPILE_CACHE}/l40s-artifacts.bin"
EXPORT_NAME = "export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16"   # == s2slab.r2verbatim.EXPORT_NAME
EXPORT = f"{VOLUME}/export/{EXPORT_NAME}/ct2"

# What a visitor sees -> the registered rung. R2 and R2 verbatim differ in the recognizer's LoRA alone.
OFFERED = {"R0": "r0-gpu", "R1": "r1-gpu-trim", "R2": "r2-gpu", "R2 verbatim": "r2-gpu-verbatim"}
ABOUT = {"R1": "its first clause starts without the 0.3 s of silence Kokoro puts before every clause: on the eval set "
               "the caller heard R1's reply after 0.50 s, against 0.80 with it (PR-018). R0, R2 and R3 keep it: PR-018 "
               "tested R1 alone",
         "R2": "slow tier small.en here; the measured R2 row is large-v3-turbo",
         "R2 verbatim": "R2 with FT-2's LoRA on its recognizer (PR-011): one speaker's disfluency, not a visitor's",
         "R3": "speech-LLM: Qwen2-Audio hears the recording itself; no transcript reaches it. Its reply may describe you, how "
               "you sound included: it is not asked to, and nothing here checks what it says",
         "R4": "speech-to-speech: Qwen2.5-Omni hears the recording and speaks the reply itself; no recognizer, no TTS. Its "
               "Talker is compiled here, which on the eval set took 1.03 of R4's time (PR-015)",
         "R4 streamed": "the same model and the same reply, decoded a block at a time and handed out, its Talker on a cache "
                        "sized to the turn, its playback started when the turn's own pace says the next block is due: on the "
                        "eval set the caller heard it after 2.3 s, against 6.8 for R4 (PR-015, PR-016, PR-017). This page "
                        "plays each block as it arrives, times the first in your browser, and says if playback ran dry there",
         "R5": "full duplex: Moshi, no turns. Your recording is the caller's side of one call"}
# The machines beside the T4, each woken only by a visitor who asks for its rung.
OTHERS = {"a10": {"offered": {"R3": "r3-gpu"}, "says": "an A10", "rate": A10_DOLLARS_PER_HOUR},
          "l40s": {"offered": {"R4": "r4-gpu-fast", "R4 streamed": "r4-gpu-stream-timed"}, "says": "an L40S",
                   "rate": L40S_DOLLARS_PER_HOUR},
          "moshi": {"offered": {"R5": "r5-moshi"}, "says": "a second A10", "rate": A10_DOLLARS_PER_HOUR}}

# What the web container is sent, and nothing else of runs/ and evalset/. tests/test_modal_demo.py
# holds both to s2slab.demo.
RUN_FILES = ("runs/r0-gpu/evalset-modal-t4-pr002.json", "runs/r1-gpu-trim/pr018.json",
             "runs/r2-gpu-turbo/evalset-modal-t4-pr004.json", "runs/r3-gpu/evalset-modal-a10.json",
             "runs/r4-gpu/evalset-modal-l40s.json", "runs/r4-gpu/pr017.json", "runs/r5-moshi/evalset-modal-a10.json",
             "runs/r5-personaplex/evalset-modal-a10.json", "runs/ft2/pr011.json", "runs/prosody/pr006.json",
             "runs/bargein/pr003-70ms.json", "runs/ft1/pr010.json", "runs/severity/pr009.json", "runs/ft3/pr012.json",
             "runs/m26/pr013.json", "runs/turn-taking/pr014-dev-vap.json",
             "runs/r4-gpu/pr015.json", "runs/r4-gpu/pr016.json")
CLIPS = ("clean-07", "clean-11", "clean-23", "clean-37", "disfluent-29", "disfluent-42")

# Every Hugging Face repo the four rungs load, and which files (None = all). The verbatim rung's
# recognizer is not here: it is on the volume.
MODELS = {
    "Systran/faster-whisper-base.en": None,
    "Systran/faster-whisper-small.en": None,
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    # af_heart speaks the rungs' replies; af_sarah and am_adam are PR-009's voices, for the severity render
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/af_sarah.pt", "voices/am_adam.pt"],
}


MODELS_A10 = {
    "Qwen/Qwen2-Audio-7B-Instruct": None,                                             # R3
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],   # its voice
}
MODELS_L40S = {"Qwen/Qwen2.5-Omni-7B": None}                                          # R4, about 22 GB
MODELS_MOSHI = {"kyutai/moshiko-pytorch-bf16": None}                                  # R5; == s2slab.r5.MOSHI_REPO


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


def download_weights_moshi() -> None:
    _download(MODELS_MOSHI)


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

moshi_image = (                     # as modal_app/r5_bench.py pins it: moshi 0.2.13 needs torch below 2.10
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("moshi==0.2.13", "numpy==2.2.6", "scipy==1.15.3", "soundfile==0.14.0")
    .run_function(download_weights_moshi)
    .pip_install("praat-parselmouth==0.4.7")
    .apt_install("gcc", "libportaudio2")
    .env({"HF_HUB_OFFLINE": "1"})
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
compile_cache = modal.Volume.from_name("s2s-lab-compile-cache", create_if_missing=True)


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

    @modal.method()
    def render(self, voice: str, sentence: str, condition: str, score: float) -> dict:
        """The severity render: Kokoro on the T4, the edit in Praat on its CPU."""
        return self.live.render(voice, sentence, condition, score)


def _load_rungs(self, offered: dict, shared_r4: bool = False, before: dict | None = None) -> None:
    import os
    os.chdir("/root")
    from s2slab.demo import Live, any_rung, kokoro_speaker
    if shared_r4:                                   # R4 and R4 streamed: one set of weights for the two (r4stream.pair)
        from s2slab.r4stream import pair
    self.live = Live(offered, get=pair() if shared_r4 else any_rung, speaker=lambda: kokoro_speaker("cuda"))
    self.load_s = self.live.load_all(before)
    self.machine = f"Modal {_gpu_name()}"
    print(f"loaded {[r for r in offered if r not in self.live.unavailable]} in {self.load_s:.1f} s on {self.machine}; "
          f"not loaded: {list(self.live.unavailable)}", flush=True)
    print("\n".join(self.live.load_report()), flush=True)


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


@app.cls(image=l40s_image, gpu="L40S", volumes={COMPILE_CACHE: compile_cache}, max_containers=1,
         scaledown_window=WINDOW_BIG_S, timeout=5 * 60, startup_timeout=15 * 60)
class Speaks:
    """R4 and R4 streamed on an L40S, one set of weights, loaded and compiled when the container starts;
    what it compiled kept on a volume for the next start (M36)."""

    @modal.enter()
    def load(self):
        from s2slab import compilecache
        box = {"loaded": {"loaded": False, "bytes": 0}}

        def load_cache():           # after R4 has loaded, before R4 streamed: R4 compiles as it did before M36
            box["loaded"] = compilecache.load(COMPILE_FILE)          # (its load took 288 s, not 57, with the cache first)

        _load_rungs(self, OTHERS["l40s"]["offered"], shared_r4=True, before={"R4 streamed": load_cache})
        loaded = box["loaded"]
        saved = compilecache.save(COMPILE_FILE, loaded)
        if saved["saved"]:
            compile_cache.commit()
        print(f"compile cache: loaded {loaded}; saved {saved}", flush=True)

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        return _turns(self, labels, raw)

    @modal.method()
    def stream(self, labels: list, raw: bytes):
        """The same, as events, each piece of R4 streamed yielded the moment it exists (demo.Live.stream)."""
        from s2slab.demo import answer_stream
        for ev in answer_stream(self.live, labels, raw):
            if "cold" in ev and "label" not in ev:
                ev = dict(ev, machine=self.machine, load_s=round(self.load_s, 1))
            yield ev


@app.cls(image=moshi_image, gpu="A10G", max_containers=1, scaledown_window=WINDOW_BIG_S, timeout=5 * 60, startup_timeout=15 * 60)
class Talks:
    """R5 (Moshi) on an A10 of its own, loaded when the container starts."""

    @modal.enter()
    def load(self):
        _load_rungs(self, OTHERS["moshi"]["offered"])

    @modal.method()
    def turns(self, labels: list, raw: bytes) -> dict:
        return _turns(self, labels, raw)


def make_app(call, store, root="/root", others=None, render=None, streams=None):
    """The web app with the T4's rungs behind `call`, the other machines' behind `others` (by
    name, as in OTHERS), a machine's stream behind `streams` (by name) and the spend in `store`: what
    web() serves, and what the tests run with functions and a dict in their place."""
    from s2slab.demo import Budget, Remote, Several
    from s2slab.demoweb import create_app
    others = dict(others or {})
    budget = Budget(store, CAP_DOLLARS, T4_DOLLARS_PER_HOUR, window_s=WINDOW_S, cold_start_s=0.0,
                    machines={name: (OTHERS[name]["rate"], WINDOW_BIG_S) for name in others})
    streams = dict(streams or {})
    live = Several(Remote(OFFERED, call, render=render),
                   {name: Remote(OTHERS[name]["offered"], c, stream=streams.get(name)) for name, c in others.items()},
                   says={name: OTHERS[name]["says"] for name in others})
    return create_app(live, budget, root=root, machine="Modal T4", about=ABOUT)


@app.function(image=web_image, secrets=[modal.Secret.from_name(SECRET, required_keys=["DEMO_ACCESS_CODE"])],
              cpu=1.0, max_containers=1, scaledown_window=WINDOW_S, timeout=10 * 60)   # a full core: the voice measurement does not hold the stream (M32a)
@modal.concurrent(max_inputs=16)                # the page is still served while a turn waits for the T4
@modal.asgi_app()
def web():
    return make_app(Rungs().turns.remote, modal.Dict.from_name(SPEND, create_if_missing=True),
                    others={"a10": Hears().turns.remote, "l40s": Speaks().turns.remote, "moshi": Talks().turns.remote},
                    render=Rungs().render.remote, streams={"l40s": Speaks().stream.remote_gen})
