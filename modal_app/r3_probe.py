"""
R3 on Modal: the probe (Qwen2-Audio hears the prosody pairs and is asked whether the caller is upset),
and the rung (R3 over the eval set on an A10, with R0 and R1 beside it in the same container).

    modal run modal_app/r3_probe.py --split dev                          21 dev pairs, every wording in WORDINGS
    modal run modal_app/r3_probe.py --split test --confirm PR-006 --wording upset --save runs/prosody/pr006.json
    modal run modal_app/r3_probe.py::own --save runs/prosody/ownvoice-r3.json     the own-voice mini set, dev-only
    modal run modal_app/r3_probe.py::evalset --rung r3-gpu --limit 8             smoke test -> results/
    modal run modal_app/r3_probe.py::evalset --rung r0-gpu,r1-gpu,r3-gpu         the R3 row: one A10 container,
                                                                                 saved to runs/<rung>/evalset-modal-a10.json

What happens where:
  laptop   verifies the frozen set, loads one side of it (the test side only
           with a registration), sends the clips as WAV bytes; afterwards
           scores the answers, prints the report, saves the run
  image    CUDA 12.6, Python 3.12, torch, transformers, R0's model libraries
           (Kokoro and its pronunciation stack, faster-whisper), and the weights:
           the 16 GB of Qwen2-Audio plus Kokoro, Whisper base.en and Qwen2.5-0.5B
           so R0 and R1 can run beside R3 -- all fetched on CPU at build time
  A10      the probe: loads the model once (bf16, 24 GB card), then one forward
           pass per clip per wording -- the next-token probabilities of "Yes"
           and "No" -- and one short generation per clip, its transcript.
           The rung: evalrun.run() per rung over the frozen eval set, one
           container, the models of one rung dropped before the next loads.

The T4 cannot hold the 7B model, which is why the rung's eval-set run lives
here and not in r0_bench.py; r0-gpu and r1-gpu run in the SAME A10 container
so the R3 row is read against them on that day's hardware (M13: the same rung
measured 17% slower three days apart, so rows are compared within a run).

The same guardrails as r0_bench: explicit timeout, no container kept warm,
scale down 60 s after the call, HF_HUB_OFFLINE=1 so a model missing from the
image fails instead of downloading on the GPU's meter, one container per run.
The model never sees a transcript: the clip and the question are all it gets.
"""

import io
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
A10_DOLLARS_PER_HOUR = 1.10     # Modal's A10 rate, checked 2026-08-19 (v2 §5)
MODELS = {
    "Qwen/Qwen2-Audio-7B-Instruct": None,                       # R3, probe and rung
    "Systran/faster-whisper-base.en": None,                     # R0 / R1 beside it
    "Qwen/Qwen2.5-0.5B-Instruct": None,
    "hexgrad/Kokoro-82M": ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
}
MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


def download_weights() -> None:
    """Runs once, inside the image build, on CPU."""
    from huggingface_hub import snapshot_download
    for repo, files in MODELS.items():
        snapshot_download(repo, allow_patterns=files)


image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.17.0", "accelerate==1.15.0", "soundfile==0.14.0", "numpy==2.2.6",
                 "huggingface_hub==1.31.0")
    # R0's model libraries (the same pins as r0_bench's image; transformers is the same 5.17.0),
    # so Kokoro speaks R3's replies and R0/R1 can run in the same container.
    .pip_install_from_requirements(str(REPO / "requirements" / "r0.txt"))
    .pip_install("scipy==1.15.3", "praat-parselmouth==0.4.7")
    .run_function(download_weights)
    # torch 2.14 runs small GPU ops through Triton kernels that compile a C helper
    # on first use; the CUDA runtime image has no compiler (r0_bench hit the same).
    # After the weights, so the 16 GB layer stays cached.
    .apt_install("gcc")
    .env({"HF_HUB_OFFLINE": "1"})
    # The code and the frozen eval set, added last so editing code never rebuilds the image.
    .add_local_dir(REPO / "src" / "s2slab", "/root/s2slab", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "evalset", "/root/evalset")
)

app = modal.App("s2s-lab-r3", image=image)


class SpeechLLM:
    """Qwen2-Audio behind three calls: load(), p_yes(audio, sr, system, question), transcribe(audio, sr, prompt).

    p_yes() runs ONE forward pass with the clip and the question and reads the
    next-token logits for "Yes" and "No":
        p_yes = P(Yes) / (P(Yes) + P(No)),   mass = P(Yes) + P(No)
    -- the same read as sidechannel.YesNoScorer, so R3 and R1 share a scale.
    """

    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.processor = self.model = None

    def load(self) -> "SpeechLLM":
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, dtype=torch.bfloat16).to("cuda").eval()
        tok = self.processor.tokenizer
        ids = [tok.encode(w, add_special_tokens=False) for w in ("Yes", "No")]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"'Yes' and 'No' must be single tokens, got {ids}")
        self.yes, self.no = ids[0][0], ids[1][0]
        self.sr = self.processor.feature_extractor.sampling_rate
        return self

    def _inputs(self, audio, sr: int, system: str, text: str):
        if sr != self.sr:
            raise ValueError(f"the model expects {self.sr} Hz audio, got {sr}")
        conversation = [{"role": "system", "content": system},
                        {"role": "user", "content": [{"type": "audio", "audio_url": "clip.wav"},
                                                     {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        try:
            inputs = self.processor(text=prompt, audio=[audio], sampling_rate=sr, return_tensors="pt")
        except TypeError:                                             # older processors spell it "audios"
            inputs = self.processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt")
        return inputs.to("cuda")

    def p_yes(self, audio, sr: int, system: str, question: str) -> tuple[float, float]:
        import torch
        inputs = self._inputs(audio, sr, system, question)
        with torch.inference_mode():
            p = torch.softmax(self.model(**inputs).logits[0, -1].float(), dim=-1)
        yes, no = float(p[self.yes]), float(p[self.no])
        return yes / (yes + no), yes + no

    def transcribe(self, audio, sr: int, prompt: str, max_new_tokens: int = 60) -> str:
        import torch
        inputs = self._inputs(audio, sr, "You are a helpful assistant.", prompt)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def run_probe(llm, clips: dict[str, bytes], system: str, questions: dict[str, str],
              transcribe_prompt: str | None) -> dict[str, dict]:
    """Ask every question of every clip. Testable with a fake llm.

    Takes:   a LOADED llm (p_yes, transcribe); {clip name: WAV bytes}; the
             system line; {wording name: question}; a transcription prompt or None.
    Returns: {clip: {"p_yes": {wording: p}, "mass": {wording: m}, "transcript": str}}
    """
    import numpy as np
    import soundfile as sf
    out = {}
    for name, data in clips.items():
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        answers = {q: llm.p_yes(np.asarray(audio, dtype=np.float32), sr, system, question)
                   for q, question in questions.items()}
        out[name] = {"p_yes": {q: a[0] for q, a in answers.items()},
                     "mass": {q: a[1] for q, a in answers.items()},
                     "transcript": llm.transcribe(audio, sr, transcribe_prompt) if transcribe_prompt else ""}
    return out


def _gpu_name() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@app.function(gpu="A10G", timeout=30 * 60, scaledown_window=60)
def probe(clips: dict[str, bytes], system: str, questions: dict[str, str], transcribe_prompt: str | None) -> dict:
    """Load the model once and run run_probe() on the A10."""
    t0 = time.perf_counter()
    llm = SpeechLLM().load()
    load_s = time.perf_counter() - t0
    answers = run_probe(llm, clips, system, questions, transcribe_prompt)
    return {"gpu": _gpu_name(), "model": MODEL, "load_s": load_s, "answers": answers,
            "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# The rung: R3 (and R0, R1 beside it) over the frozen eval set, on the A10
# --------------------------------------------------------------------------- #


def _free_gpu() -> None:
    """Drop the last rung's models before loading the next, so the second rung
    does not run in whatever memory the first left over."""
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


@app.function(gpu="A10G", timeout=45 * 60, scaledown_window=60)
def evaluate(rung_ids: list[str], commit: str, limit: int = 0) -> dict:
    """Run each rung in turn over the frozen eval set, on the A10 (r0_bench.evaluate, on the bigger card).

    Takes:   registered rung ids, run in the order given, one container; the
             laptop's commit; an optional limit for a smoke test (0 = all 200).
    Calls:   evalset.verify() first -- a WAV that differs from the manifest
             stops the run before any GPU time is spent; then evalrun.run()
             per rung, which also records what R3 hears when asked, untimed,
             and each rung's peak GPU memory.
    Returns: {"results": {rung id: the evalrun results dict}, "seconds": time inside}.
    """
    import os
    os.chdir("/root")                       # evalset.EVAL_ROOT is relative to here
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
    return {"results": results, "seconds": time.perf_counter() - t0}


def results_path(rung_id: str, limit: int = 0, tag: str = "") -> pathlib.Path:
    """runs/<rung>/evalset-modal-a10[-tag].json for the real run (committed evidence);
    results/<rung>/...-limitN.json for a smoke test (ignored, never mistaken for the row)."""
    suffix = f"-{tag}" if tag else ""
    if limit:
        return REPO / "results" / rung_id / f"evalset-modal-a10{suffix}-limit{limit}.json"
    return REPO / "runs" / rung_id / f"evalset-modal-a10{suffix}.json"


@app.local_entrypoint()
def evalset(rung: str = "r3-gpu", limit: int = 0, tag: str = ""):
    """The eval set through the named rungs, one A10 container; scored and saved on the laptop."""
    from s2slab.evalrun import report, save
    from s2slab.sidechannel import refuse_to_overwrite

    rung_ids = [r.strip() for r in rung.split(",") if r.strip()]
    if not limit:
        for r in rung_ids:
            refuse_to_overwrite(results_path(r))
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    out = evaluate.remote(rung_ids, commit, limit)
    wall = time.time() - t0
    for rung_id, results in out["results"].items():
        path = save(results, results_path(rung_id, limit, tag))
        print(report(results, root=REPO / "evalset"))
        print(f"\nsaved -> {path.relative_to(REPO)}\n")
    print(f"{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def main(split: str = "dev", confirm: str = "", wording: str = "", save: str = ""):
    import json

    from s2slab.r3probe import (REGISTERED_WORDING, SYSTEM, TRANSCRIBE, WORDINGS, clips_for, compare,
                                format_criteria, format_report, per_pair, sidechannel_outcomes)
    from s2slab.r3probe import save as save_run
    from s2slab.sidechannel import refuse_to_overwrite

    if save:
        refuse_to_overwrite(pathlib.Path(save))
    wordings = WORDINGS if not wording else {wording: WORDINGS[wording]}
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    pairs, clips = clips_for(split, confirm)
    t0 = time.time()
    out = probe.remote(clips, SYSTEM, wordings, TRANSCRIBE)
    wall = time.time() - t0
    head = (f"{len(pairs)} {split} pairs" + (f" · confirms {confirm}" if confirm else "")
            + f" · {out['model']} on {out['gpu']} · commit {commit} · load {out['load_s']:.0f} s")
    print(head, "\n")
    print(format_report(pairs, out["answers"], wordings))
    if split == "test" and REGISTERED_WORDING in wordings:
        p_yes = {c: a["p_yes"][REGISTERED_WORDING] for c, a in out["answers"].items()}
        transcripts = {c: a["transcript"] for c, a in out["answers"].items()}
        side = sidechannel_outcomes(json.loads((REPO / "runs" / "prosody" / "pr005.json").read_text(encoding="utf-8")))
        print("\n" + format_criteria(compare(pairs, p_yes, side), confirm))
        print("\n" + per_pair(pairs, p_yes, side, transcripts))
    if save:
        save_run(pathlib.Path(save), head, pairs, out["answers"], wordings)
        print(f"\nsaved to {save}")
    print(f"\n{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")


@app.local_entrypoint()
def own(save: str = ""):
    """The own-voice mini set (s2slab.ownvoice, dev-only): every take windowed
    and as recorded, every wording; the registered wording's per-pair table first."""
    from s2slab.ownvoice import clips, split_answers
    from s2slab.r3probe import REGISTERED_WORDING, SYSTEM, TRANSCRIBE, WORDINGS, format_report
    from s2slab.r3probe import save as save_run
    from s2slab.sidechannel import refuse_to_overwrite

    if save:
        refuse_to_overwrite(pathlib.Path(save))
    wordings = {REGISTERED_WORDING: WORDINGS[REGISTERED_WORDING],
                **{k: v for k, v in WORDINGS.items() if k != REGISTERED_WORDING}}
    commit = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"],
                            capture_output=True, text=True).stdout.strip()
    pairs, data = clips()
    t0 = time.time()
    out = probe.remote(data, SYSTEM, wordings, TRANSCRIBE)
    wall = time.time() - t0
    head = (f"{len(pairs)} own-voice pairs, each take windowed and as recorded · {out['model']} on {out['gpu']}"
            f" · commit {commit} · load {out['load_s']:.0f} s")
    print(head)
    for label, answers in zip(("windowed", "as recorded"), split_answers(out["answers"])):
        print(f"\n{label}:\n" + format_report(pairs, answers, wordings))
    if save:
        save_run(pathlib.Path(save), head, pairs, out["answers"], wordings)
        print(f"\nsaved to {save}")
    print(f"\n{out['seconds']:.0f} s inside the function, {wall:.0f} s end to end (queue, cold start and run): "
          f"about ${wall / 3600 * A10_DOLLARS_PER_HOUR:.3f} at the A10 rate. Modal's dashboard has the exact bill.")
