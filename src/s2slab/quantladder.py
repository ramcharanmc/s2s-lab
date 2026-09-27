"""
The quantization ladder with a prosody guardrail (v2 week 8; v1's P3): one TTS, Kokoro-82M, spoken
through smaller and smaller copies of itself, and each copy's speech measured against the fp32
copy's -- pitch level, pitch range, rate and pauses with severity's own measures, and
intelligibility by a recognizer's WER on the synthesized audio.

    PYTHONPATH=src python -m s2slab.quantladder dev [--listen]
        -> results/m26/dev.json and the table below it; laptop CPU, no GPU

The rungs. `torch` is the kokoro package the ladder's rungs R0 to R3 speak with. The three ONNX
files are onnx-community/Kokoro-82M-v1.0-ONNX's own, pinned by revision and SHA-256; nothing here
quantizes anything. The same phoneme ids and the same voice vector go into every rung (the ONNX
repo's vocabulary and voice files are the package's: checked on 2026-09-27).

What the pilot found, and what this module does about it (docs/m26-log.md):
  - ONNX Runtime on a CPU is deterministic; the torch package is not (the vocoder's noise), and
    two torch renders of one sentence differ by up to 9% in pitch range. So the NOISE FLOOR is
    torch against torch: TORCH_REPEATS renders of every sentence, each against the first.
  - A rung can return audio that is not a number. Such a render is counted, kept out of every
    measure, and never scored as speech.

The reference for every comparison is `onnx-fp32`'s render of the same sentence and voice.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

from . import severity, wer
from .prosody import pause_segments

SR = 24_000
ASR_SR = 16_000
REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
ROOT = "data/m26/kokoro-onnx"                  # ignored by git; docs/m26-log.md has the download commands
RUNGS = {   # rung -> (file under ROOT, SHA-256)
    "onnx-fp32": ("onnx/model.onnx", "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"),
    "onnx-fp16": ("onnx/model_fp16.onnx", "ba4527a874b42b21e35f468c10d326fdff3c7fc8cac1f85e9eb6c0dfc35c334a"),
    "onnx-q8": ("onnx/model_quantized.onnx", "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478"),
}
REFERENCE = "onnx-fp32"
TORCH = "torch"
TORCH_REPEATS = 5                              # renders of each sentence by the torch package: the noise floor
MAX_TOKENS = 510                               # the model's context is 512, a pad id at each end
DEV_VOICES = severity.DEV_VOICES
DEV_SENTENCES = {**severity.DEV_SENTENCES, **severity.TEST_SENTENCES}    # PR-009's twelve, all spent: dev data here
NOTE = ("SYNTHETIC SPEECH: text-to-speech, one model at several numeric precisions. Research code; "
        "no clinical validity is claimed or implied.")
IDENTITY = {"f0_offset_hz": 0.0, "f0_dev_scale": 1.0, "rate": 1.0, "pause_scale": 1.0}   # a copy changes nothing


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rung_file(rung: str, root=ROOT, rungs: dict = RUNGS) -> pathlib.Path:
    """The rung's file, its SHA-256 the pinned one. Raises before anything is loaded."""
    name, digest = rungs[rung]
    path = pathlib.Path(root) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: docs/m26-log.md has the download commands")
    got = sha256(path)
    if got != digest:
        raise ValueError(f"{path} has SHA-256 {got}, the pinned file has {digest}")
    return path


class TorchKokoro:
    """The kokoro package: text to audio, and the phoneme ids it spoke, chunk by chunk."""

    def __init__(self):
        self.pipeline = None

    def load(self) -> None:
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")

    def speak(self, text: str, voice: str) -> tuple[np.ndarray, list[list[int]]]:
        vocab = self.pipeline.model.vocab
        audio, chunks = [], []
        for r in self.pipeline(text, voice=voice):
            if r.audio is None:
                continue
            audio.append(np.asarray(r.audio, dtype=np.float32))
            chunks.append([vocab[p] for p in r.phonemes if p in vocab])
        return (np.concatenate(audio) if audio else np.zeros(0, np.float32)), chunks


class OnnxKokoro:
    """One ONNX file through ONNX Runtime on the CPU: phoneme ids to audio."""

    def __init__(self, rung: str, root=ROOT, rungs: dict = RUNGS):
        self.rung, self.root, self.rungs = rung, pathlib.Path(root), rungs
        self.session, self.voices, self.load_s, self.size_mb = None, {}, None, None

    def load(self) -> None:
        import onnxruntime as ort
        path = rung_file(self.rung, self.root, self.rungs)
        self.size_mb = round(path.stat().st_size / 1e6, 1)
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.load_s = round(time.perf_counter() - t0, 2)

    def voice(self, name: str) -> np.ndarray:
        if name not in self.voices:
            self.voices[name] = np.fromfile(self.root / "voices" / f"{name}.bin", dtype=np.float32).reshape(-1, 1, 256)
        return self.voices[name]

    def speak(self, chunks: list[list[int]], voice: str) -> np.ndarray:
        out = []
        for ids in chunks:
            if not 0 < len(ids) <= MAX_TOKENS:
                raise ValueError(f"{len(ids)} phoneme ids in one chunk; the model takes 1 to {MAX_TOKENS}")
            feed = {"input_ids": np.asarray([[0, *ids, 0]], dtype=np.int64), "style": self.voice(voice)[len(ids)],
                    "speed": np.ones(1, dtype=np.float32)}
            out.append(np.asarray(self.session.run(None, feed)[0], dtype=np.float32).reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def compare(ref: np.ndarray, test: np.ndarray, sr: int = SR) -> dict:
    """severity's four measures of `test` against `ref`, each clip's pauses found in itself.
    A render with a sample that is not a number has no measures: {"finite": False, ...}."""
    test = np.asarray(test, dtype=np.float32)
    bad = int((~np.isfinite(test)).sum())
    if bad or len(test) == 0:
        return {"finite": False, "nonfinite_samples": bad, "n_samples": len(test), "measures": None}
    a, b = severity.trim(ref, sr), severity.trim(test, sr)
    if len(b) == 0:                                                    # finite and silent
        return {"finite": True, "nonfinite_samples": 0, "n_samples": len(test), "measures": None}
    r = severity.Render(b, [(s, s + d) for s, d in pause_segments(a, sr)], [(s, s + d) for s, d in pause_segments(b, sr)])
    got = severity.achieved(a, r, sr)
    return {"finite": True, "nonfinite_samples": 0, "n_samples": len(test), "speech_s": round(len(b) / sr, 3),
            "measures": {k: float(v) for k, v in got._asdict().items()}}


def score_heard(text: str, heard: str) -> dict:
    """The recognizer's transcript of a render against the text it was given: raw and normalized."""
    out = {}
    for name, normalizer in wer.NORMALIZERS.items():
        e = wer.wer(text, heard, normalizer)
        out[name] = {"errors": e.errors, "words": e.n_words}
    return out


def whisper_hearer(model_name: str = "base.en"):
    """hear(audio at SR) -> transcript, through the rungs' own recognizer class."""
    from scipy.signal import resample_poly

    from .r0 import WhisperASR
    asr = WhisperASR(model_name)
    asr.load()
    return lambda audio: asr.transcribe(resample_poly(np.asarray(audio, dtype=np.float32), ASR_SR, SR).astype(np.float32))


def write_wav(path, audio: np.ndarray, rung: str, voice: str, text: str) -> None:
    """A WAV whose own metadata says it is synthetic, and which copy of the model spoke it."""
    import soundfile as sf
    with sf.SoundFile(str(path), "w", samplerate=SR, channels=1, subtype="PCM_16") as f:
        f.title = f"synthetic {rung}"
        f.comment = f"{NOTE} Rung {rung}, voice {voice}. Text: {text}"
        f.software = "s2s-lab quantladder.py"
        f.write(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0))


def _row(voice, sid, rung, repeat, against, audio, seconds, ref, text, hear):
    c = compare(ref, audio)
    row = {"voice": voice, "sentence": sid, "rung": rung, "repeat": repeat, "against": against,
           "seconds": round(seconds, 3), "rtf": round(seconds / (len(audio) / SR), 3) if len(audio) else None,
           "audio_sha256": hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest(), **c}
    if hear is not None and c["finite"]:
        row["heard"] = hear(audio)
        row["wer"] = score_heard(text, row["heard"])
    return row


def run(sentences: dict, voices, torch_tts, onnx: dict, hear=None, listen=None, torch_repeats: int = TORCH_REPEATS) -> dict:
    """Every voice x sentence through every rung. Takes the loaded speakers (torch_tts.speak(text,
    voice) -> (audio, chunks); onnx[rung].speak(chunks, voice) -> audio), hear(audio) -> text or
    None. Returns {"rows": one per voice x sentence x rung (torch: one per repeat), all against the
    reference rung's render; "floor_rows": torch's repeats against torch's first render}."""
    if REFERENCE not in onnx:
        raise ValueError(f"the reference rung {REFERENCE} is not among {sorted(onnx)}")
    rows, floor_rows = [], []
    for voice in voices:
        for sid, text in sentences.items():
            torch_renders = []
            for k in range(torch_repeats):
                t0 = time.perf_counter()
                audio, chunks = torch_tts.speak(text, voice)
                torch_renders.append((audio, time.perf_counter() - t0))
            renders = {}
            for rung, speaker in onnx.items():
                t0 = time.perf_counter()
                audio = speaker.speak(chunks, voice)
                renders[rung] = (audio, time.perf_counter() - t0)
            ref = renders[REFERENCE][0]
            if not np.isfinite(ref).all() or len(ref) == 0:
                raise RuntimeError(f"the reference render of {voice} {sid} is not audio")
            for rung, (audio, seconds) in renders.items():
                rows.append(_row(voice, sid, rung, 0, REFERENCE, audio, seconds, ref, text, hear))
            for k, (audio, seconds) in enumerate(torch_renders):
                rows.append(_row(voice, sid, TORCH, k, REFERENCE, audio, seconds, ref, text, hear if k == 0 else None))
                if k:
                    floor_rows.append(_row(voice, sid, TORCH, k, TORCH, audio, seconds, torch_renders[0][0], text, None))
            if listen is not None:
                for rung, audio in [(r, a) for r, (a, _) in renders.items()] + [(TORCH, torch_renders[0][0])]:
                    if np.isfinite(audio).all():
                        write_wav(pathlib.Path(listen) / f"synthetic-{rung}-{voice}-{sid}.wav", audio, rung, voice, text)
    return {"sr": SR, "reference": REFERENCE, "repo": REPO, "revision": REVISION, "torch_repeats": torch_repeats,
            "rungs": {r: {"file": RUNGS[r][0], "sha256": RUNGS[r][1], "size_mb": getattr(s, "size_mb", None),
                          "load_s": getattr(s, "load_s", None)} for r, s in onnx.items() if r in RUNGS},
            "voices": list(voices), "sentences": dict(sentences), "rows": rows, "floor_rows": floor_rows}


def deviation(field: str, value: float) -> float:
    """How far a measure is from 'nothing changed': Hz for the offset, a share of 1 for the others."""
    return abs(value - IDENTITY[field])


def floor(results: dict) -> dict:
    """The noise floor: torch against torch. {"field": {"n", "median", "p95", "max"}} of the deviations."""
    out = {}
    for field in severity.FIELDS:
        d = np.array([deviation(field, r["measures"][field]) for r in results["floor_rows"]
                      if r["measures"] and r["measures"][field] == r["measures"][field]])
        out[field] = ({"n": int(len(d)), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max())}
                      if len(d) else {"n": 0, "median": None, "p95": None, "max": None})
    return out


def summarize(results: dict) -> dict:
    """Per rung: renders, how many were audio, each measure's deviations and how many are within
    PR-009's tolerance (3 Hz for the offset, 10% for the others), WER, real-time factor."""
    out = {}
    for rung in dict.fromkeys(r["rung"] for r in results["rows"]):
        mine = [r for r in results["rows"] if r["rung"] == rung and r["repeat"] == 0]
        s = {"renders": len(mine), "finite": sum(r["finite"] for r in mine),
             "rtf_median": float(np.median([r["rtf"] for r in mine if r["rtf"] is not None])) if mine else None, "measures": {}}
        for field in severity.FIELDS:
            vals = [r["measures"][field] for r in mine if r["measures"] and r["measures"][field] == r["measures"][field]]
            d = np.array([deviation(field, v) for v in vals])
            s["measures"][field] = {"n": len(vals), "within": sum(severity.within(field, IDENTITY[field], v) for v in vals),
                                    "median": float(np.median(d)) if len(d) else None, "max": float(d.max()) if len(d) else None,
                                    "mean_signed": float(np.mean([v - IDENTITY[field] for v in vals])) if vals else None}
        for name in wer.NORMALIZERS:
            scored = [r["wer"][name] for r in mine if "wer" in r]
            words = sum(x["words"] for x in scored)
            s[f"wer_{name}"] = {"errors": sum(x["errors"] for x in scored), "words": words, "renders": len(scored),
                                "rate": sum(x["errors"] for x in scored) / words if words else None}
        out[rung] = s
    return out


def report(results: dict) -> str:
    """The ladder's table, then the noise floor."""
    s, f = summarize(results), floor(results)
    lines = [NOTE, "",
             f"reference: {results['reference']}; {len(results['voices'])} voices x {len(results['sentences'])} sentences; "
             f"within = PR-009's tolerance (3 Hz; 10%)", "",
             "| rung | size MB | audio | pitch offset | pitch range | rate | pauses | WER raw / norm | RTF |",
             "|---|---:|---:|---|---|---|---|---:|---:|"]
    cell = lambda m: "—" if not m["n"] else f"{m['within']} of {m['n']} (median {m['median']:.2f}, max {m['max']:.2f})"
    pct = lambda w: "—" if w["rate"] is None else f"{100 * w['rate']:.1f}%"
    for rung, r in s.items():
        size = results["rungs"].get(rung, {}).get("size_mb")
        lines.append(f"| {rung} | {size if size is not None else '—'} | {r['finite']} of {r['renders']} | "
                     + " | ".join(cell(r["measures"][k]) for k in severity.FIELDS)
                     + f" | {pct(r['wer_raw'])} / {pct(r['wer_normalized'])} | {r['rtf_median']:.2f} |")
    lines += ["", f"noise floor, torch against torch ({results['torch_repeats'] - 1} repeats a sentence): deviation median / 95th percentile / max"]
    for field in severity.FIELDS:
        x = f[field]
        lines.append(f"  {field:13s} " + ("no pairs" if not x["n"] else f"{x['median']:.3f} / {x['p95']:.3f} / {x['max']:.3f}   ({x['n']} pairs)"))
    bad = [(r["rung"], r["voice"], r["sentence"], r["nonfinite_samples"], r["n_samples"]) for r in results["rows"] if not r["finite"]]
    lines += ["", f"renders that are not audio: {len(bad)}"] + [f"  {rung} {voice} {sid}: {n} of {total} samples not a number"
                                                                for rung, voice, sid, n, total in bad]
    return "\n".join(lines)


def dev(listen: bool = False, out="results/m26", hear=None, root=ROOT) -> dict:
    out = pathlib.Path(out)
    folder = out / "listen-dev" if listen else None
    (folder or out).mkdir(parents=True, exist_ok=True)
    for rung in RUNGS:                                                  # every file checked before anything loads
        rung_file(rung, root)
    torch_tts = TorchKokoro()
    torch_tts.load()
    onnx = {rung: OnnxKokoro(rung, root) for rung in RUNGS}
    for speaker in onnx.values():
        speaker.load()
    t0 = time.time()
    results = run(DEV_SENTENCES, DEV_VOICES, torch_tts, onnx, hear=hear or whisper_hearer(), listen=folder)
    results["seconds"] = round(time.time() - t0)
    results["set"] = "dev"
    (out / "dev.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["dev"] and not set(args[1:]) - {"--listen"}:
        results = dev(listen="--listen" in args)
        print(report(results))
        print(f"\nsaved -> results/m26/dev.json; {results['seconds']} s on the CPU")
    else:
        sys.exit("usage: python -m s2slab.quantladder dev [--listen]")
