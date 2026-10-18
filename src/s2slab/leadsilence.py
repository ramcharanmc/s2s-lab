"""
M38: leading silence across TTS engines (v1's P3; page 07 §1.1: "dead air between the first audio byte and
the first audio energy ... engine spreads here are large").

Page 20 found Kokoro starting every reply with about 320 ms of silence, about 40% of what a caller waits for
R1's first word. This asks how much each engine puts there, where it comes from, and what a trim would cost.

    the text        R1's first clause of each of R0's 200 eval-set replies (chunking.split_clauses): the clause
                    whose silence a caller waits through
    the engines     Kokoro-82M (the rungs' TTS, voice af_heart); the same model through ONNX Runtime, fp32
                    (M26's file, the torch package's phoneme ids); Piper (VITS and espeak-ng, the LJSpeech
                    voice); NeMo FastPitch + HiFi-GAN, the LJSpeech base voice and FT-3's fine-tuned 9017
                    (rendered in FT-3's environment, src.ft3.leadsilence_fp); and R4's own speech, read from
                    its eval-set run (its own replies, the rungs' measure only: not the same text)
    the measures    leading silence by the rungs' measure (rungs.silence_before_speech_ms: 20 ms frames, the
                    first within 40 dB of the render's loudest) and by finer ones (5 ms frames, 30 and 50 dB);
                    the engine's own duration for the padding before the first phoneme, where it gives it
                    (Kokoro's duration predictor, FastPitch's); the first sound's class, from Kokoro's
                    phonemes for the clause (a stop's closure is speech that is silent, not padding)
    the trim        each render cut to KEEP_MS before its first frame within 50 dB, and Whisper small.en on
                    the first word, trimmed against untrimmed: what a trim would cost a quiet onset

Descriptive: nothing is set against a bar. The rule for what follows is written in docs/m38-log.md.

    PYTHONPATH=src python -m s2slab.leadsilence render       Kokoro, its ONNX copy, Piper -> results/m38/
    cd ft && PYTHONPATH=.:../src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.leadsilence_fp     FastPitch, both voices
    PYTHONPATH=src python -m s2slab.leadsilence analyze      -> runs/tts/leading-silence.json and the table
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

from .chunking import split_clauses
from .rungs import silence_before_speech_ms
from .vad import frame_rms

RUN_R0 = "runs/r0-gpu/evalset-modal-t4-pr002.json"         # the replies whose first clauses are spoken
RUN_R4 = "runs/r4-gpu/evalset-modal-l40s.json"             # R4's own speech, the rungs' measure only
OUT = "results/m38"                                        # renders and their sidecars (ignored by git)
RESULT = "runs/tts/leading-silence.json"
PIPER_VOICE = "data/piper/en_US-ljspeech-medium.onnx"      # ignored by git; docs/m38-log.md has the download
KOKORO_VOICE = "af_heart"
FINE_MS = 5.0
FINE_DB = (30.0, 50.0)
KEEP_MS = 20.0                                             # what a trim keeps before the first frame within 50 dB
ASR_SR = 16_000
NOTE = "SYNTHETIC SPEECH: text-to-speech renders of an agent's replies, for timing. Research code."

VOWELS = set("aeiouyæɑɐɒɔəɚɛɜɝɪʊʌøœɨʉɯɤᵊᵻAIOWY")           # misaki writes diphthongs as A I O W Y
SONORANTS = set("mnŋlɹrwjɾɫ")
VOICED_OBSTRUENTS = set("bdɡgvðzʒʤ")
VOICELESS_STOPS = set("ptkʔ")
VOICELESS_FRICATIVES = set("fθsʃhçxʧ")
MARKS = set("ˈˌːˑ ‿'\"-.,;:!?()[]")


# ------------------------------------------------------------------ the measures


def onset_ms(audio: np.ndarray, sr: int, frame_ms: float = FINE_MS, db: float = 50.0) -> float:
    """The start of the first frame within `db` of the render's loudest, in ms; NaN if nothing rises."""
    rms = frame_rms(np.asarray(audio, dtype=np.float32), sr, frame_ms)
    if rms.size == 0 or rms.max() < 1e-5:
        return math.nan
    return float(np.argmax(rms >= rms.max() * 10 ** (-db / 20))) * frame_ms


def measures(audio: np.ndarray, sr: int) -> dict:
    """The rungs' leading silence and the two finer ones, in ms."""
    return {"rung_ms": silence_before_speech_ms(audio, sr),
            **{f"fine{int(db)}_ms": onset_ms(audio, sr, FINE_MS, db) for db in FINE_DB}}


def trim(audio: np.ndarray, sr: int, keep_ms: float = KEEP_MS) -> tuple[np.ndarray, float]:
    """The render from keep_ms before its first frame within 50 dB; and the ms cut."""
    at = onset_ms(audio, sr, FINE_MS, 50.0)
    if math.isnan(at):
        return audio, 0.0
    cut = int(max(0.0, at - keep_ms) / 1000 * sr)
    return audio[cut:], cut / sr * 1000


def sound_class(phonemes: str | None) -> str:
    """The class of a clause's first sound, from its phonemes: vowel, sonorant, voiced obstruent, voiceless
    stop or voiceless fricative; "other" for what is not one of them, "unknown" with no phonemes."""
    for ch in phonemes or "":
        if ch in MARKS:
            continue
        for name, chars in (("vowel", VOWELS), ("sonorant", SONORANTS), ("voiced obstruent", VOICED_OBSTRUENTS),
                            ("voiceless stop", VOICELESS_STOPS), ("voiceless fricative", VOICELESS_FRICATIVES)):
            if ch in chars:
                return name
        return "other"
    return "unknown"


def first_clauses(run=RUN_R0, root=".") -> dict:
    """{clip: R1's first clause of R0's reply}, the clips with a reply, in the run's order."""
    turns = json.loads((pathlib.Path(root) / run).read_text(encoding="utf-8"))["turns"]
    out = {}
    for clip, t in turns.items():
        clauses = split_clauses(t.get("reply_text") or "")
        if clauses:
            out[clip] = clauses[0]
    return out


# ------------------------------------------------------------------ the engines


class KokoroEngine:
    """The kokoro package, as the rungs speak: the audio, the phonemes, and the duration its predictor gave the
    padding before the first phoneme (pred_dur[0], in the model's frames)."""

    name, sr = "kokoro", 24_000

    def __init__(self, voice: str = KOKORO_VOICE):
        self.voice, self.pipeline = voice, None

    def load(self) -> None:
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
        self.pipeline.load_voice(self.voice)

    def render(self, text: str) -> dict:
        results = [r for r in self.pipeline(text, voice=self.voice) if r.audio is not None]
        return kokoro_row(results, self.pipeline.model.vocab, self.sr)


def kokoro_row(results, vocab: dict, sr: int) -> dict:
    """Kokoro's results for one text as a render: the padding read off the first result's durations."""
    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32).reshape(-1) for r in results]) if results else np.zeros(0, np.float32)
    pad_ms = None
    if results and getattr(results[0], "output", None) is not None and getattr(results[0].output, "pred_dur", None) is not None:
        dur = np.asarray(results[0].output.pred_dur, dtype=float).reshape(-1)
        first = np.asarray(results[0].audio, dtype=np.float32).reshape(-1)
        if dur.sum() > 0:
            pad_ms = float(dur[0] * len(first) / dur.sum() / sr * 1000)        # the model's frames, as samples
    return {"audio": audio, "sr": sr, "pad_ms": pad_ms, "phonemes": results[0].phonemes if results else None,
            "ids": [[vocab[p] for p in r.phonemes if p in vocab] for r in results]}


class KokoroOnnxEngine:
    """M26's fp32 ONNX copy of Kokoro through ONNX Runtime, given the torch package's phoneme ids: the same
    model, another runtime. It returns the waveform only, so no padding."""

    name, sr = "kokoro-onnx", 24_000

    def __init__(self, voice: str = KOKORO_VOICE):
        from .quantladder import OnnxKokoro
        self.voice, self.model = voice, OnnxKokoro("onnx-fp32")

    def load(self) -> None:
        self.model.load()

    def render(self, ids: list[list[int]]) -> dict:
        return {"audio": self.model.speak(ids, self.voice), "sr": self.sr, "pad_ms": None, "phonemes": None}


class PiperEngine:
    """Piper (VITS, espeak-ng's phonemes), the LJSpeech voice. Its duration predictor is inside the ONNX
    file, so no padding."""

    name = "piper"

    def __init__(self, path: str = PIPER_VOICE):
        self.path, self.voice, self.sr = path, None, None

    def load(self) -> None:
        from piper import PiperVoice
        if not pathlib.Path(self.path).exists():
            raise FileNotFoundError(f"{self.path} is missing: docs/m38-log.md has the download")
        self.voice = PiperVoice.load(self.path)
        self.sr = int(self.voice.config.sample_rate)

    def render(self, text: str) -> dict:
        return piper_row(self.voice, text, self.sr)


def piper_row(voice, text: str, sr: int) -> dict:
    """One Piper render, through either of its APIs: synthesize() yielding chunks (1.3 on), or
    synthesize_stream_raw() yielding 16-bit bytes (before)."""
    if hasattr(voice, "synthesize_stream_raw"):
        raw = b"".join(voice.synthesize_stream_raw(text))
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    else:
        audio = np.concatenate([np.asarray(c.audio_float_array, dtype=np.float32).reshape(-1) for c in voice.synthesize(text)])
    phonemes = None
    try:
        phonemes = "".join(voice.phonemize(text)[0])
    except Exception:                                           # the phonemes are a label, not a measure
        pass
    return {"audio": audio, "sr": sr, "pad_ms": None, "phonemes": phonemes}


# ------------------------------------------------------------------ rendering and analysis


def save_renders(engine_name: str, rows: dict, out=OUT) -> pathlib.Path:
    """Each render as a WAV under out/<engine>/, and renders.json beside them: the text, rate, padding and
    phonemes of each, and the note that the audio is synthetic."""
    import soundfile as sf
    folder = pathlib.Path(out) / engine_name
    folder.mkdir(parents=True, exist_ok=True)
    side = {"engine": engine_name, "note": NOTE, "renders": {}}
    for clip, r in rows.items():
        sf.write(folder / f"{clip}.wav", np.asarray(r["audio"], dtype=np.float32), r["sr"], subtype="FLOAT")
        side["renders"][clip] = {k: r.get(k) for k in ("text", "sr", "pad_ms", "phonemes", "render_s")}
    (folder / "renders.json").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
    return folder


def render(clauses: dict, out=OUT, engines=None, log=print) -> dict:
    """Kokoro, its ONNX copy and Piper over the clauses, saved under out/. An engine that will not load is
    reported and left out; the ONNX copy needs Kokoro's phoneme ids."""
    made = {}
    kokoro = (engines or {}).get("kokoro") or KokoroEngine()
    kokoro.load()
    rows = {}
    for clip, text in clauses.items():
        t0 = time.perf_counter()
        rows[clip] = dict(kokoro.render(text), text=text, render_s=round(time.perf_counter() - t0, 3))
    made["kokoro"] = save_renders("kokoro", rows, out)
    for name, make in (("kokoro-onnx", KokoroOnnxEngine), ("piper", PiperEngine)):
        try:
            engine = (engines or {}).get(name) or make()
            engine.load()
        except Exception as e:
            log(f"{name} left out: {type(e).__name__}: {e}")
            continue
        out_rows = {}
        for clip, text in clauses.items():
            t0 = time.perf_counter()
            r = engine.render(rows[clip]["ids"]) if name == "kokoro-onnx" else engine.render(text)
            out_rows[clip] = dict(r, text=text, render_s=round(time.perf_counter() - t0, 3))
        made[name] = save_renders(name, out_rows, out)
    return made


def _first_word(text: str) -> str:
    from .wer import light
    words = light(text or "").split()
    return words[0] if words else ""


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(int(sr), ASR_SR)
    return resample_poly(np.asarray(audio, dtype=np.float32), ASR_SR // g, int(sr) // g).astype(np.float32)


def analyze(out=OUT, hear=None, root=".", r4_run=RUN_R4) -> dict:
    """Every saved render measured, trimmed and heard; R4's leading silence from its run. hear(audio_16k) -> text;
    Whisper small.en on the CPU if None."""
    import soundfile as sf

    from .wer import light, wer
    if hear is None:
        from .r0 import WhisperASR
        asr = WhisperASR("small.en")
        asr.load()
        hear = asr.transcribe
    out = pathlib.Path(out)
    kokoro_side = json.loads((out / "kokoro" / "renders.json").read_text(encoding="utf-8"))["renders"]
    engines = {}
    for side_path in sorted(out.glob("*/renders.json")):
        side = json.loads(side_path.read_text(encoding="utf-8"))
        rows = {}
        for clip, r in side["renders"].items():
            audio, sr = sf.read(side_path.parent / f"{clip}.wav", dtype="float32")
            cut_audio, cut_ms = trim(audio, sr)
            heard, heard_cut = hear(_to_16k(audio, sr)), hear(_to_16k(cut_audio, sr))
            first = _first_word(r["text"])
            rows[clip] = {**measures(audio, sr), "pad_ms": r.get("pad_ms"), "seconds": round(len(audio) / sr, 3),
                          "class": sound_class(kokoro_side.get(clip, {}).get("phonemes") or r.get("phonemes")),
                          "cut_ms": round(cut_ms, 1), "first_word": first,
                          "first_heard": _first_word(heard) == first, "first_heard_cut": _first_word(heard_cut) == first,
                          "errors": wer(r["text"], heard, light).errors, "errors_cut": wer(r["text"], heard_cut, light).errors,
                          "words": len(light(r["text"]).split()), "render_s": r.get("render_s")}
        engines[side["engine"]] = {"rows": rows}
    r4 = json.loads((pathlib.Path(root) / r4_run).read_text(encoding="utf-8"))
    engines["r4-omni"] = {"rows": {clip: {"rung_ms": t.get("leading_silence_ms")} for clip, t in r4["turns"].items()},
                          "note": "R4's own replies (not these clauses), its eval-set run, the rungs' measure only"}
    return {"run": "M38: leading silence across TTS engines, R1's 200 first clauses (R4 from its eval-set run)",
            "keep_ms": KEEP_MS, "fine_ms": FINE_MS, "engines": engines, "summary": summary(engines)}


def _pct(values, q):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.percentile(v, q)), 1) if v else None


def summary(engines: dict) -> dict:
    """Per engine: the measures' p50 and p90, the padding, by first-sound class, and the trim's cost."""
    out = {}
    for name, e in engines.items():
        rows = list(e["rows"].values())
        s = {"n": len(rows)}
        for key in ("rung_ms", "fine30_ms", "fine50_ms", "pad_ms", "cut_ms"):
            if any(r.get(key) is not None for r in rows):
                s[key] = (_pct([r.get(key) for r in rows], 50), _pct([r.get(key) for r in rows], 90))
        if any("class" in r for r in rows):
            classes: dict = {}
            for r in rows:
                classes.setdefault(r["class"], []).append(r["rung_ms"])
            s["by_class"] = {c: (len(v), _pct(v, 50)) for c, v in sorted(classes.items())}
        if any("first_heard" in r for r in rows):
            words = sum(r["words"] for r in rows)
            s["first_heard"] = (sum(r["first_heard"] for r in rows), sum(r["first_heard_cut"] for r in rows))
            s["wer"] = (round(100 * sum(r["errors"] for r in rows) / words, 1) if words else None,
                        round(100 * sum(r["errors_cut"] for r in rows) / words, 1) if words else None)
        out[name] = s
    return out


def format_report(blob: dict) -> str:
    f = lambda x: "-" if x is None else f"{x:,.0f}"                                     # noqa: E731
    pair = lambda p: "-" if p is None else f"{f(p[0])} / {f(p[1])}"                     # noqa: E731
    lines = [blob["run"], "",
             "| engine | n | leading silence, rungs' measure, p50 / p90 ms | 5 ms frames at 30 dB | at 50 dB | padding by "
             "its own durations | a trim cuts | first word heard, as is / trimmed | WER %, as is / trimmed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in blob["summary"].items():
        heard = "-" if s.get("first_heard") is None else "{} / {}".format(*s["first_heard"])
        errs = "-" if s.get("wer") is None else "{} / {}".format(*s["wer"])
        lines.append(f"| {name} | {s['n']} | {pair(s.get('rung_ms'))} | {pair(s.get('fine30_ms'))} | {pair(s.get('fine50_ms'))} | "
                     f"{pair(s.get('pad_ms'))} | {pair(s.get('cut_ms'))} | {heard} | {errs} |")
    lines += ["", "By the clause's first sound (renders, the rungs' measure p50 ms):"]
    for name, s in blob["summary"].items():
        if "by_class" in s:
            lines.append(f"  {name}: " + ", ".join(f"{c} {f(m)} ({n})" for c, (n, m) in s["by_class"].items()))
    return "\n".join(lines)


def save(blob: dict, path=RESULT) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["render"]:
        clauses = first_clauses()
        for name, folder in render(clauses).items():
            print(f"{name}: {len(clauses)} renders -> {folder}")
    elif argv[:1] == ["analyze"]:
        blob = analyze()
        path = save(blob)
        print(format_report(blob))
        print(f"\nsaved -> {path}")
    else:
        raise SystemExit("usage: python -m s2slab.leadsilence render | analyze")


if __name__ == "__main__":
    main()
