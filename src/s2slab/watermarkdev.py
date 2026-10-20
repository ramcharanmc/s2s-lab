"""
M43's development check, on the laptop, free: does the watermark stay findable, stay out of what was never marked,
and leave the severity render as it was? Against a rule written before the check (docs/m43-log.md, section 1).

    PYTHONPATH=src python -m s2slab.watermarkdev       Kokoro, the dev renders, the eval set; runs/watermark/dev.json

THE MATERIAL. severity's development set: its two dev voices, its four dev sentences, every one of its twelve dev
scores (both scales), as PR-009's dev work rendered them: 8 neutral clips and 96 renders, at 24 kHz. And as
controls never marked, those same clips unmarked and the eval set's 200 real clips (16 kHz).

FOR EACH RENDER, marked as the demo will mark it (the neutral clip and the render, each on its own):
  * severity's four measures (severity.achieved) on the marked pair against the unmarked pair;
  * the mark read back as written, after a 16-bit WAV, after a trip through 16 kHz, and after the browser's
    WebM/Opus at 48 kHz (reported only);
  * how far below the speech the mark sits (watermark.level_db).
FOR EACH NEUTRAL CLIP as Kokoro spoke it, untrimmed (a cascade's reply): the leading silence the rungs measure
(rungs.silence_before_speech_ms), marked against unmarked.

THE RULE (gate): GO to wiring the mark in if, on every clip,
  * every marked render and neutral clip reads as marked after the 16-bit WAV and after the trip through 16 kHz;
  * nothing unmarked reads as marked: the renders, the neutral clips, the eval set's 200;
  * the pitch offset moves under 0.3 Hz, and the pitch range, the rate and the pauses under 1%: a tenth of
    PR-009's own tolerances (3 Hz; 10%);
  * a reply's leading silence moves under 20 ms, one of the measure's frames.
Otherwise STOP: nothing is wired in, and why is recorded.
"""

from __future__ import annotations

import io
import json
import math
import pathlib

import numpy as np

RUN = pathlib.Path("runs/watermark/dev.json")
OFFSET_HZ_MAX, REL_MAX, SILENCE_MS_MAX = 0.3, 0.01, 20.0
RELATIVE = ("f0_dev_scale", "rate", "pause_scale")


def quantize16(x) -> np.ndarray:
    """A 16-bit WAV's samples, read back as float32."""
    return (np.round(np.clip(np.asarray(x, dtype=np.float32), -1, 1) * 32767) / 32767).astype(np.float32)


def opus_roundtrip(x, sr: int) -> tuple[np.ndarray, int]:
    """Through WebM/Opus at 48 kHz, as a browser records and plays it; (audio, 48000)."""
    import av

    from .watermark import to_rate
    x48 = to_rate(x, sr, 48_000)
    buf = io.BytesIO()
    out = av.open(buf, "w", format="webm")
    stream = out.add_stream("libopus", rate=48_000)
    stream.layout = "mono"
    n = stream.codec_context.frame_size or 960
    for i in range(0, len(x48), n):
        chunk = x48[i:i + n]
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk[None, :], format="flt", layout="mono")
        frame.sample_rate, frame.pts = 48_000, i
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    buf.seek(0)
    with av.open(buf) as cin:
        parts = [f.to_ndarray().reshape(-1) for f in cin.decode(audio=0)]
    return np.concatenate(parts).astype(np.float32), 48_000


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def run(speak, wm, controls, opus: bool = True) -> dict:
    """The check. speak(text, voice) -> float32 at 24 kHz; wm: a loaded watermark.Watermarker; controls:
    [(id, audio, sr)], clips never marked."""
    from . import severity
    from .rungs import silence_before_speech_ms
    from .watermark import MARK_RATE, level_db, to_rate
    sr = severity.KOKORO_SR
    assert sr == MARK_RATE
    neutrals, renders = [], []
    for voice in severity.DEV_VOICES:
        for sid, text in severity.DEV_SENTENCES.items():
            raw = speak(text, voice)
            raw_marked = wm.embed(raw, sr)
            neutral = severity.trim(raw, sr)
            neutral_marked = wm.embed(neutral, sr)
            nid = f"{voice}-{sid}"
            neutrals.append({"id": nid, "silence_ms": [_num(silence_before_speech_ms(raw, sr)),
                                                       _num(silence_before_speech_ms(raw_marked, sr))],
                             "unmarked": wm.detect(neutral, sr), "as_written": wm.detect(neutral_marked, sr),
                             "wav16": wm.detect(quantize16(neutral_marked), sr),
                             "via16k": wm.detect(to_rate(neutral_marked, sr, 16_000), 16_000)})
            for condition, scores in severity.SCORES.items():
                for score in scores:
                    t = severity.CONDITIONS[condition](score)
                    r = severity.render(neutral, sr, t)
                    marked = severity.Render(wm.embed(r.audio, sr), r.pauses_in, r.pauses_out)
                    plain, got = severity.achieved(neutral, r, sr), severity.achieved(neutral_marked, marked, sr)
                    row = {"id": f"{nid}-{condition}-{score:g}",
                           "plain": {f: _num(getattr(plain, f)) for f in severity.FIELDS},
                           "marked_measures": {f: _num(getattr(got, f)) for f in severity.FIELDS},
                           "level_db": level_db(r.audio, marked.audio),
                           "unmarked": wm.detect(r.audio, sr), "as_written": wm.detect(marked.audio, sr),
                           "wav16": wm.detect(quantize16(marked.audio), sr),
                           "via16k": wm.detect(to_rate(marked.audio, sr, 16_000), 16_000)}
                    if opus:
                        row["opus"] = wm.detect(*opus_roundtrip(marked.audio, sr))
                    renders.append(row)
    return {"neutrals": neutrals, "renders": renders,
            "controls": [{"id": cid, **wm.detect(audio, csr)} for cid, audio, csr in controls]}


def moved(row: dict) -> dict:
    """How far the mark moved each measure: Hz for the pitch offset, a share of the plain value for the rest."""
    out = {}
    for f, a in row["plain"].items():
        b = row["marked_measures"][f]
        if a is None or b is None:
            out[f] = None
        elif f == "f0_offset_hz":
            out[f] = abs(b - a)
        else:
            out[f] = abs(b - a) / abs(a) if a else None
    return out


def gate(blob: dict) -> dict:
    """The rule written before the check (module docstring)."""
    marked = blob["neutrals"] + blob["renders"]
    lost = [r["id"] for r in marked if not (r["wav16"]["marked"] and r["via16k"]["marked"])]
    alarms = ([r["id"] for r in marked if r["unmarked"]["marked"]] + [c["id"] for c in blob["controls"] if c["marked"]])
    worst = {f: max((moved(r)[f] for r in blob["renders"] if moved(r)[f] is not None), default=0.0)
             for f in ("f0_offset_hz",) + RELATIVE}
    over = [f for f, v in worst.items() if v >= (OFFSET_HZ_MAX if f == "f0_offset_hz" else REL_MAX)]
    shifts = [abs(n["silence_ms"][1] - n["silence_ms"][0]) for n in blob["neutrals"]
              if None not in n["silence_ms"]]
    silence = max(shifts, default=0.0)
    checks = {"found": (not lost, f"{len(marked) - len(lost)} of {len(marked)} marked clips found after a 16-bit WAV "
                                  f"and a trip through 16 kHz"),
              "no false alarm": (not alarms, f"{len(alarms)} of {len(marked) + len(blob['controls'])} unmarked clips "
                                             f"read as marked"),
              "render unchanged": (not over, "the most the mark moved: pitch offset "
                                             f"{worst['f0_offset_hz']:.3f} Hz (under {OFFSET_HZ_MAX}), "
                                             + ", ".join(f"{f} {100 * worst[f]:.2f}%" for f in RELATIVE)
                                             + f" (under {100 * REL_MAX:.0f}%)"),
              "silence unchanged": (silence < SILENCE_MS_MAX, f"a reply's leading silence moved {silence:.0f} ms at most "
                                                              f"(under {SILENCE_MS_MAX:.0f})")}
    go = all(ok for ok, _ in checks.values())
    return {"go": go, "checks": checks, "lost": lost, "alarms": alarms, "worst": worst, "silence_ms": silence}


def format_dev(blob: dict) -> str:
    g = gate(blob)
    R = blob["renders"]
    p = lambda v, q: float(np.percentile(v, q)) if v else float("nan")             # noqa: E731
    lines = [f"M43, the watermark's development check: {len(blob['neutrals'])} neutral clips and {len(R)} renders "
             f"(severity's dev set, 24 kHz), {len(blob['controls'])} eval-set clips never marked", ""]
    lines += [f"{name:<18} {'PASS' if ok else 'FAIL'}  {detail}" for name, (ok, detail) in g["checks"].items()]
    lines += ["", f"The rule (written before the check): {'GO: wire the mark in' if g['go'] else 'STOP: nothing wired in'}",
              "", "Descriptive, not the rule:",
              f"  the mark below the speech: {p([r['level_db'] for r in R], 50):.1f} dB at the median "
              f"({min(r['level_db'] for r in R):.1f} to {max(r['level_db'] for r in R):.1f})"]
    for k, what in (("as_written", "as written"), ("wav16", "after a 16-bit WAV"), ("via16k", "after 16 kHz"),
                    ("opus", "after WebM/Opus (reported only)")):
        rows = [r[k] for r in R if k in r]
        if rows:
            lines.append(f"  {what}: found on {sum(x['marked'] for x in rows)} of {len(rows)}, share p50 "
                         f"{p([x['share'] for x in rows], 50):.3f}, all 16 bits on {sum(x['bits'] == 16 for x in rows)}")
    ctrl = [c["share"] for c in blob["controls"]] + [r["unmarked"]["share"] for r in R]
    lines.append(f"  unmarked clips: the share at most {max(ctrl, default=0.0):.3f}")
    return "\n".join(lines)


def main(argv=None) -> None:
    import warnings

    from .evalset import load, verify
    from .severity import kokoro_speaker
    from .sweep import _git_commit
    from .watermark import TAG, Watermarker
    warnings.filterwarnings("ignore")
    if RUN.exists():
        raise SystemExit(f"{RUN} exists: the check is recorded once")
    problems = verify()
    if problems:
        raise SystemExit("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    wm = Watermarker().load()
    wm.embed(np.zeros(24_000, np.float32), 24_000)                   # the first mark compiles AudioSeal's layers
    controls = [(it.id, it.audio, 16_000) for it in load()]
    blob = {"run": "M43, the watermark's development check", "commit": _git_commit(), "watermark": TAG,
            **run(kokoro_speaker(), wm, controls)}
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(format_dev(blob))
    print(f"\nsaved -> {RUN}")


if __name__ == "__main__":
    main()
