"""
PR-015's run and analysis: the streamed R4 against R4, on the frozen eval set, once.

Every definition here is fixed by docs/preregistration.md (PR-015), committed with this file and before
any eval-set turn of r4-gpu-fast or r4-gpu-stream existed. The run (r4streamdev.run with the frozen
setting, its start buffer in the rung) answers the 200 eval clips with R4, r4-gpu-fast and
r4-gpu-stream in one container, on one set of weights, Whisper small.en hearing every reply; this
module then says which of V1-V3 and S1-S5 hold.

    modal run modal_app/r4_bench.py::pr015 --confirm PR-015      once; saves runs/r4-gpu/pr015.json

    V1  the stream's reply text is R4's on >= 190 of 200 turns       else: not a valid test
    V2  the stream's codes are r4-gpu-fast's on >= 190 of 200 turns  else: not a valid test
    S1  first-sound p50, stream over R4, <= 0.70       upper end of the 95% interval
    S2  first-sound p90, stream over R4, <= 0.55       upper end of the 95% interval
    S3  total p50, stream over R4, <= 1.40             upper end of the 95% interval
    S4  <= 20 turns stall at all, <= 10 for more than 200 ms
    S5  Whisper's word error rate on the stream's replies <= R4's + 2.0 points

THE BOOTSTRAP IS PAIRED, as PR-002's (compare.py): 10,000 resamples of the 200 clip ids, seed 0, both
rungs' percentiles recomputed on the same resampled clips.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, percentile_ratio, resamples
from .r4stream import BLOCK_CODES, FIRST_CODES, START_BUFFER_S

REGISTRATION = "PR-015"
RUN = pathlib.Path("runs/r4-gpu/pr015.json")
STREAM = "r4-gpu-stream"
SAME_MIN = 190                               # V1, V2, of 200
P50_MAX, P90_MAX, TOTAL_MAX = 0.70, 0.55, 1.40
STALL_TURNS_MAX, LONG_STALL_TURNS_MAX, LONG_STALL_MS = 20, 10, 200.0
WER_MARGIN = 2.0                             # points


def eval_clips(confirm: str = "", root=None) -> tuple[dict, dict]:
    """The 200 eval clips as {id: audio} and {id: cohort}. PR-015's confirmation's: refused without it."""
    if confirm != REGISTRATION:
        raise PermissionError(f"the eval clips are {REGISTRATION}'s confirmation's: pass confirm=\"{REGISTRATION}\"")
    from .evalset import EVAL_ROOT, load, verify
    problems = verify(root or EVAL_ROOT)
    if problems:
        raise RuntimeError("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    items = load(root or EVAL_ROOT)
    return {it.id: it.audio for it in items}, {it.id: it.cohort for it in items}


def run(clips: dict, sr: int, cohorts: dict, **kw) -> dict:
    """R4, r4-gpu-fast and r4-gpu-stream as frozen (its start buffer in the rung), every clip, no WAVs kept."""
    from .r4streamdev import run as dev_run
    blob = dev_run(clips, sr, settings=((FIRST_CODES, BLOCK_CODES),), start_buffer_s=START_BUFFER_S, listen=0, **kw)
    blob["rungs"][STREAM] = blob["rungs"].pop(f"{STREAM}/{FIRST_CODES}+{BLOCK_CODES}")
    for name, r in blob["rungs"].items():
        for clip, t in r["turns"].items():
            t["cohort"] = cohorts[clip]
    blob.pop("listen")
    blob["setting"] = {"first_codes": FIRST_CODES, "block_codes": BLOCK_CODES, "start_buffer_s": START_BUFFER_S}
    return blob


def _wer(turns) -> float:
    words = sum(t["words"] for t in turns)
    return 100 * sum(t["word_errors"] for t in turns) / words if words else float("nan")


def criteria(blob: dict, n_boot: int = N_BOOT) -> dict:
    """Everything PR-015 asks of the run: {"valid", "v", "checks": {name: (passed, detail)}, "verdict", "describe"}.
    Raises ValueError if a turn of R4 or the stream has no first sound (PR-015 drops no clip)."""
    ids = blob["clips"]
    R = blob["rungs"]
    b = [R["r4-gpu"]["turns"][i] for i in ids]
    f = [R["r4-gpu-fast"]["turns"][i] for i in ids]
    s = [R[STREAM]["turns"][i] for i in ids]
    silent = [i for i, bt, st in zip(ids, b, s) if bt["first_sound_ms"] is None or st["first_sound_ms"] is None]
    if silent:
        raise ValueError(f"no first sound for {silent[:5]}: PR-015 drops no clip -- report this as a deviation")
    idx = resamples(len(ids), n_boot)
    same_text = sum(bt["reply_text"] == st["reply_text"] for bt, st in zip(b, s))
    same_codes = sum(bool(st.get("codes_as_fast")) for st in s)
    s1 = percentile_ratio([t["first_sound_ms"] for t in s], [t["first_sound_ms"] for t in b], 50, idx)
    s2 = percentile_ratio([t["first_sound_ms"] for t in s], [t["first_sound_ms"] for t in b], 90, idx)
    s3 = percentile_ratio([t["total_ms"] for t in s], [t["total_ms"] for t in b], 50, idx)
    stalls = [t["stall_ms"] for t in s]
    n_stall, n_long = sum(v > 0 for v in stalls), sum(v > LONG_STALL_MS for v in stalls)
    wer_b, wer_s = _wer(b), _wer(s)
    checks = {
        "S1 median": (s1.hi <= P50_MAX, f"first-sound p50 ratio {s1}; needs upper end <= {P50_MAX}"),
        "S2 tail": (s2.hi <= P90_MAX, f"first-sound p90 ratio {s2}; needs upper end <= {P90_MAX}"),
        "S3 speed guardrail": (s3.hi <= TOTAL_MAX, f"total p50 ratio {s3}; needs upper end <= {TOTAL_MAX}"),
        "S4 stall guardrail": (n_stall <= STALL_TURNS_MAX and n_long <= LONG_STALL_TURNS_MAX,
                               f"{n_stall} turns stall (max {STALL_TURNS_MAX}), {n_long} for more than "
                               f"{LONG_STALL_MS:.0f} ms (max {LONG_STALL_TURNS_MAX})"),
        "S5 words": (wer_s <= wer_b + WER_MARGIN,
                     f"Whisper WER on the replies {wer_s:.1f}% against R4's {wer_b:.1f}%; needs <= R4's + {WER_MARGIN:g}"),
    }
    v = {"V1 same text": (same_text >= SAME_MIN, f"the stream's reply text is R4's on {same_text} of {len(ids)}; needs >= {SAME_MIN}"),
         "V2 same codes": (same_codes >= SAME_MIN, f"the stream's codes are r4-gpu-fast's on {same_codes} of {len(ids)}; needs >= {SAME_MIN}")}
    valid = all(ok for ok, _ in v.values())
    failed = [name for name, (ok, _) in checks.items() if not ok]
    verdict = ("NOT A VALID TEST: " + ", ".join(k for k, (ok, _) in v.items() if not ok) if not valid
               else "SUPPORTED: all five criteria hold" if not failed else "NOT SUPPORTED: failed " + ", ".join(failed))
    p = lambda values, q: float(np.percentile(values, q))                         # noqa: E731
    cohorts: dict = {}
    for bt, st in zip(b, s):
        cohorts.setdefault(bt["cohort"], []).append((st["first_sound_ms"], bt["first_sound_ms"]))
    fast_total = percentile_ratio([t["total_ms"] for t in f], [t["total_ms"] for t in b], 50, idx)
    describe = {
        "turns": len(ids),
        "first_sound": {k: (p([t["first_sound_ms"] for t in T], 50), p([t["first_sound_ms"] for t in T], 90))
                        for k, T in (("stream", s), ("r4", b), ("fast", f))},
        "total": {k: (p([t["total_ms"] for t in T], 50), p([t["total_ms"] for t in T], 90))
                  for k, T in (("stream", s), ("r4", b), ("fast", f))},
        "reply_s_p50": {k: p([t["reply_s"] for t in T], 50) for k, T in (("stream", s), ("r4", b), ("fast", f))},
        "wer_fast": _wer(f),
        "fast_over_r4_total_p50": [fast_total.value, fast_total.lo, fast_total.hi],
        "cohort_first_sound_p50": {c: (p([x[0] for x in xs], 50), p([x[1] for x in xs], 50)) for c, xs in cohorts.items()},
        "stalls": sorted((v for v in stalls if v > 0), reverse=True),
    }
    return {"valid": valid, "v": v, "checks": checks, "verdict": verdict, "describe": describe}


def format_pr015(blob: dict, result: dict) -> str:
    """The text recorded under PR-015's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                                     # noqa: E731
    st = blob["setting"]
    lines = [f"PR-015: r4-gpu-stream (first block {st['first_codes']} codes, then {st['block_codes']}, start buffer "
             f"{st['start_buffer_s']:g} s) against r4-gpu, with r4-gpu-fast beside them",
             f"  commit {blob['commit']} · {blob['machine']} · {blob['versions']}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", ""]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["v"].items()]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              "  ms, p50 / p90          r4-gpu-stream      r4-gpu   r4-gpu-fast"]
    for label, key in (("first sound", "first_sound"), ("total", "total")):
        row = d[key]
        lines.append(f"  {label:<20} " + "  ".join(f"{row[k][0]:>6.0f} /{row[k][1]:>6.0f}" for k in ("stream", "r4", "fast")))
    lines.append("  reply s p50          " + "  ".join(f"{d['reply_s_p50'][k]:>14.2f}" for k in ("stream", "r4", "fast")))
    v, lo, hi = d["fast_over_r4_total_p50"]
    lines.append(f"  r4-gpu-fast's total p50 over R4's: {v:.3f} [95% CI {lo:.3f}, {hi:.3f}]; its Whisper WER {d['wer_fast']:.1f}%")
    lines.append("  first-sound p50 by cohort, stream vs R4: " + ", ".join(
        f"{c} {a:.0f} vs {r:.0f}" for c, (a, r) in d["cohort_first_sound_p50"].items()))
    stalls = d["stalls"]
    lines.append(f"  stalls: {len(stalls)} turns" + (f", longest {stalls[0]:.0f} ms" if stalls else ""))
    return "\n".join(lines)


def save(blob: dict, path: pathlib.Path = RUN) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path
