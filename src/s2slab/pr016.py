"""
PR-016's run and analysis: R4 streamed on a right-sized cache against R4 streamed as frozen for PR-015, on
the frozen eval set, once.

Every definition here is fixed by docs/preregistration.md (PR-016), committed with this file and before
any eval-set turn of r4-gpu-stream-sized existed. The run (r4sizeddev.run with the start buffer frozen in
both streams) answers the 200 eval clips with r4-gpu-fast, r4-gpu-stream and r4-gpu-stream-sized in one
container, on one set of weights, Whisper small.en hearing every reply; this module then says which of
V1-V3 and S1-S5 hold.

    modal run modal_app/r4_bench.py::pr016 --confirm PR-016      once; saves runs/r4-gpu/pr016.json

    V1  the sized stream's reply text is the stream's on >= 190 of 200 turns     else: not a valid test
    V2  the sized stream's codes are r4-gpu-fast's on >= 190 of 200 turns        else: not a valid test
    V3  at most 2 of 200 sized replies cut at their cache's end                  else: not a valid test
    S1  first-sound p50, sized over stream, <= 0.90     upper end of the 95% interval
    S2  first-sound p90, sized over stream, <= 0.92     upper end of the 95% interval
    S3  total p50, sized over stream, <= 1.00           upper end of the 95% interval
    S4  <= 20 sized turns stall at all, <= 10 for more than 200 ms
    S5  Whisper's word error rate on the sized stream's replies <= the stream's + 1.0 point

THE BOOTSTRAP IS PAIRED, as PR-002's and PR-015's (compare.py): 10,000 resamples of the 200 clip ids,
seed 0, both rungs' percentiles recomputed on the same resampled clips.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, percentile_ratio, resamples
from .r4sized import ROOM_CODES, SIZES
from .r4stream import BLOCK_CODES, FIRST_CODES, START_BUFFER_S

REGISTRATION = "PR-016"
RUN = pathlib.Path("runs/r4-gpu/pr016.json")
SIZED, STREAM = "r4-gpu-stream-sized", "r4-gpu-stream"
SAME_MIN, CUT_MAX = 190, 2                     # V1, V2 of 200; V3
P50_MAX, P90_MAX, TOTAL_MAX = 0.90, 0.92, 1.00
STALL_TURNS_MAX, LONG_STALL_TURNS_MAX, LONG_STALL_MS = 20, 10, 200.0
WER_MARGIN = 1.0                               # points


def eval_clips(confirm: str = "", root=None) -> tuple[dict, dict]:
    """The 200 eval clips as {id: audio} and {id: cohort}. PR-016's confirmation's: refused without it."""
    if confirm != REGISTRATION:
        raise PermissionError(f"the eval clips are {REGISTRATION}'s confirmation's: pass confirm=\"{REGISTRATION}\"")
    from .evalset import EVAL_ROOT, load, verify
    problems = verify(root or EVAL_ROOT)
    if problems:
        raise RuntimeError("the eval set is not the frozen set:\n  " + "\n  ".join(problems[:10]))
    items = load(root or EVAL_ROOT)
    return {it.id: it.audio for it in items}, {it.id: it.cohort for it in items}


def run(clips: dict, sr: int, cohorts: dict, **kw) -> dict:
    """r4-gpu-fast and both streams as frozen (the start buffer in each), every clip, no WAVs kept."""
    from .r4sizeddev import run as dev_run
    blob = dev_run(clips, sr, listen=0, start_buffer_s=START_BUFFER_S, **kw)
    for r in blob["rungs"].values():
        for clip, t in r["turns"].items():
            t["cohort"] = cohorts[clip]
    blob.pop("listen")
    blob["setting"] = {"first_codes": FIRST_CODES, "block_codes": BLOCK_CODES, "start_buffer_s": START_BUFFER_S,
                       "sizes": list(SIZES), "room_codes": ROOM_CODES}
    return blob


def _wer(turns) -> float:
    words = sum(t["words"] for t in turns)
    return 100 * sum(t["word_errors"] for t in turns) / words if words else float("nan")


def criteria(blob: dict, n_boot: int = N_BOOT) -> dict:
    """Everything PR-016 asks of the run: {"valid", "v", "checks": {name: (passed, detail)}, "verdict", "describe"}.
    Raises ValueError if a turn of either stream has no first sound (PR-016 drops no clip)."""
    ids = blob["clips"]
    R = blob["rungs"]
    s = [R[STREAM]["turns"][i] for i in ids]
    z = [R[SIZED]["turns"][i] for i in ids]
    silent = [i for i, st, zt in zip(ids, s, z) if st["first_sound_ms"] is None or zt["first_sound_ms"] is None]
    if silent:
        raise ValueError(f"no first sound for {silent[:5]}: PR-016 drops no clip -- report this as a deviation")
    idx = resamples(len(ids), n_boot)
    same_text = sum(st["reply_text"] == zt["reply_text"] for st, zt in zip(s, z))
    same_codes = sum(bool(zt.get("codes_as_fast")) for zt in z)
    cut = sum(bool(zt.get("cut")) for zt in z)
    s1 = percentile_ratio([t["first_sound_ms"] for t in z], [t["first_sound_ms"] for t in s], 50, idx)
    s2 = percentile_ratio([t["first_sound_ms"] for t in z], [t["first_sound_ms"] for t in s], 90, idx)
    s3 = percentile_ratio([t["total_ms"] for t in z], [t["total_ms"] for t in s], 50, idx)
    stalls = [t["stall_ms"] for t in z]
    n_stall, n_long = sum(v > 0 for v in stalls), sum(v > LONG_STALL_MS for v in stalls)
    wer_s, wer_z = _wer(s), _wer(z)
    v = {"V1 same text": (same_text >= SAME_MIN, f"the sized stream's reply text is the stream's on {same_text} of {len(ids)}; needs >= {SAME_MIN}"),
         "V2 same codes": (same_codes >= SAME_MIN, f"the sized stream's codes are r4-gpu-fast's on {same_codes} of {len(ids)}; needs >= {SAME_MIN}"),
         "V3 no reply cut": (cut <= CUT_MAX, f"{cut} sized replies cut at their cache's end; needs <= {CUT_MAX}")}
    checks = {
        "S1 median": (s1.hi <= P50_MAX, f"first-sound p50 ratio {s1}; needs upper end <= {P50_MAX}"),
        "S2 tail": (s2.hi <= P90_MAX, f"first-sound p90 ratio {s2}; needs upper end <= {P90_MAX}"),
        "S3 speed guardrail": (s3.hi <= TOTAL_MAX, f"total p50 ratio {s3}; needs upper end <= {TOTAL_MAX}"),
        "S4 stall guardrail": (n_stall <= STALL_TURNS_MAX and n_long <= LONG_STALL_TURNS_MAX,
                               f"{n_stall} turns stall (max {STALL_TURNS_MAX}), {n_long} for more than "
                               f"{LONG_STALL_MS:.0f} ms (max {LONG_STALL_TURNS_MAX})"),
        "S5 words": (wer_z <= wer_s + WER_MARGIN,
                     f"Whisper WER on the replies {wer_z:.1f}% against the stream's {wer_s:.1f}%; needs <= the stream's + {WER_MARGIN:g}"),
    }
    valid = all(ok for ok, _ in v.values())
    failed = [name for name, (ok, _) in checks.items() if not ok]
    verdict = ("NOT A VALID TEST: " + ", ".join(k for k, (ok, _) in v.items() if not ok) if not valid
               else "SUPPORTED: all five criteria hold" if not failed else "NOT SUPPORTED: failed " + ", ".join(failed))
    p = lambda values, q: float(np.percentile(values, q))                         # noqa: E731
    f = [R["r4-gpu-fast"]["turns"][i] for i in ids]
    cohorts: dict = {}
    for st, zt in zip(s, z):
        cohorts.setdefault(st["cohort"], []).append((zt["first_sound_ms"], st["first_sound_ms"]))
    places: dict = {}
    for zt in z:
        places[str(zt.get("places"))] = places.get(str(zt.get("places")), 0) + 1
    describe = {
        "turns": len(ids),
        "first_sound": {k: (p([t["first_sound_ms"] for t in T], 50), p([t["first_sound_ms"] for t in T], 90))
                        for k, T in (("sized", z), ("stream", s), ("fast", f))},
        "total": {k: (p([t["total_ms"] for t in T], 50), p([t["total_ms"] for t in T], 90))
                  for k, T in (("sized", z), ("stream", s), ("fast", f))},
        "reply_s_p50": {k: p([t["reply_s"] for t in T], 50) for k, T in (("sized", z), ("stream", s), ("fast", f))},
        "stream_stalls": sorted((t["stall_ms"] for t in s if t["stall_ms"] > 0), reverse=True),
        "stream_codes_as_fast": sum(bool(t.get("codes_as_fast")) for t in s),
        "cohort_first_sound_p50": {c: (p([x[0] for x in xs], 50), p([x[1] for x in xs], 50)) for c, xs in cohorts.items()},
        "places": places, "stalls": sorted((x for x in stalls if x > 0), reverse=True),
    }
    return {"valid": valid, "v": v, "checks": checks, "verdict": verdict, "describe": describe}


def format_pr016(blob: dict, result: dict) -> str:
    """The text recorded under PR-016's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                                     # noqa: E731
    st = blob["setting"]
    lines = [f"PR-016: r4-gpu-stream-sized (caches of {', '.join(map(str, st['sizes']))} places, {st['room_codes']} codes of "
             f"room) against r4-gpu-stream, both first block {st['first_codes']} codes, then {st['block_codes']}, start "
             f"buffer {st['start_buffer_s']:g} s; r4-gpu-fast beside them",
             f"  commit {blob['commit']} · {blob['machine']} · {blob['versions']}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", ""]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["v"].items()]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              "  ms, p50 / p90          sized stream     the stream   r4-gpu-fast"]
    for label, key in (("first sound", "first_sound"), ("total", "total")):
        row = d[key]
        lines.append(f"  {label:<20} " + "  ".join(f"{row[k][0]:>6.0f} /{row[k][1]:>6.0f}" for k in ("sized", "stream", "fast")))
    lines.append("  reply s p50          " + "  ".join(f"{d['reply_s_p50'][k]:>14.2f}" for k in ("sized", "stream", "fast")))
    lines.append("  first-sound p50 by cohort, sized vs stream: " + ", ".join(
        f"{c} {a:.0f} vs {r:.0f}" for c, (a, r) in d["cohort_first_sound_p50"].items()))
    lines.append("  the sized stream's caches: " + ", ".join(f"{n} places on {k} turns" for n, k in sorted(d["places"].items())))
    for name, key in (("sized stream", "stalls"), ("the stream", "stream_stalls")):
        x = d[key]
        lines.append(f"  stalls, {name}: {len(x)} turns" + (f", longest {x[0]:.0f} ms" if x else ""))
    lines.append(f"  the stream's codes are r4-gpu-fast's on {d['stream_codes_as_fast']} of {d['turns']}")
    return "\n".join(lines)


def save(blob: dict, path: pathlib.Path = RUN) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path
