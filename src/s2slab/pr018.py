"""
PR-018's analysis: R1 with its first piece's leading silence trimmed (r1-gpu-trim) against R1 as it is, on the same
turns of the frozen eval set, once.

Every definition here is fixed by docs/preregistration.md (PR-018), committed with this file and before any eval-set
turn of r1-gpu-trim existed. The run is the eval harness's own (evalrun.run, on one T4) for r1-gpu-trim, each turn
keeping what its trim cut (r1.FirstTrimmed.turn_extra); R1 as it is on the very same turn is computed from that
turn's pieces (r1trim.as_is: the first piece longer by the cut, its leading silence too). This module says which of
V1 and S1 to S3 hold.

    modal run modal_app/r0_bench.py::pr018 --confirm PR-018      once; saves runs/r1-gpu-trim/pr018.json

    V1  on all 200 turns what was cut is silent: its loudest 5 ms frame at least 50 dB below the piece's
        loudest                                                                      else: not a valid test
    S1  first-sound p50, trimmed over as is, <= 0.75     upper end of the 95% interval
    S2  first-sound p90, trimmed over as is, <= 0.85     upper end of the 95% interval
    S3  the trimmed turns that stall no more than 4 above as is's, and none for more than 200 ms

THE BOOTSTRAP IS PAIRED, as PR-002's to PR-017's (compare.py): 10,000 resamples of the 200 clip ids, seed 0.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, percentile_ratio, resamples
from .r1trim import OUT_SR, as_is

REGISTRATION = "PR-018"
RUNG = "r1-gpu-trim"
RUN = pathlib.Path("runs/r1-gpu-trim/pr018.json")
CUT_DB_MAX = -50.0                              # V1
P50_MAX, P90_MAX = 0.75, 0.85                   # S1, S2
STALL_EXTRA_MAX, STALL_LONGEST_MS = 4, 200.0    # S3


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def criteria(blob: dict, n_boot: int = N_BOOT) -> dict:
    """Everything PR-018 asks of the run: {"valid", "v", "checks", "verdict", "describe"}. Raises ValueError if a turn
    has no first sound either way (PR-018 drops no clip)."""
    ids = list(blob["turns"])
    T = [blob["turns"][i] for i in ids]
    back = [as_is(t, OUT_SR) for t in T]
    silent = [i for i, t, b in zip(ids, T, back) if _missing(t["first_sound_ms"]) or _missing(b["first_sound_ms"])]
    if silent:
        raise ValueError(f"no first sound for {silent[:5]}: PR-018 drops no clip -- report this as a deviation")
    idx = resamples(len(ids), n_boot)
    trimmed = [t["first_sound_ms"] for t in T]
    before = [b["first_sound_ms"] for b in back]
    s1, s2 = percentile_ratio(trimmed, before, 50, idx), percentile_ratio(trimmed, before, 90, idx)
    loud = [t.get("cut_peak_db") for t in T if t.get("cut_peak_db") is not None and t["cut_peak_db"] > CUT_DB_MAX]
    stall_t = [t["stall_ms"] for t in T]
    stall_b = [b["stall_ms"] for b in back]
    n_t, n_b = sum(x > 0 for x in stall_t), sum(x > 0 for x in stall_b)
    v = {"V1 the cut was silent": (not loud, f"{len(ids) - len(loud)} of {len(ids)} cuts at least {-CUT_DB_MAX:.0f} dB below "
                                              f"their piece's loudest; needs all")}
    checks = {
        "S1 median": (s1.hi <= P50_MAX, f"first-sound p50 ratio {s1}; needs upper end <= {P50_MAX}"),
        "S2 tail": (s2.hi <= P90_MAX, f"first-sound p90 ratio {s2}; needs upper end <= {P90_MAX}"),
        "S3 stalls": (n_t <= n_b + STALL_EXTRA_MAX and max(stall_t) <= STALL_LONGEST_MS,
                      f"{n_t} trimmed turns stall against {n_b} as is (max {n_b + STALL_EXTRA_MAX}), the longest "
                      f"{max(stall_t):.0f} ms (max {STALL_LONGEST_MS:.0f})"),
    }
    valid = all(ok for ok, _ in v.values())
    failed = [name for name, (ok, _) in checks.items() if not ok]
    verdict = ("NOT A VALID TEST: " + ", ".join(k for k, (ok, _) in v.items() if not ok) if not valid
               else "SUPPORTED: all three criteria hold" if not failed else "NOT SUPPORTED: failed " + ", ".join(failed))
    p = lambda values, q: float(np.percentile(values, q))                         # noqa: E731
    cohorts: dict = {}
    for t, b in zip(T, back):
        cohorts.setdefault(t.get("cohort", "?"), []).append((t["first_sound_ms"], b["first_sound_ms"]))
    early = [c[0][0] + c[0][1] / OUT_SR * 1000 - c[1][0] for c in (t["chunks"] for t in T) if len(c) > 1]
    describe = {
        "turns": len(ids),
        "first_sound": {"trimmed": (p(trimmed, 50), p(trimmed, 90)), "as_is": (p(before, 50), p(before, 90))},
        "first_audio_p50": p([t["first_audio_ms"] for t in T], 50),
        "trim_ms": (p([t["trim_ms"] for t in T], 50), p([t["trim_ms"] for t in T], 90)),
        "loudest_cut_db": max((t["cut_peak_db"] for t in T if t.get("cut_peak_db") is not None), default=None),
        "second_piece_ahead_ms": (p(early, 50), min(early)) if early else None,
        "single_piece": len(T) - len(early),
        "cohort_first_sound_p50": {c: (p([x[0] for x in xs], 50), p([x[1] for x in xs], 50)) for c, xs in cohorts.items()},
        "stalls": {"trimmed": sorted((x for x in stall_t if x > 0), reverse=True),
                   "as_is": sorted((x for x in stall_b if x > 0), reverse=True)},
    }
    return {"valid": valid, "v": v, "checks": checks, "verdict": verdict, "describe": describe}


def format_pr018(blob: dict, result: dict) -> str:
    """The text recorded under PR-018's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                                     # noqa: E731
    fs = d["first_sound"]
    lines = [f"PR-018: {RUNG} (R1, its reply's first piece cut to 20 ms before its first frame within 50 dB) against R1 as "
             "it is on the same turns",
             f"  commit {blob.get('commit')} · {blob.get('machine')}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", ""]
    lines += [f"{name:<22} {mark(ok)}  {detail}" for name, (ok, detail) in result["v"].items()]
    lines += [f"{name:<22} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    ahead = d["second_piece_ahead_ms"]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              f"  first sound, ms, p50 / p90: trimmed {fs['trimmed'][0]:.0f} / {fs['trimmed'][1]:.0f}; as is "
              f"{fs['as_is'][0]:.0f} / {fs['as_is'][1]:.0f}; first audio p50 {d['first_audio_p50']:.0f}",
              f"  the trim cut {d['trim_ms'][0]:.0f} ms at the median, {d['trim_ms'][1]:.0f} at p90; the loudest cut "
              f"{d['loudest_cut_db']} dB against its piece's loudest",
              ("  the second piece ahead of when playback needed it, after the trim: p50 "
               f"{ahead[0]:.0f} ms, least {ahead[1]:.0f}" if ahead else "  no reply had a second piece")
              + f"; {d['single_piece']} replies a single piece",
              "  first-sound p50 by cohort, trimmed vs as is: " + ", ".join(
                  f"{c} {a:.0f} vs {b:.0f}" for c, (a, b) in d["cohort_first_sound_p50"].items()),
              f"  stalls: trimmed {len(d['stalls']['trimmed'])} turns, as is {len(d['stalls']['as_is'])}"]
    return "\n".join(lines)


def save(blob: dict, path: pathlib.Path = RUN) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path
