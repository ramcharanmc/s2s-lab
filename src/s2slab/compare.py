"""
PR-002's analysis: a streaming rung against the batch rung it was built from.

Every definition here is fixed by docs/preregistration.md (PR-002), which was
committed before this file existed. The script has no choices left to make: it
reads two saved eval-set runs -- same clips, same container -- and says which
of V3 and S1-S5 hold.

    python -m s2slab.compare runs/r0-gpu/evalset-modal-t4-pr002.json \
                             runs/r1-gpu/evalset-modal-t4-pr002.json

    V3  same words on >= 190 of 200 turns        else: not a valid test
    S1  first-audio p50 ratio <= 0.75            upper end of the 95% interval
    S2  first-audio p90 ratio <= 0.70            upper end of the 95% interval
    S3  r(first audio, reply words) < 0.3, and its interval below the batch rung's r
    S4  total p50 ratio <= 1.15                  upper end of the 95% interval
    S5  <= 20 turns stall at all, <= 10 stall for more than 200 ms

THE BOOTSTRAP IS PAIRED. Each resample draws 200 clip ids with replacement and
recomputes BOTH rungs' percentiles on those same clips, so a clip that is slow
for both rungs moves both numbers together and drops out of the ratio. 10,000
resamples, seed 0, percentile interval -- as registered.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

N_BOOT, SEED = 10_000, 0
SAME_WORDS_MIN = 190                       # V3, of 200
P50_MAX, P90_MAX, TOTAL_MAX = 0.75, 0.70, 1.15
R_MAX = 0.3
STALL_TURNS_MAX, LONG_STALL_TURNS_MAX, LONG_STALL_MS = 20, 10, 200.0


@dataclass
class Estimate:
    """A statistic on all the clips, and its paired-bootstrap 95% interval."""

    value: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.value:.3f} [95% CI {self.lo:.3f}, {self.hi:.3f}]"



def resamples(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n) clip indices, drawn with replacement. One draw serves every
    statistic, so all of them are computed on the same resampled clips."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def percentile_ratio(a, b, q: float, idx: np.ndarray) -> Estimate:
    """percentile_q(a) / percentile_q(b), with its interval.

    Takes: two arrays aligned clip by clip (a = streaming, b = batch), the
           percentile, and the resample indices.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    boot = np.percentile(a[idx], q, axis=1) / np.percentile(b[idx], q, axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Estimate(float(np.percentile(a, q) / np.percentile(b, q)), float(lo), float(hi))


def correlation(x, y, idx: np.ndarray) -> Estimate:
    """Pearson r between x and y, with its interval. A resample where either
    side has no spread has no r; it is left out of the interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(axis=1, keepdims=True)
    ys = ys - ys.mean(axis=1, keepdims=True)
    denom = np.sqrt((xs ** 2).sum(axis=1) * (ys ** 2).sum(axis=1))
    boot = (xs * ys).sum(axis=1)[denom > 0] / denom[denom > 0]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Estimate(float(np.corrcoef(x, y)[0, 1]), float(lo), float(hi))


def words(turn: dict) -> list[str]:
    return (turn["reply_text"] or "").split()



def compare(batch: dict, stream: dict, n_boot: int = N_BOOT) -> dict:
    """Everything PR-002 asks of two runs.

    Takes:   the batch rung's results dict and the streaming rung's (evalrun.run).
    Returns: {"checks": {name: (passed, detail)}, "valid": V3, "verdict": str,
              "describe": {...}} -- numbers only; format_report() turns it into text.
    Raises:  ValueError if the runs are not over the same clips, or a turn in
             either has no reply audio (first audio undefined: PR-002 drops no clip).
    """
    ids = list(batch["turns"])
    if ids != list(stream["turns"]):
        raise ValueError("the two runs are not over the same clips in the same order")
    b = [batch["turns"][i] for i in ids]
    s = [stream["turns"][i] for i in ids]
    silent = [i for i, bt, st in zip(ids, b, s)
              if bt.get("first_audio_ms") is None or st.get("first_audio_ms") is None]
    if silent:
        raise ValueError(f"no reply audio for {silent[:5]}: first audio is undefined, and "
                         "PR-002 drops no clip -- report this as a deviation")
    idx = resamples(len(ids), n_boot)

    same = sum(words(bt) == words(st) for bt, st in zip(b, s))
    s_first = [t["first_audio_ms"] for t in s]
    b_first = [t["first_audio_ms"] for t in b]          # a batch rung's first audio is its total
    s1 = percentile_ratio(s_first, b_first, 50, idx)
    s2 = percentile_ratio(s_first, b_first, 90, idx)
    r_stream = correlation([len(words(t)) for t in s], s_first, idx)
    r_batch = correlation([len(words(t)) for t in b], [t["total_ms"] for t in b], idx)
    s4 = percentile_ratio([t["total_ms"] for t in s], [t["total_ms"] for t in b], 50, idx)
    stalls = [t["stall_ms"] for t in s]
    n_stall = sum(v > 0 for v in stalls)
    n_long = sum(v > LONG_STALL_MS for v in stalls)

    checks = {
        "S1 median": (s1.hi <= P50_MAX, f"first-audio p50 ratio {s1}; needs upper end <= {P50_MAX}"),
        "S2 tail": (s2.hi <= P90_MAX, f"first-audio p90 ratio {s2}; needs upper end <= {P90_MAX}"),
        "S3 reply length": (r_stream.value < R_MAX and r_stream.hi < r_batch.value,
                            f"r(first audio, reply words) {r_stream}; needs r < {R_MAX} and upper "
                            f"end below the batch rung's r(total, reply words) = {r_batch.value:.3f}"),
        "S4 speed guardrail": (s4.hi <= TOTAL_MAX, f"total p50 ratio {s4}; needs upper end <= {TOTAL_MAX}"),
        "S5 stall guardrail": (n_stall <= STALL_TURNS_MAX and n_long <= LONG_STALL_TURNS_MAX,
                               f"{n_stall} turns stall (max {STALL_TURNS_MAX}), {n_long} for more "
                               f"than {LONG_STALL_MS:.0f} ms (max {LONG_STALL_TURNS_MAX})"),
    }
    valid = same >= SAME_WORDS_MIN
    failed = [name for name, (ok, _) in checks.items() if not ok]
    verdict = ("NOT A VALID TEST (V3 failed): timing is confounded by different words" if not valid
               else "SUPPORTED: all five criteria hold" if not failed
               else "NOT SUPPORTED: failed " + ", ".join(failed))

    pieces = [len(t["chunks"]) for t in s]
    seams = [(st["reply_s"] - bt["reply_s"]) / (n - 1)
             for bt, st, n in zip(b, s, pieces) if n > 1 and words(bt) == words(st)]
    by_pieces = {}
    for n, st, bt in zip(pieces, s, b):
        by_pieces.setdefault(min(n, 4), []).append(st["first_audio_ms"] / bt["first_audio_ms"])
    cohorts = {}
    for bt, st in zip(b, s):
        cohorts.setdefault(bt["cohort"], []).append((st["first_audio_ms"], bt["first_audio_ms"]))

    def p(values, q):
        return float(np.percentile(values, q))

    describe = {
        "turns": len(ids), "same_words": same,
        "first_audio": {"stream": (p(s_first, 50), p(s_first, 90)), "batch": (p(b_first, 50), p(b_first, 90))},
        "first_sound_p50": {"stream": p([t["first_sound_ms"] for t in s if t["first_sound_ms"] is not None], 50),
                            "batch": p([t["first_sound_ms"] for t in b if t["first_sound_ms"] is not None], 50)},
        "total": {"stream": (p([t["total_ms"] for t in s], 50), p([t["total_ms"] for t in s], 90)),
                  "batch": (p([t["total_ms"] for t in b], 50), p([t["total_ms"] for t in b], 90))},
        "cohort_p50": {c: (p([v[0] for v in vs], 50), p([v[1] for v in vs], 50)) for c, vs in cohorts.items()},
        "pieces": {n: (len(v), float(np.median(v))) for n, v in sorted(by_pieces.items())},
        "stalls": sorted((v for v in stalls if v > 0), reverse=True),
        "seam_extra_s": float(np.median(seams)) if seams else None,
        "same_transcripts": sum((bt["transcript"] or "") == (st["transcript"] or "") for bt, st in zip(b, s)),
    }
    return {"valid": valid, "checks": checks, "verdict": verdict, "describe": describe,
            "v3": f"same words on {same} of {len(ids)} turns; needs >= {SAME_WORDS_MIN}"}



def format_report(batch: dict, stream: dict, result: dict) -> str:
    """The text recorded under PR-002's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                       # noqa: E731
    lines = [f"PR-002: {stream['rung']} against {batch['rung']}",
             f"  {batch['rung']}: commit {batch['commit']} · {batch['machine']} · run {batch['ran']}",
             f"  {stream['rung']}: commit {stream['commit']} · {stream['machine']} · run {stream['ran']}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", "",
             f"V3 same words        {mark(result['valid'])}  {result['v3']}"]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              "  ms, p50 / p90        " + f"{stream['rung']:>16} {batch['rung']:>16}"]
    for label, key in (("first audio", "first_audio"), ("total", "total")):
        (s50, s90), (b50, b90) = d[key]["stream"], d[key]["batch"]
        lines.append(f"  {label:<20} {s50:>8.0f} /{s90:>6.0f} {b50:>8.0f} /{b90:>6.0f}")
    lines.append(f"  first sound p50      {d['first_sound_p50']['stream']:>16.0f} {d['first_sound_p50']['batch']:>16.0f}")
    lines.append("  first-audio p50 by cohort: " + ", ".join(
        f"{c} {sv:.0f} vs {bv:.0f}" for c, (sv, bv) in d["cohort_p50"].items()))
    lines.append("  pieces per reply (turns, median first-audio ratio): " + ", ".join(
        f"{n}{'+' if n == 4 else ''}: {k} turns, {r:.2f}" for n, (k, r) in d["pieces"].items()))
    stalls = d["stalls"]
    lines.append(f"  stalls: {len(stalls)} turns" + (f", longest {stalls[0]:.0f} ms, median of those "
                 f"{float(np.median(stalls)):.0f} ms" if stalls else ""))
    if d["seam_extra_s"] is not None:
        lines.append(f"  extra reply audio per seam: {d['seam_extra_s']:.2f} s (median, same-words turns)")
    lines.append(f"  identical transcripts: {d['same_transcripts']} of {d['turns']}")
    return "\n".join(lines)


def wer_lines(runs: list[dict], root=None) -> str:
    """Normalized WER of each run, over the whole set. Needs the eval-set manifest."""
    from .evalset import EVAL_ROOT
    from .wer import ZERO, score
    out = []
    for run in runs:
        scored = score({i: t["transcript"] or "" for i, t in run["turns"].items()}, root or EVAL_ROOT)
        total = sum((row["normalized"] for row in scored.values()), ZERO)
        out.append(f"  WER normalized, {run['rung']}: {100 * total.rate:.1f}%")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("usage: python -m s2slab.compare <batch run>.json <streaming run>.json")
    batch, stream = (json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:])
    print(format_report(batch, stream, compare(batch, stream)))
    print(wer_lines([batch, stream]))