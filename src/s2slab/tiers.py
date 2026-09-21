"""
PR-004's analysis: a slow-tier recognizer (R2) against the fast tier it replaces (R1).

Every definition here is fixed by docs/preregistration.md (PR-004), which was
committed before this file existed. The script has no choices left to make: it
reads saved eval-set runs -- same clips, same container -- and says which of
V1, V3 and S1-S3 hold.

    python -m s2slab.tiers runs/r1-gpu/evalset-modal-t4-pr004.json \
                           runs/r2-gpu-turbo/evalset-modal-t4-pr004.json \
                           runs/r2-gpu/evalset-modal-t4-pr004.json \
                           --previous runs/r1-gpu/evalset-modal-t4-pr002.json

    the first run is R1; the second is the registered slow tier, and gets a
    verdict; any further run is described with the same numbers and no verdict.

    S1  overall normalized WER, slow minus fast   <= -3.0 points, interval's upper end < 0
    S2  the same on the accented cohort           <= -5.0 points, interval's upper end < 0
    S3  the slow tier's first-audio p50           interval's upper end <= 1000 ms

THE BOOTSTRAP IS PAIRED, OVER CLIPS. A resample draws clip ids with replacement
and recomputes BOTH rungs' corpus WER -- total errors over total reference
words -- on those same clips. A clip both tiers find hard moves both numbers
together and drops out of the difference. 10,000 resamples, seed 0, percentile
interval (compare.resamples), all 200 clips for S1 and S3, the 50 accented
clips for S2 -- as registered.

WHAT THE REST DESCRIBES. Where the added wait comes from (the recognizer, or a
reply that changed because the words did); how much the two tiers' errors
overlap (the v2 plan's test for whether fusing them could pay: not above about
70%); how often they agree word for word (what starting the LLM early on the
fast tier's transcript would need); how often the reply changes.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, Estimate, resamples, words

WER_DROP_MIN, ACCENTED_DROP_MIN = 3.0, 5.0      # S1, S2: percentage points
FIRST_AUDIO_MAX_MS = 1000.0                     # S3
FUSION_OVERLAP_MAX = 0.70                       # the plan's rule of thumb; described, not tested
ACCENTED = "accented"


def clip_errors(run: dict, refs: dict, normalizer) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per clip, in the run's own order: errors, reference words, and the
    normalized transcript.

    Takes: a saved run; {clip id: (cohort, reference text)} (wer.references());
           the normalizer both sides go through.
    """
    from .wer import wer
    errors, n_words, heard = [], [], []
    for clip, turn in run["turns"].items():
        e = wer(refs[clip][1], turn["transcript"] or "", normalizer)
        errors.append(e.errors)
        n_words.append(e.n_words)
        heard.append(normalizer(turn["transcript"] or ""))
    return np.array(errors, float), np.array(n_words, float), heard


def wer_difference(fast_errors, slow_errors, n_words, idx: np.ndarray) -> Estimate:
    """Corpus WER of the slow tier minus the fast tier's, in percentage points.

    Takes: errors per clip for each tier and reference words per clip, aligned;
           resample indices INTO those arrays.
    Negative means the slow tier hears more. Example: 3 and 1 errors in 16
    reference words -> 6.25% - 18.75% = -12.5 points.
    """
    f, s, n = (np.asarray(v, float) for v in (fast_errors, slow_errors, n_words))
    boot = 100 * (s[idx].sum(axis=1) - f[idx].sum(axis=1)) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return Estimate(float(100 * (s.sum() - f.sum()) / n.sum()), float(lo), float(hi))


def percentile_ci(values, q: float, idx: np.ndarray) -> Estimate:
    """One rung's percentile_q, with its bootstrap interval."""
    v = np.asarray(values, float)
    lo, hi = np.percentile(np.percentile(v[idx], q, axis=1), [2.5, 97.5])
    return Estimate(float(np.percentile(v, q)), float(lo), float(hi))


def _median(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def analyse(fast: dict, slow: dict, refs: dict, normalizer=None, n_boot: int = N_BOOT) -> dict:
    """Everything PR-004 asks of R1's run and one slow tier's run.

    Takes:   two results dicts (evalrun.run), the references, and the
             normalizer (default: wer.standard, as registered).
    Returns: {"valid": {name: (passed, detail)}, "checks": {name: (passed,
             detail)}, "verdict": str, "describe": {...}} -- numbers only;
             format_report() turns it into text.
    Raises:  ValueError if the runs are not over the same clips, or a turn has
             no reply audio (first audio undefined: PR-004 drops no clip).
    """
    if normalizer is None:
        from .wer import standard as normalizer
    ids = list(fast["turns"])
    if ids != list(slow["turns"]):
        raise ValueError("the two runs are not over the same clips in the same order")
    f, s = [fast["turns"][i] for i in ids], [slow["turns"][i] for i in ids]
    silent = [i for i, ft, st in zip(ids, f, s)
              if ft.get("first_audio_ms") is None or st.get("first_audio_ms") is None]
    if silent:
        raise ValueError(f"no reply audio for {silent[:5]}: first audio is undefined, and "
                         "PR-004 drops no clip -- report this as a deviation")

    f_err, n_words, f_heard = clip_errors(fast, refs, normalizer)
    s_err, _, s_heard = clip_errors(slow, refs, normalizer)
    idx = resamples(len(ids), n_boot)
    acc = np.array([refs[i][0] == ACCENTED for i in ids])
    s1 = wer_difference(f_err, s_err, n_words, idx)
    s2 = (wer_difference(f_err[acc], s_err[acc], n_words[acc], resamples(int(acc.sum()), n_boot))
          if acc.any() else Estimate(float("nan"), float("nan"), float("nan")))
    s3 = percentile_ci([t["first_audio_ms"] for t in s], 50, idx)

    checks = {
        "S1 hears more": (s1.value <= -WER_DROP_MIN and s1.hi < 0,
                          f"normalized WER, slow minus fast: {s1} points; needs <= -{WER_DROP_MIN} "
                          "and upper end < 0"),
        "S2 accented": (s2.value <= -ACCENTED_DROP_MIN and s2.hi < 0,
                        f"accented cohort ({int(acc.sum())} clips): {s2} points; needs <= "
                        f"-{ACCENTED_DROP_MIN} and upper end < 0"),
        "S3 within a second": (s3.hi <= FIRST_AUDIO_MAX_MS,
                               f"first-audio p50 {s3} ms; needs upper end <= {FIRST_AUDIO_MAX_MS:.0f}"),
    }
    commits = {fast["commit"], slow["commit"]}
    valid = {
        "V1 same run": (fast["machine"] == slow["machine"] and "one container" in fast["machine"],
                        f"{fast['machine']}" if fast["machine"] == slow["machine"]
                        else f"{fast['machine']}  vs  {slow['machine']}"),
        "V3 clean tree": (len(commits) == 1 and not any(c.endswith("-dirty") for c in commits),
                          "commit " + ", ".join(sorted(commits))),
    }
    failed = [name for name, (ok, _) in checks.items() if not ok]
    broken = [name for name, (ok, _) in valid.items() if not ok]
    verdict = ("NOT THE REGISTERED RUN: failed " + ", ".join(broken) if broken
               else "SUPPORTED: all three criteria hold" if not failed
               else "NOT SUPPORTED: failed " + ", ".join(failed))

    # Where the added wait comes from, turn by turn.
    d_first = np.array([st["first_audio_ms"] - ft["first_audio_ms"] for ft, st in zip(f, s)])
    d_asr = np.array([st["stages_ms"]["asr"] - ft["stages_ms"]["asr"] for ft, st in zip(f, s)])
    reply_same = np.array([words(ft) == words(st) for ft, st in zip(f, s)])
    raw_same = np.array([(ft["transcript"] or "") == (st["transcript"] or "") for ft, st in zip(f, s)])
    f_wrong, s_wrong = f_err > 0, s_err > 0
    either = int((f_wrong | s_wrong).sum())

    def p(values, q):
        return float(np.percentile(values, q))

    def lat(turns, key):
        return p([t[key] for t in turns], 50), p([t[key] for t in turns], 90)

    describe = {
        "turns": len(ids),
        "wer_overall": (100 * f_err.sum() / n_words.sum(), 100 * s_err.sum() / n_words.sum()),
        "first_audio": {"fast": lat(f, "first_audio_ms"), "slow": lat(s, "first_audio_ms")},
        "first_sound_p50": (p([t["first_sound_ms"] for t in f if t["first_sound_ms"] is not None], 50),
                            p([t["first_sound_ms"] for t in s if t["first_sound_ms"] is not None], 50)),
        "total": {"fast": lat(f, "total_ms"), "slow": lat(s, "total_ms")},
        "asr_p50": (p([t["stages_ms"]["asr"] for t in f], 50), p([t["stages_ms"]["asr"] for t in s], 50)),
        "stalls": (sum(t["stall_ms"] > 0 for t in f), sum(t["stall_ms"] > 0 for t in s)),
        "added_wait": {"first_audio": _median(d_first), "asr": _median(d_asr), "rest": _median(d_first - d_asr),
                       "rest_same_reply": (int(reply_same.sum()), _median((d_first - d_asr)[reply_same])),
                       "rest_new_reply": (int((~reply_same).sum()), _median((d_first - d_asr)[~reply_same]))},
        "overlap": {"both": int((f_wrong & s_wrong).sum()), "only_fast": int((f_wrong & ~s_wrong).sum()),
                    "only_slow": int((~f_wrong & s_wrong).sum()), "neither": int((~f_wrong & ~s_wrong).sum()),
                    "share": (int((f_wrong & s_wrong).sum()) / either) if either else float("nan")},
        "agree": sum(a == b for a, b in zip(f_heard, s_heard)),
        "reply_changed": {"all": int((~reply_same).sum()),
                          "transcript_differs": (int((~raw_same).sum()), int((~reply_same & ~raw_same).sum())),
                          "transcript_same": (int(raw_same.sum()), int((~reply_same & raw_same).sum()))},
    }
    return {"valid": valid, "checks": checks, "verdict": verdict, "describe": describe}


def format_report(fast: dict, slow: dict, result: dict, tested: bool = True) -> str:
    """The text recorded under PR-004's Results, verbatim. tested=False is for a
    slow tier that is described only: the same numbers, no PASS or FAIL, no verdict."""
    d = result["describe"]
    mark = (lambda ok: "PASS" if ok else "FAIL") if tested else (lambda ok: "    ")     # noqa: E731
    head = "PR-004" if tested else "Described only, not tested"
    lines = [f"{head}: {slow['rung']} against {fast['rung']}",
             f"  {fast['rung']}: commit {fast['commit']} · {fast['machine']} · run {fast['ran']}",
             f"  {slow['rung']}: commit {slow['commit']} · {slow['machine']} · run {slow['ran']}",
             f"  paired bootstrap: {N_BOOT} resamples of {d['turns']} clips, seed {SEED}", ""]
    if tested:
        lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["valid"].items()]
    lines += [f"{name:<20} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    if tested:
        lines += ["", result["verdict"]]
    lines += ["", "Descriptive, not criteria:",
              "  ms, p50 / p90        " + f"{slow['rung']:>16} {fast['rung']:>16}"]
    for label, key in (("first audio", "first_audio"), ("total", "total")):
        (s50, s90), (f50, f90) = d[key]["slow"], d[key]["fast"]
        lines.append(f"  {label:<20} {s50:>8.0f} /{s90:>6.0f} {f50:>8.0f} /{f90:>6.0f}")
    lines.append(f"  first sound p50      {d['first_sound_p50'][1]:>16.0f} {d['first_sound_p50'][0]:>16.0f}")
    lines.append(f"  asr p50              {d['asr_p50'][1]:>16.0f} {d['asr_p50'][0]:>16.0f}")
    lines.append(f"  WER normalized       {d['wer_overall'][1]:>15.1f}% {d['wer_overall'][0]:>15.1f}%")
    lines.append(f"  turns that stalled   {d['stalls'][1]:>16} {d['stalls'][0]:>16}")
    w = d["added_wait"]
    (n_same, rest_same), (n_new, rest_new) = w["rest_same_reply"], w["rest_new_reply"]
    fmt = lambda v: "-" if v is None else f"{v:+.0f} ms"                                # noqa: E731
    lines += ["  the added wait, per turn, at the median:",
              f"    first audio {fmt(w['first_audio'])} = recognizer {fmt(w['asr'])} + everything after it {fmt(w['rest'])}",
              f"    everything after it: {fmt(rest_same)} on the {n_same} turns whose reply kept its words, "
              f"{fmt(rest_new)} on the {n_new} whose reply changed"]
    o = d["overlap"]
    lines += [f"  clips with any error: both tiers {o['both']}, only {fast['rung']} {o['only_fast']}, "
              f"only {slow['rung']} {o['only_slow']}, neither {o['neither']}",
              f"    error overlap {100 * o['share']:.0f}% (both / either); the plan's rule: above about "
              f"{100 * FUSION_OVERLAP_MAX:.0f}%, fusing the tiers will not pay"]
    lines.append(f"  the tiers agree word for word (normalized) on {d['agree']} of {d['turns']} clips")
    r = d["reply_changed"]
    (n_diff, ch_diff), (n_eq, ch_eq) = r["transcript_differs"], r["transcript_same"]
    lines.append(f"  the reply's words changed on {r['all']} of {d['turns']} turns: {ch_diff} of the {n_diff} "
                 f"whose transcript differed, {ch_eq} of the {n_eq} whose transcript was identical")
    return "\n".join(lines)


def reproduces(run: dict, previous: dict) -> str:
    """One rung's run against an earlier run of the same rung: did it come out the same?"""
    ids = [i for i in run["turns"] if i in previous["turns"]]
    now, then = [run["turns"][i] for i in ids], [previous["turns"][i] for i in ids]
    same_heard = sum((a["transcript"] or "") == (b["transcript"] or "") for a, b in zip(now, then))
    same_said = sum(words(a) == words(b) for a, b in zip(now, then))
    p50 = [float(np.percentile([t["first_audio_ms"] for t in ts], 50)) for ts in (now, then)]
    diff = float(np.median([a["first_audio_ms"] - b["first_audio_ms"] for a, b in zip(now, then)]))
    return (f"{run['rung']} now (commit {run['commit']}) against its earlier run (commit {previous['commit']}), "
            f"{len(ids)} clips:\n  identical transcripts {same_heard}, identical replies {same_said}; "
            f"first-audio p50 {p50[0]:.0f} vs {p50[1]:.0f} ms, median per-turn difference {diff:+.0f} ms")


def cohort_wer(runs: list[dict], root=None) -> str:
    """Normalized (raw) WER per cohort for each run. Needs the eval-set manifest."""
    from .evalset import EVAL_ROOT
    from .wer import ZERO, score
    rows = []
    for run in runs:
        scored = score({i: t["transcript"] or "" for i, t in run["turns"].items()}, root or EVAL_ROOT)
        cells = [f"{c} {100 * v['normalized'].rate:.1f}% ({100 * v['raw'].rate:.1f}%)" for c, v in scored.items()]
        total = {k: sum((v[k] for v in scored.values()), ZERO) for k in ("normalized", "raw")}
        rows.append(f"  {run['rung']}: " + ", ".join(cells)
                    + f", all {100 * total['normalized'].rate:.1f}% ({100 * total['raw'].rate:.1f}%)")
    return "WER normalized (raw), by cohort:\n" + "\n".join(rows)


if __name__ == "__main__":
    import argparse

    from .wer import references

    ap = argparse.ArgumentParser(description="PR-004: a slow-tier recognizer against R1.")
    ap.add_argument("fast", help="R1's run")
    ap.add_argument("slow", help="the registered slow tier's run: gets the verdict")
    ap.add_argument("others", nargs="*", help="further slow tiers: described, not tested")
    ap.add_argument("--previous", default="", help="an earlier run of R1, to check this one reproduces it")
    a = ap.parse_args()
    load = lambda path: json.loads(pathlib.Path(path).read_text(encoding="utf-8"))      # noqa: E731
    fast, slow, others = load(a.fast), load(a.slow), [load(p) for p in a.others]
    refs = references()
    print(format_report(fast, slow, analyse(fast, slow, refs)))
    for other in others:
        print("\n" + format_report(fast, other, analyse(fast, other, refs), tested=False))
    print("\n" + cohort_wer([fast, slow, *others]))
    if a.previous:
        print("\n" + reproduces(fast, load(a.previous)))