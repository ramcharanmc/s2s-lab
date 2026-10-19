"""
PR-019's analysis: v1 §8's P1b on real disfluent speech. Does the prosodic endpointer, T2, dominate the fixed
threshold on DisfluencySpeech's test split at an operating point that cuts off under 5% of turns?

Every definition here is fixed by docs/preregistration.md (PR-019), committed with this file and before any test
turn was read. The setting is DEV's (runs/turn-taking/p1b-dev.json), by the rule written in the entry: T2 at 900 ms
with a 20 Hz/s slope threshold, against T0 at the same 900 ms.

    PYTHONPATH=src python -m s2slab.pr019 --confirm PR-019      once, on the laptop; saves runs/turn-taking/pr019.json

    S1  T2's cut-off rate under 5%: the upper end of its Wilson 95% interval below 0.05
    S2  T2 cuts off no more turns than T0 on the same turns (the turns only T2 cuts <= those only T0 cuts)
    S3  T2 strictly better, shown: fewer cut-offs by an exact McNemar test (p < 0.05), OR a lower median wait on
        the turns both end correctly, the paired bootstrap 95% interval of T2's over T0's entirely below 1
    Supported only if S1, S2 and S3 all hold. A turn neither ends is a deviation, not a drop.

THE BOOTSTRAP IS PAIRED, as compare.py's: 10,000 resamples of the turns, seed 0.
"""

from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

from .compare import N_BOOT, SEED, percentile_ratio, resamples

REGISTRATION = "PR-019"
RUN = pathlib.Path("runs/turn-taking/pr019.json")
BASE_MS, SLOPE = 900, 20.0              # DEV's choice (the entry's rule)
CUT_MAX, ALPHA = 0.05, 0.05             # S1; S3's McNemar


def policies() -> dict:
    """{"T2": the registered prosodic setting, "T0": the fixed threshold it is held against}."""
    from .policies import FixedThreshold, ProsodicThreshold
    t2 = ProsodicThreshold(BASE_MS, slope_threshold=SLOPE)
    t2.name = f"T2_prosodic_{BASE_MS}ms_s{SLOPE:g}"
    return {"T2": t2, "T0": FixedThreshold(BASE_MS)}


def decide(policy, u) -> dict:
    """One policy on one turn: where it ended the turn, whether that cut the speaker off, how long after the end."""
    d = policy.decide(u.context())
    cut = bool(d.fired and d.endpoint_s < u.true_end_s)
    return {"fired": bool(d.fired), "endpoint_s": round(float(d.endpoint_s), 4), "cut": cut, "reason": d.reason,
            "wait_ms": None if cut or not d.fired else round((d.endpoint_s - u.true_end_s) * 1000, 1)}


def run(items: list) -> dict:
    """Both policies on every (id, Utterance), and the whole DEV sweep beside them (descriptive)."""
    from .dsturns import dev_policies, report
    pols = policies()
    turns = {i: {"true_end_s": round(float(u.true_end_s), 4), "pauses_s": [round(float(p), 4) for p in u.mid_turn_pauses_s],
                 **{k: decide(p, u) for k, p in pols.items()}} for i, u in items}
    return {"policies": {k: p.name for k, p in pols.items()}, "n": len(items), "turns": turns,
            "sweep": report(items, dev_policies())}


def criteria(blob: dict, n_boot: int = N_BOOT) -> dict:
    """S1 to S3 and what is reported beside them. Raises ValueError if either policy never ends a turn."""
    from .stats import mcnemar_exact, wilson_interval
    ids = list(blob["turns"])
    T = [blob["turns"][i] for i in ids]
    silent = [i for i, t in zip(ids, T) if not (t["T2"]["fired"] and t["T0"]["fired"])]
    if silent:
        raise ValueError(f"a policy never ended {silent[:5]}: PR-019 drops no turn -- report this as a deviation")
    n = len(T)
    c2, c0 = [t["T2"]["cut"] for t in T], [t["T0"]["cut"] for t in T]
    k2, k0 = sum(c2), sum(c0)
    lo2, hi2 = wilson_interval(k2, n)
    mc = mcnemar_exact(c2, c0)                                   # only_a: only T2 cut; only_b: only T0 cut
    both = [t for t in T if not t["T2"]["cut"] and not t["T0"]["cut"]]
    w2, w0 = [t["T2"]["wait_ms"] for t in both], [t["T0"]["wait_ms"] for t in both]
    wait = percentile_ratio(w2, w0, 50, resamples(len(both), n_boot)) if both else None
    s3_cut = mc.only_b > mc.only_a and mc.p < ALPHA
    s3_wait = wait is not None and wait.hi < 1.0
    checks = {
        "S1 under 5%": (hi2 < CUT_MAX, f"T2 cut off {k2} of {n} ({100 * k2 / n:.1f}%, 95% interval {100 * lo2:.1f} to "
                                       f"{100 * hi2:.1f}%); needs the upper end below {100 * CUT_MAX:.0f}%"),
        "S2 no more cut-offs": (mc.only_a <= mc.only_b, f"only T2 cut {mc.only_a}, only T0 cut {mc.only_b} "
                                                        f"(T0 cut {k0} of {n}); needs only T2 <= only T0"),
        "S3 strictly better": (s3_cut or s3_wait, f"fewer cut-offs: McNemar p = {mc.p:.3f} (needs < {ALPHA} with fewer); "
                                                  f"median wait, T2 over T0 on the {len(both)} turns both end: {wait} "
                                                  f"(needs the upper end below 1)"),
    }
    failed = [k for k, (ok, _) in checks.items() if not ok]
    verdict = "SUPPORTED: all three criteria hold" if not failed else "NOT SUPPORTED: failed " + ", ".join(failed)
    p = lambda v, q: float(np.percentile(v, q)) if v else None                    # noqa: E731
    reasons = collections.Counter(t["T2"]["reason"].split("+", 1)[1] if "+" in t["T2"]["reason"] else "plain"
                                  for t in T if not t["T2"]["cut"])
    describe = {
        "turns": n, "cut": {"T2": k2, "T0": k0}, "only": {"T2": mc.only_a, "T0": mc.only_b},
        "wait": {"T2": (p(w2, 50), p(w2, 90)), "T0": (p(w0, 50), p(w0, 90))},
        "t2_at_the_end": dict(sorted(reasons.items())),
        "cut_ids": {"T2": [i for i, t in zip(ids, T) if t["T2"]["cut"]], "T0": [i for i, t in zip(ids, T) if t["T0"]["cut"]]},
        "longest_pause_p50_s": p([max(t["pauses_s"], default=0.0) for t in T], 50),
    }
    return {"checks": checks, "verdict": verdict, "describe": describe,
            "values": {"cut_hi": hi2, "mcnemar_p": mc.p, "wait_ratio": None if wait is None else [wait.value, wait.lo, wait.hi]}}


def format_pr019(blob: dict, result: dict) -> str:
    """The text recorded under PR-019's Results, verbatim."""
    d = result["describe"]
    mark = lambda ok: "PASS" if ok else "FAIL"                                     # noqa: E731
    f = lambda v: "—" if v is None else f"{v:.0f}"                                 # noqa: E731
    pol = blob["policies"]
    lines = [f"PR-019: {pol['T2']} against {pol['T0']}, DisfluencySpeech's test split",
             f"  commit {blob.get('commit')} · laptop CPU · {d['turns']} turns",
             f"  paired bootstrap: {N_BOOT} resamples of the turns both end, seed {SEED}", ""]
    lines += [f"{name:<22} {mark(ok)}  {detail}" for name, (ok, detail) in result["checks"].items()]
    lines += ["", result["verdict"], "", "Descriptive, not criteria:",
              f"  cut off: T2 {d['cut']['T2']}, T0 {d['cut']['T0']} of {d['turns']}; only T2 {d['only']['T2']}, only T0 {d['only']['T0']}",
              f"  wait, ms, p50 / p90 on the turns both end: T2 {f(d['wait']['T2'][0])} / {f(d['wait']['T2'][1])}; "
              f"T0 {f(d['wait']['T0'][0])} / {f(d['wait']['T0'][1])}",
              "  what T2 read at the end of the turns it did not cut: "
              + ", ".join(f"{k} {v}" for k, v in d["t2_at_the_end"].items()),
              f"  the turns' longest pause, the median: {1000 * d['longest_pause_p50_s']:.0f} ms",
              "", "The whole DEV sweep on the test turns:"]
    for r in blob["sweep"]["rows"]:
        lines.append(f"  {r['policy']:<24} cut {r['cut']:>3} of {r['n']}  p50 {f(r['p50']):>5}  p90 {f(r['p90']):>5}")
    return "\n".join(lines)


def save(blob: dict, path: pathlib.Path = RUN) -> pathlib.Path:
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> None:
    import argparse

    from .dsturns import turns
    from .sweep import _git_commit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--confirm", default="")
    a = ap.parse_args(argv)
    if a.confirm != REGISTRATION:
        raise SystemExit(f"add --confirm {REGISTRATION}: the test turns are its confirmation's, and it runs once")
    if RUN.exists():
        raise SystemExit(f"{RUN} exists: PR-019 runs once")
    commit = _git_commit()
    if commit.endswith("-dirty"):
        raise SystemExit(f"{commit}: commit first (git status must be empty); PR-019 runs from a clean commit")
    blob = {"registration": REGISTRATION, "commit": commit, "split": "test", **run(turns("test", confirm=REGISTRATION))}
    path = save(blob)
    print(format_pr019(blob, criteria(blob)))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
