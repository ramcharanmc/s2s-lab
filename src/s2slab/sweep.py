"""
Policy x cohort sweep: the tradeoff curve, sliced by who is talking.

THE MEASUREMENT BUG THIS FILE EXISTS TO AVOID.

Earlier we reported a median latency of -798 ms for T1 on the disfluent
cohort. That number is meaningless. It mixes two different outcomes:

    +500 ms   waited half a second after they finished    (dead air)
   -2000 ms   started talking two seconds before they did (interruption)

Averaging those produces a figure that describes neither. A cutoff is not
"negative latency" -- it is a different kind of failure, with a different cost
and a different fix. So this module reports THREE numbers per cell:

    cutoff_rate   how often we interrupted           (over ALL turns)
    latency       percentiles over the turns we got RIGHT
    no_fire_rate  how often no decision was reached at all

Only the last two are latencies. Quoting latency without cutoff rate beside it
is how a policy that interrupts everybody gets reported as fast.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from dataclasses import dataclass, field

from .policies import POLICIES, EndpointPolicy
from .simulate import Utterance, make_all
from .timing import LatencySummary, summarize
from .stats import Comparison, adjust, compare, wilson_interval

@dataclass
class PolicyResult:
    """One policy against one cohort. Three outcomes, kept separate."""

    policy: str
    cohort: str
    n: int
    n_cutoff: int
    n_never_fired: int
    latency: LatencySummary        # over correctly-endpointed turns ONLY
        # One entry per turn, in cohort order: did this policy interrupt that turn?
    # Paired tests compare two of these entry by entry, so the ORDER is the
    # contract -- entry i must be the same turn for every policy.
    cut_mask: tuple[bool, ...] = field(default=(), repr=False, compare=False)

    @property
    def cutoff_rate(self) -> float:
        """Over ALL turns. Excluding the ones we failed on would flatter it."""
        return self.n_cutoff / self.n if self.n else float("nan")

    @property
    def no_fire_rate(self) -> float:
        return self.n_never_fired / self.n if self.n else float("nan")

    @property
    def n_scored(self) -> int:
        """Turns the latency figure is actually built from."""
        return self.n - self.n_cutoff - self.n_never_fired

    def as_dict(self) -> dict:
        return {
            "policy": self.policy, "cohort": self.cohort, "n": self.n,
            "cutoff_rate": self.cutoff_rate, "no_fire_rate": self.no_fire_rate,
            "n_scored": self.latency.n,
            "latency_p50": self.latency.p50,
            "latency_p90": self.latency.p90,
            "latency_p99": self.latency.p99,
            "latency_unsupported": self.latency.unsupported,
        }
    
def evaluate(policy: EndpointPolicy, utterances: list[Utterance]) -> PolicyResult:
    """Run one policy over one cohort, sorting turns into three buckets.

    A turn contributes to EXACTLY ONE of: interrupted, never-decided, or
    correctly endpointed. Only the last carries a latency, because only there
    does "how long after they finished" mean anything.
    """
    latencies: list[float] = []
    mask: list[bool] = []                 # one entry per turn, in cohort order
    n_cutoff = n_never_fired = 0

    for u in utterances:
        d = policy.decide(u.context())
        cut = d.fired and d.endpoint_s < u.true_end_s
        mask.append(cut)
        if not d.fired:
            n_never_fired += 1
        elif cut:
            n_cutoff += 1
        else:
            latencies.append((d.endpoint_s - u.true_end_s) * 1000)

    return PolicyResult(
        policy=policy.name,
        cohort=utterances[0].cohort if utterances else "?",
        n=len(utterances),
        n_cutoff=n_cutoff,
        n_never_fired=n_never_fired,
        latency=summarize(latencies),
        cut_mask=tuple(mask),
    )


def run_sweep(n_per_cohort: int = 60, seed: int = 0,
              policies: dict[str, EndpointPolicy] | None = None,
              split: str = "dev", turns: str = "prefix") -> list[PolicyResult]:
    """Every policy against every cohort. Deterministic given `seed`, `split` and `turns`."""
    cohorts = make_all(n_per_cohort, seed, split=split, turns=turns)
    pols = policies if policies is not None else POLICIES
    return [evaluate(p, cohorts[c])
            for c in sorted(cohorts) for p in pols.values()]

# The questions the comparisons answer. Written down here BEFORE the run they
# are confirmed on (seed 1000), because choosing comparisons after seeing the
# results is exactly the freedom Holm cannot correct for. Kept short on
# purpose: Holm multiplies p-values by roughly the number of comparisons, so
# each one added makes every other one harder to pass.
COMPARISONS: list[tuple[str, str]] = [
    ("T0_fixed_500ms", "T1_semantic_500ms"),    # does syntax help at a matched threshold?
    ("T1_semantic_500ms", "T2_prosodic_500ms"), # does pitch change WHO gets interrupted?
    ("T0_fixed_700ms", "T2_prosodic_500ms"),    # T2 against the nearest fixed threshold
]

# PR-001's family (docs/preregistration.md): T1 against T1b, in each cohort: 3 comparisons, nothing else.
PR001_COMPARISONS: list[tuple[str, str]] = [("T1_semantic_500ms", "T1b_complete_500ms")]
PR001_RUN = dict(split="eval", turns="sentence", seed=2000, n=250)      # the registration, with its Deviation 1
PR001_OUT = "runs/turn-taking/pr001.json"


def pr001_criteria(results: list[PolicyResult], comps: list[Comparison]) -> dict:
    """PR-001's success criteria, S1 to S4, from one sweep of T1 and T1b and their comparisons."""
    by = {(r.policy, r.cohort): r for r in results}
    t1, t1b = "T1_semantic_500ms", "T1b_complete_500ms"
    dis = next(c for c in comps if c.cohort == "disfluent" and (c.policy_a, c.policy_b) == (t1, t1b))
    clean_a, clean_b = by[(t1, "clean")].latency, by[(t1b, "clean")].latency
    d50, d90 = clean_b.p50 - clean_a.p50, clean_b.p90 - clean_a.p90
    s = {"S1": dis.diff >= 0.10,                                      # at least 10 pp fewer disfluent cut-offs
         "S2": dis.ci[0] > 0 and dis.p_adjusted < 0.05,               # the CI above 0, and Holm p < 0.05
         "S3": d50 <= 50 and d90 <= 100,                              # clean wait: +50 ms median, +100 ms p90
         "S4": all(by[(t1b, c)].n_cutoff <= by[(t1, c)].n_cutoff for c in ("clean", "slow"))}
    return {"disfluent_diff": dis.diff, "disfluent_ci": list(dis.ci), "disfluent_p_holm": dis.p_adjusted,
            "clean_p50_delta_ms": d50, "clean_p90_delta_ms": d90,
            **{k: bool(v) for k, v in s.items()}, "supported": all(s.values())}


def format_pr001(crit: dict) -> str:
    lo, hi = crit["disfluent_ci"]
    rows = [("S1 · effect", f"disfluent cut-offs {crit['disfluent_diff'] * 100:+.1f} pp (T1 minus T1b); needs >= +10", crit["S1"]),
            ("S2 · confidence", f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}] pp, Holm p {_fmt_p(crit['disfluent_p_holm'])}; needs CI above 0 and p < 0.05", crit["S2"]),
            ("S3 · speed", f"clean median {crit['clean_p50_delta_ms']:+.0f} ms, p90 {crit['clean_p90_delta_ms']:+.0f} ms; needs <= +50 and <= +100", crit["S3"]),
            ("S4 · safety", "clean and slow cut-offs not above T1's (guaranteed by construction)", crit["S4"])]
    lines = ["### PR-001's criteria", "", "| criterion | measured | met |", "|---|---|---|"]
    lines += [f"| {k} | {m} | {'yes' if ok else 'no'} |" for k, m, ok in rows]
    lines += ["", f"**PR-001: {'supported' if crit['supported'] else 'not supported'}**"]
    return "\n".join(lines)


def run_comparisons(results: list[PolicyResult],
                    pairs: list[tuple[str, str]] = COMPARISONS,
                    seed: int = 0) -> list[Comparison]:
    """Paired A-vs-B comparison for every declared pair, in every cohort.

    Takes:   the list run_sweep() returned -- each result carries its cut_mask.
    Calls:   stats.compare() once per (cohort, pair), then stats.adjust() ONCE
             over all of them: 3 pairs x 3 cohorts = a family of 9.
    Returns: a list of Comparison with p_adjusted filled in.
    """
    by_key = {(r.policy, r.cohort): r for r in results}
    comps: list[Comparison] = []
    for cohort in sorted({r.cohort for r in results}):
        for a, b in pairs:
            ra, rb = by_key.get((a, cohort)), by_key.get((b, cohort))
            if ra is None or rb is None:
                continue                 # the pair names a policy that was not swept
            comps.append(compare(cohort, a, ra.cut_mask, b, rb.cut_mask, seed=seed))
    return adjust(comps)


def pareto_frontier(results: list[PolicyResult]) -> list[PolicyResult]:
    """The policies nothing else beats on both axes at once.

    A policy is DOMINATED if another is at least as fast and at least as safe,
    and strictly better on one. What survives is the real menu of choices --
    everything else is a worse version of something on the list.

    Results whose p50 the sample size cannot support are excluded. A policy
    that interrupted 97% of speakers has a "latency" built from two turns -- a
    real number, so nothing dominates it, so it lands on the frontier looking
    like a fast option. Reusing the sample-size rule from timing.py is what
    stops the frontier recommending a policy that mostly fails.    
    """
   

    usable = [r for r in results if "p50" not in r.latency.unsupported]

    out = []
    for r in usable:
        if not any(
            o is not r
            and o.latency.p50 <= r.latency.p50
            and o.cutoff_rate <= r.cutoff_rate
            and (o.latency.p50 < r.latency.p50 or o.cutoff_rate < r.cutoff_rate)
            for o in usable
        ):
            out.append(r)
    return sorted(out, key=lambda r: r.latency.p50)

def format_cohort(results: list[PolicyResult], cohort: str) -> str:
    rows = sorted([r for r in results if r.cohort == cohort],
                  key=lambda r: r.cutoff_rate)
    frontier = {id(r) for r in pareto_frontier(rows)}
    lines = [f"### {cohort}", "",
             "| policy | cut% [95% CI] | no-fire% | p50 ms | p90 ms | n scored | |",
             "|---|---:|---:|---:|---:|---:|:--|"]
    for r in rows:
        p50 = "—" if r.latency.n == 0 else f"{r.latency.p50:.0f}"
        p90 = "—" if "p90" in r.latency.unsupported else f"{r.latency.p90:.0f}"
        mark = "**frontier**" if id(r) in frontier else ""
        lo, hi = wilson_interval(r.n_cutoff, r.n)
        cut = f"{r.cutoff_rate:.0%} [{lo:.0%}–{hi:.0%}]"
        lines.append(f"| {r.policy} | {cut} | {r.no_fire_rate:.0%} "
                     f"| {p50} | {p90} | {r.latency.n} | {mark} |")
    return "\n".join(lines)

def _fmt_p(p: float) -> str:
    """'<0.001' rather than '0.000' -- a p-value is never actually zero."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def format_comparisons(comps: list[Comparison], alpha: float = 0.05) -> str:
    """Markdown table of the paired comparisons.

    Takes:   the list run_comparisons() returned (p_adjusted already filled in).
    Returns: one markdown string, one row per cohort x pair.
    """
    lines = [f"### Paired comparisons (Holm-adjusted across {len(comps)})", "",
             "| cohort | A | B | cut A | cut B | only A / only B | p | p Holm "
             "| A−B [95% CI] | verdict |",
             "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for c in comps:
        lines.append(
            f"| {c.cohort} | {c.policy_a} | {c.policy_b} "
            f"| {c.rate_a:.0%} | {c.rate_b:.0%} "
            f"| {c.test.only_a} / {c.test.only_b} "
            f"| {_fmt_p(c.test.p)} | {_fmt_p(c.p_adjusted)} " 
            f"| {c.diff * 100:+.0f} pp [{c.ci[0] * 100:+.0f}, {c.ci[1] * 100:+.0f}] "
            f"| {c.verdict(alpha)} |")
    return "\n".join(lines)


def _git_commit() -> str:
    """Stamp results with the code that produced them.

    Silent changes to a benchmark are how people accidentally claim
    improvements. A results file that cannot name its own commit is not
    evidence of anything.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True).strip()
        # Uncommitted changes mean no commit contains the code that ran.
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def write_results(results: list[PolicyResult], n_per_cohort: int, seed: int,
                  path: str = "results/sweep.json",
                  comparisons: list[Comparison] | None = None,
                  split: str = "dev", registration: str = "", turns: str = "prefix",
                  criteria: dict | None = None) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "commit": _git_commit(),
        "n_per_cohort": n_per_cohort,
        "seed": seed,
        "split": split,
        "turns": turns,
        "registration": registration or None,
        "results": [r.as_dict() for r in results],
        "comparisons": [
            {"cohort": c.cohort, "a": c.policy_a, "b": c.policy_b, "n": c.n,
             "rate_a": c.rate_a, "rate_b": c.rate_b,
             "only_a": c.test.only_a, "only_b": c.test.only_b,
             "p": c.test.p, "p_holm": c.p_adjusted,
             "diff": c.diff, "ci": list(c.ci), "verdict": c.verdict()}
            for c in (comparisons or [])
        ],
        **({"criteria": criteria} if criteria is not None else {}),
    }, indent=2))
    return p



def main(n: int = 60, seed: int = 0, split: str = "dev",
         registration: str = "", turns: str = "prefix") -> None:
    pr001 = registration == "PR-001"
    if pr001:                                   # T1 and T1b only, PR-001's family, its own file
        from .completion import T1B_THETA, t1b
        policies = {p.name: p for p in (POLICIES["T1_semantic_500ms"], t1b())}
        results = run_sweep(n, seed, policies=policies, split=split, turns=turns)
        comps = run_comparisons(results, PR001_COMPARISONS, seed=seed)
    else:
        results = run_sweep(n, seed, split=split, turns=turns)
        comps = run_comparisons(results, seed=seed)
    for cohort in sorted({r.cohort for r in results}):
        print(format_cohort(results, cohort))
        print()
    print(format_comparisons(comps))
    print()
    crit = pr001_criteria(results, comps) if pr001 else None
    if crit is not None:
        print(format_pr001(crit))
        print(f"\nT1b: cut-off {T1B_THETA} (frozen), model {t1b().scorer.model_name}")
        print()
    out = write_results(results, n, seed, comparisons=comps,
                        split=split, registration=registration, turns=turns, criteria=crit,
                        **({"path": PR001_OUT} if pr001 else {}))
    print(f"commit {_git_commit()} · seed {seed} · n={n}/cohort · "
          f"sentences {split} · turns {turns} → {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--turns", default="prefix", choices=["prefix", "sentence"],
                    help="prefix: the first words of a sentence (every published run); "
                         "sentence: one whole sentence a turn (PR-001, Deviations)")
    ap.add_argument("--confirm", default="",
                    help="registration this run confirms; required with --split eval")
    a = ap.parse_args()
    if a.split == "eval" and not a.confirm:
        ap.error("--split eval is reserved for a registered confirmation run "
                 "(docs/preregistration.md). Pass e.g. --confirm PR-001.")
    if a.confirm == "PR-001" and a.turns != "sentence":
        ap.error("PR-001's confirmation uses whole-sentence turns (its Deviations): "
                 "add --turns sentence")
    if a.confirm == "PR-001" and (a.split, a.seed, a.n) != (PR001_RUN["split"], PR001_RUN["seed"], PR001_RUN["n"]):
        ap.error("PR-001's confirmation is --split eval --turns sentence --seed 2000 --n 250 (its registration)")
    if a.confirm == "PR-001" and pathlib.Path(PR001_OUT).exists():
        ap.error(f"{PR001_OUT} exists: PR-001's confirmation runs once")
    main(a.n, a.seed, a.split, a.confirm, a.turns)