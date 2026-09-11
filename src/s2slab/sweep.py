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

@dataclass
class PolicyResult:
    """One policy against one cohort. Three outcomes, kept separate."""

    policy: str
    cohort: str
    n: int
    n_cutoff: int
    n_never_fired: int
    latency: LatencySummary        # over correctly-endpointed turns ONLY

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
    n_cutoff = n_never_fired = 0

    for u in utterances:
        d = policy.decide(u.context())
        if not d.fired:
            n_never_fired += 1
        elif d.endpoint_s < u.true_end_s:
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
    )


def run_sweep(n_per_cohort: int = 60, seed: int = 0,
              policies: dict[str, EndpointPolicy] | None = None
              ) -> list[PolicyResult]:
    """Every policy against every cohort. Deterministic given `seed`."""
    cohorts = make_all(n_per_cohort, seed)
    pols = policies if policies is not None else POLICIES
    return [evaluate(p, cohorts[c])
            for c in sorted(cohorts) for p in pols.values()]

def pareto_frontier(results: list[PolicyResult]) -> list[PolicyResult]:
    """The policies nothing else beats on both axes at once.

    A policy is DOMINATED if another is at least as fast and at least as safe,
    and strictly better on one. What survives is the real menu of choices --
    everything else is a worse version of something on the list.

    Results with no scored turns are excluded: a policy that interrupted every
    single speaker has no latency, and admitting it to the frontier on a NaN
    would make it look unbeatable.
    """
   

    """ Results whose p50 the sample size cannot support are excluded. A policy that
    interrupted 97% of speakers has a "latency" built from two turns -- a real
    number, so nothing dominates it, so it lands on the frontier looking like a
    fast option. Reusing the sample-size rule from timing.py is what stops the
    frontier recommending a policy that mostly fails. """

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
             "| policy | cut% | no-fire% | p50 ms | p90 ms | n scored | |",
             "|---|---:|---:|---:|---:|---:|:--|"]
    for r in rows:
        p50 = "—" if r.latency.n == 0 else f"{r.latency.p50:.0f}"
        p90 = "—" if "p90" in r.latency.unsupported else f"{r.latency.p90:.0f}"
        mark = "**frontier**" if id(r) in frontier else ""
        lines.append(f"| {r.policy} | {r.cutoff_rate:.0%} | {r.no_fire_rate:.0%} "
                     f"| {p50} | {p90} | {r.latency.n} | {mark} |")
    return "\n".join(lines)


def _git_commit() -> str:
    """Stamp results with the code that produced them.

    Silent changes to a benchmark are how people accidentally claim
    improvements. A results file that cannot name its own commit is not
    evidence of anything.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def write_results(results: list[PolicyResult], n_per_cohort: int, seed: int,
                  path: str = "results/sweep.json") -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "commit": _git_commit(),
        "n_per_cohort": n_per_cohort,
        "seed": seed,
        "results": [r.as_dict() for r in results],
    }, indent=2))
    return p



def main(n: int = 60, seed: int = 0) -> None:
    results = run_sweep(n, seed)
    for cohort in sorted({r.cohort for r in results}):
        print(format_cohort(results, cohort))
        print()
    out = write_results(results, n, seed)
    print(f"commit {_git_commit()} · seed {seed} · n={n}/cohort → {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.seed)