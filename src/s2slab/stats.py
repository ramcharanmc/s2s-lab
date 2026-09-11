"""
Statistics for comparing endpointing policies. Pure Python + numpy.

Every policy in the sweep runs on the SAME turns, so the right comparison is
PAIRED: for each turn, did A interrupt, did B interrupt? Only the turns where
exactly one of them interrupted say anything about which is better.

Four tools, each answering one question:

    wilson_interval        how uncertain is ONE interruption rate?
    mcnemar_exact          do TWO policies differ on the same turns?
    paired_bootstrap_diff  by HOW MUCH do they differ, plausibly?
    holm                   after running SEVERAL tests, which survive?
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

def wilson_interval(k:int, n: int, z:float = 1.96) -> tuple[float,float]:
    """95% confidence interval for a proportion k/n.

    Takes:   k events out of n trials.
    Returns: (low, high) as fractions in [0, 1].

    Why not the textbook  p +- 1.96*sqrt(p(1-p)/n)?  At 0% or 100% that
    formula gives [0, 0] or [1, 1] -- it claims certainty exactly where the
    data says least. Wilson stays honest at the edges: 0/60 means "somewhere
    under ~6%", never "never".
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

#wilson_interval(0, 60)   → (0.000, 0.060)   "0% interrupted" really means "under 6%"
#wilson_interval(38, 60)  → (0.507, 0.744)   T1's 63% on disfluent: anywhere from 51% to 74%


@dataclass(frozen=True)
class McNemar:
    """Result of a paired test on interruptions."""
    only_a: int      # turns where A interrupted and B did not
    only_b: int      # turns where B interrupted and A did not
    p: float         # exact two-sided p-value

    @property
    def discordant(self) -> int:
        """The only turns that carry information about the difference."""
        return self.only_a + self.only_b

def mcnemar_exact(a_cut, b_cut) -> McNemar:
    """Exact McNemar test.

    Takes:   two equal-length sequences of bools, one entry per turn, same order.
    Returns: McNemar(only_a, only_b, p).

    Turns where both interrupted, or neither did, are ignored -- they say
    nothing about which policy is better. If the policies were equally good,
    each remaining turn would be a coin flip between "A-only" and "B-only".
    The p-value is how lopsided the observed split is for a fair coin.
    """
    a = np.asarray(a_cut, dtype=bool)
    b = np.asarray(b_cut, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired test needs the same turns in the same order")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    m = only_a + only_b
    if m == 0:
        return McNemar(only_a, only_b, 1.0)
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return McNemar(only_a, only_b, min(1.0, 2 * tail))

# mcnemar_exact(t0_cuts, t1_cuts)  → McNemar(only_a=8, only_b=0, p=0.0078)
# Worked example (T0 500 vs T1 500 on disfluent): 8 turns where only T0 interrupted and 0 where only T1 did. That's like 8 coin flips all landing the same way: 2 × 0.5⁸ = 0.0078
# The other 52 turns (38 where both interrupted, 14 where neither did) don't enter the calculation at all.


def paired_bootstrap_diff(a_cut, b_cut, n_boot: int = 5000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Difference in interruption rate (A minus B) with a 95% CI.

    Takes:   two equal-length sequences of bools, same turn order.
    Returns: (diff, low, high) as fractions. Positive diff = B interrupts less.

    Resamples whole TURNS, keeping each turn's pair of outcomes together --
    that is what "paired" means. Seeded, so the interval is reproducible.
    """
    a = np.asarray(a_cut, dtype=float)
    b = np.asarray(b_cut, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs the same turns in the same order")

    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    d = a - b                                   # per turn: +1, 0 or -1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)                 # 5000 resampled differences
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))

#paired_bootstrap_diff(t0_cuts, t1_cuts)  → (0.133, 0.050, 0.217)
def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the ORIGINAL order.

    Takes:   raw p-values from every test you intend to report.
    Returns: adjusted p-values, same length, same order.

    Run 20 tests at alpha = 0.05 and you expect one "significant" result from
    pure noise. Holm corrects for that: sort ascending, multiply the smallest
    by m, the next by m-1, and so on, never letting an adjusted value fall
    below the one before it.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted

#holm([0.0078, 0.118, 1.0])  → [0.023, 0.236, 1.0]    smallest × 3 = 0.023   next × 2 = 0.236   last × 1 = 1.0

@dataclass
class Comparison:
    """Everything known about one A-vs-B comparison on one cohort."""

    cohort: str
    policy_a: str
    policy_b: str
    n: int                      # turns compared (the same turns for both)
    rate_a: float               # A's interruption rate, 0..1
    rate_b: float
    test: McNemar               # only_a, only_b, raw p
    diff: float                 # rate_a - rate_b; positive means B interrupts less
    ci: tuple[float, float]     # paired bootstrap 95% CI on diff
    p_adjusted: float = float("nan")   # NaN until adjust() fills it in

    def verdict(self, alpha: float = 0.05) -> str:
        """Plain-English reading. Uses the ADJUSTED p, so run adjust() first."""
        if self.test.discordant == 0:
            return "identical: same turns interrupted"
        if not self.p_adjusted < alpha:     # also catches NaN (adjust() not run)
            return "no detectable difference"
        winner = self.policy_b if self.diff > 0 else self.policy_a
        return f"{winner} interrupts less"


def compare(cohort: str, name_a: str, a_cut, name_b: str, b_cut,
            seed: int = 0) -> Comparison:
    """Run both tests on one pair of policies and bundle the answers.

    Takes:   names plus two equal-length bool sequences, same turn order.
    Calls:   mcnemar_exact() and paired_bootstrap_diff().
    Returns: a Comparison with p_adjusted still NaN -- adjusting needs every
            comparison in the family at once, so it happens in adjust().
    """
    a = np.asarray(a_cut, dtype=bool)
    b = np.asarray(b_cut, dtype=bool)
    diff, lo, hi = paired_bootstrap_diff(a, b, seed=seed)
    return Comparison(
        cohort=cohort, policy_a=name_a, policy_b=name_b, n=int(a.size),
        rate_a=float(a.mean()) if a.size else float("nan"),
        rate_b=float(b.mean()) if b.size else float("nan"),
        test=mcnemar_exact(a, b), diff=diff, ci=(lo, hi),
    )


def adjust(comparisons: list[Comparison]) -> list[Comparison]:
    """Fill in p_adjusted on every comparison, using Holm across all of them.

    Changes the Comparison objects in place and returns the same list, so it
    chains:  adjust([compare(...), compare(...)])
    """
    for c, p in zip(comparisons, holm([c.test.p for c in comparisons])):
        c.p_adjusted = p
    return comparisons