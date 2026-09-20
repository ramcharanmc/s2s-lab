"""
Scoring barge-in policies on a probe set: wrong decisions and stop latency, together.

    python -m s2slab.bargein_eval                         B0 at four thresholds, frozen set
    python -m s2slab.bargein_eval B1_words --confirm PR-003   anything else needs a registration

ONE ITEM, ONE WINDOW. A policy is asked about an item from WINDOW_BEFORE_S before
its labelled onset to WINDOW_AFTER_S after its labelled end. A stray noise eight
seconds later is not the item; what happens inside the window is.

    stop item       stops inside the window  -> hit, latency = stop - labelled onset
                    does not                 -> MISSED INTERRUPTION
    continue item   stops inside the window  -> FALSE BARGE-IN
                    does not                 -> correct

THE FROZEN SET IS A TEST SET. Its 20 items are scored once per registered
claim. B0 is the baseline the set was designed against and has nothing to tune,
so it may be run freely; any other policy needs --confirm with its registration
(docs/preregistration.md), as sweep.py requires for the EVAL sentences.

NO MEDIAN FROM TEN NUMBERS. timing.py wants 20 samples for a p50, and a set
with 10 stop items cannot give one. The latencies of the hits are listed.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .bargein import BARGEIN_ROOT, CELLS, BargeIn, load_set, read_lag_s
from .bargein_policies import POLICIES, BargeInContext, BargeInPolicy, get
from .probes import TARGET_SR

WINDOW_BEFORE_S = 1.0
WINDOW_AFTER_S = 2.0


def context_for(item: BargeIn, lag_s: float) -> BargeInContext:
    """The caller's channel moved onto the agent's clock, and the item's window.

    Takes:   a loaded item, and the overdub lag its recording carries.
    Returns: a context whose time zero is the moment the agent started: the
             first lag_s of the recording are dropped, so a sound the label
             puts at onset_s really is at onset_s in ctx.audio.
    """
    audio = item.audio[int(round(lag_s * TARGET_SR)):]
    end_s = min(len(audio) / TARGET_SR, item.onset_s + item.duration_s + WINDOW_AFTER_S)
    return BargeInContext(audio, TARGET_SR, max(0.0, item.onset_s - WINDOW_BEFORE_S), end_s)



@dataclass
class Outcome:
    """What one policy did with one item."""

    item: str
    cell: str
    kind: str
    should_stop: bool
    stop_s: float | None
    reason: str
    latency_ms: float | None       # stop time minus labelled onset; None if it did not stop

    @property
    def correct(self) -> bool:
        return (self.stop_s is not None) == self.should_stop

    @property
    def error(self) -> str:
        if self.correct:
            return ""
        return "missed interruption" if self.should_stop else "false barge-in"


def score(policy: BargeInPolicy, items, lag_s: float) -> list[Outcome]:
    """Run one policy over every item. Takes loaded items; returns one Outcome each."""
    out = []
    for item in items:
        decision = policy.decide(context_for(item, lag_s))
        latency = (decision.stop_s - item.onset_s) * 1000 if decision.fired else None
        out.append(Outcome(item.id, item.cell, item.kind, item.should_stop,
                           decision.stop_s, decision.reason, latency))
    return out


def summarize(outcomes: list[Outcome]) -> dict:
    """Counts, by cell and by kind of error, and the latencies of the hits."""
    wrong = {cell: sum(1 for o in outcomes if o.cell == cell and not o.correct) for cell in CELLS}
    return {
        "n": len(outcomes),
        "wrong": wrong,
        "wrong_total": sum(wrong.values()),
        "missed": sum(1 for o in outcomes if o.error == "missed interruption"),
        "false_bargein": sum(1 for o in outcomes if o.error == "false barge-in"),
        "wrong_noise": sum(1 for o in outcomes if o.kind == "noise" and not o.correct),
        "n_noise": sum(1 for o in outcomes if o.kind == "noise"),
        "hit_latencies_ms": sorted(o.latency_ms for o in outcomes if o.should_stop and o.correct),
    }



def format_report(results: dict[str, list[Outcome]]) -> str:
    """One row per policy, then every wrong decision by name.

    Takes: {policy name: outcomes}. The latency cell lists the hits' latencies
    (fastest, middle, slowest) and how many there were -- never a bare median.
    """
    lines = ["| policy | " + " | ".join(CELLS) + " | wrong | missed | false barge-in | of which noise | "
             "stop latency of the hits, ms |", "|---|" + "---:|" * (len(CELLS) + 4) + "---|"]
    for name, outcomes in results.items():
        s = summarize(outcomes)
        lat = s["hit_latencies_ms"]
        latency = (f"{lat[0]:.0f} / {lat[len(lat) // 2]:.0f} / {lat[-1]:.0f}  (fastest / middle / slowest of "
                   f"{len(lat)} hits)") if lat else "no hits"
        lines.append(f"| {name} | " + " | ".join(str(s["wrong"][c]) for c in CELLS)
                     + f" | **{s['wrong_total']} of {s['n']}** | {s['missed']} | {s['false_bargein']} | "
                       f"{s['wrong_noise']} of {s['n_noise']} | {latency} |")
    for name, outcomes in results.items():
        errors = [o for o in outcomes if not o.correct]
        lines += ["", f"{name}: " + (", ".join(
            f"{o.item} ({o.error}" + (f" at {o.latency_ms:+.0f} ms" if o.latency_ms is not None else "") + ")"
            for o in errors) or "every decision right")]
    return "\n".join(lines)


def score_set(names: list[str], root: pathlib.Path = BARGEIN_ROOT, confirm: str = "") -> str:
    """Score named policies on a probe set. The frozen set is locked to baselines
    unless `confirm` names the registration this run belongs to."""
    root = pathlib.Path(root)
    policies = [get(n) for n in names]
    locked = [p.name for p in policies if not p.baseline]
    if locked and root.resolve() == BARGEIN_ROOT.resolve() and not confirm:
        raise PermissionError(f"{', '.join(locked)}: the frozen barge-in set is a test set. Scoring a "
                              "policy on it is a registered confirmation run "
                              "(docs/preregistration.md); pass --confirm PR-003.")
    items = list(load_set(root).values())
    head = f"{len(items)} items from {root}" + (f" · confirms {confirm}" if confirm else "")
    return head + "\n\n" + format_report({p.name: score(p, items, read_lag_s(root)) for p in policies})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score barge-in policies on a probe set.")
    ap.add_argument("policies", nargs="*", help="policy names; default: every B0 baseline")
    ap.add_argument("--root", default=str(BARGEIN_ROOT))
    ap.add_argument("--confirm", default="", help="the registration this run confirms, e.g. PR-003")
    a = ap.parse_args()
    names = a.policies or [n for n, p in POLICIES.items() if p.baseline]
    print(score_set(names, pathlib.Path(a.root), a.confirm))