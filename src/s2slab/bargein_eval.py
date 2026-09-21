"""
Scoring barge-in policies on a probe set: wrong decisions and stop latency, together.

    python -m s2slab.bargein_eval                                     B0 at four thresholds, frozen set
    python -m s2slab.bargein_eval B1_words --root probes/bargein_dev  develop on the synthetic set, freely
    python -m s2slab.bargein_eval B1_words --asr-latency-ms 70        ...charging a T4's recognizer time
    python -m s2slab.bargein_eval B1_words B2_speechlike --noise-dir probes/bargein_dev/noise
                                                                      real coughs and laughs: all "keep going"    
    python -m s2slab.bargein_eval B1_words --confirm PR-003           the frozen set needs a registration
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

A POLICY THAT READS WORDS PAYS FOR THEM. For such a policy the recognizer is run
over each item first (bargein_asr.py), and every transcript carries the time it
was DELIVERED. By default that is what each call really took on this machine;
--asr-latency-ms charges a stated figure instead. The transcripts are cached
under results/bargein/, so a changed rule is re-scored without the recognizer.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .bargein import BARGEIN_ROOT, CELLS, BargeIn, load_set, read_lag_s
from .bargein_asr import cache_path, partials_for, whisper
from .bargein_policies import POLICIES, BargeInContext, BargeInPolicy, get
from .probes import TARGET_SR

WINDOW_BEFORE_S = 1.0
WINDOW_AFTER_S = 2.0


def context_for(item: BargeIn, lag_s: float, partials=()) -> BargeInContext:
    """The caller's channel moved onto the agent's clock, and the item's window.

    Takes:   a loaded item, the overdub lag its recording carries, and the
             recognizer's partials for it, if the policy reads words.
    Returns: a context whose time zero is the moment the agent started: the
             first lag_s of the recording are dropped, so a sound the label
             puts at onset_s really is at onset_s in ctx.audio.
    """
    audio = item.audio[int(round(lag_s * TARGET_SR)):]
    end_s = min(len(audio) / TARGET_SR, item.onset_s + item.duration_s + WINDOW_AFTER_S)
    return BargeInContext(audio, TARGET_SR, max(0.0, item.onset_s - WINDOW_BEFORE_S), end_s,
                          partials=list(partials))



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
    overrun_words: int | None = None   # agent words finished between the caller's onset and the stop    

    @property
    def correct(self) -> bool:
        return (self.stop_s is not None) == self.should_stop

    @property
    def error(self) -> str:
        if self.correct:
            return ""
        return "missed interruption" if self.should_stop else "false barge-in"


def score(policy: BargeInPolicy, items, lag_s: float, partials: dict | None = None) -> list[Outcome]:
    """Run one policy over every item. Takes loaded items and, for a policy that
    reads words, {item id: [Partial, ...]}; returns one Outcome each."""
    out = []
    for item in items:
        decision = policy.decide(context_for(item, lag_s, (partials or {}).get(item.id, ())))
        latency = (decision.stop_s - item.onset_s) * 1000 if decision.fired else None
        overrun = (item.words_finished_by(decision.stop_s) - item.words_finished_by(item.onset_s)
                   if decision.fired else None)
        out.append(Outcome(item.id, item.cell, item.kind, item.should_stop,
                           decision.stop_s, decision.reason, latency, overrun))    
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
        "hit_overrun_words": sorted(o.overrun_words for o in outcomes if o.should_stop and o.correct),
    }



def format_report(results: dict[str, list[Outcome]]) -> str:
    """One row per policy, then every wrong decision by name.

    Takes: {policy name: outcomes}. The latency cell lists the hits' latencies
    (fastest, middle, slowest) and how many there were -- never a bare median.
    """
    lines = ["| policy | " + " | ".join(CELLS) + " | wrong | missed | false barge-in | of which noise | "
             "stop latency of the hits, ms | agent words said over the caller |",
             "|---|" + "---:|" * (len(CELLS) + 4) + "---|---|"]
    for name, outcomes in results.items():
        s = summarize(outcomes)
        lat = s["hit_latencies_ms"]
        latency = (f"{lat[0]:.0f} / {lat[len(lat) // 2]:.0f} / {lat[-1]:.0f}  (fastest / middle / slowest of "
                   f"{len(lat)} hits)") if lat else "no hits"
        over = s["hit_overrun_words"]
        overrun = f"{over[0]} / {over[len(over) // 2]} / {over[-1]}" if over else "-"
        lines.append(f"| {name} | " + " | ".join(str(s["wrong"][c]) for c in CELLS)
                     + f" | **{s['wrong_total']} of {s['n']}** | {s['missed']} | {s['false_bargein']} | "
                       f"{s['wrong_noise']} of {s['n_noise']} | {latency} | {overrun} |")
    for name, outcomes in results.items():
        errors = [o for o in outcomes if not o.correct]
        lines += ["", f"{name}: " + (", ".join(
            f"{o.item} ({o.error}" + (f" at {o.latency_ms:+.0f} ms" if o.latency_ms is not None else "") + ")"
            for o in errors) or "every decision right")]
    return "\n".join(lines)


def score_set(names: list[str], root: pathlib.Path = BARGEIN_ROOT, confirm: str = "",
              asr_latency_ms: float | None = None, refresh: bool = False, recognizer=whisper) -> str:
    """Score named policies on a probe set. The frozen set is locked to baselines
    unless `confirm` names the registration this run belongs to. If any policy
    reads words, the recognizer runs first (or its cache is read)."""
    root = pathlib.Path(root)
    policies = [get(n) for n in names]
    locked = [p.name for p in policies if not p.baseline]
    if locked and root.resolve() == BARGEIN_ROOT.resolve() and not confirm:
        raise PermissionError(f"{', '.join(locked)}: the frozen barge-in set is a test set. Scoring a "
                              "policy on it is a registered confirmation run "
                              "(docs/preregistration.md); pass --confirm PR-003.")
    items = list(load_set(root).values())
    lag_s = read_lag_s(root)
    head = f"{len(items)} items from {root}" + (f" · confirms {confirm}" if confirm else "")
    partials = None
    if any(p.needs_partials for p in policies):
        contexts = {item.id: context_for(item, lag_s) for item in items}
        partials = partials_for(contexts, recognizer, asr_latency_ms,
                                cache_path(root.resolve().name, asr_latency_ms), refresh)
        head += (" · recognizer time: " + ("as measured on this machine" if asr_latency_ms is None
                                           else f"charged at {asr_latency_ms:.0f} ms a call"))
    return head + "\n\n" + format_report({p.name: score(p, items, lag_s, partials) for p in policies})


def score_noise(names: list[str], noise_dir: pathlib.Path, asr_latency_ms: float | None = None,
                refresh: bool = False, recognizer=whisper) -> str:
    """Score policies on a folder of real non-speech clips: every one is "keep going".

    Returns the usual table, then one line per clip: what the recognizer wrote
    for the whole sound and how sure it was that there was no speech -- which is
    what a policy that reads words is up against.
    """
    from .bargein_dev import noise_items
    noise_dir = pathlib.Path(noise_dir)
    items = noise_items(noise_dir)
    policies = [get(n) for n in names]
    partials = None
    if any(p.needs_partials for p in policies):
        contexts = {item.id: context_for(item, 0.0) for item in items}
        partials = partials_for(contexts, recognizer, asr_latency_ms,
                                cache_path(f"noise-{noise_dir.resolve().parent.name}", asr_latency_ms), refresh)
    head = f"{len(items)} real noise clips from {noise_dir}: the right answer is always to keep going"
    lines = [head, "", format_report({p.name: score(p, items, 0.0, partials) for p in policies}), ""]
    for item in items:
        last = (partials or {}).get(item.id) or []
        heard = f"{last[-1].text[:70]!r} (no-speech {last[-1].no_speech:.2f})" if last else "nothing transcribed"
        lines.append(f"{item.id}: lasts {item.duration_s:.2f} s, recognizer wrote {heard}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score barge-in policies on a probe set.")
    ap.add_argument("policies", nargs="*", help="policy names; default: every B0 baseline")
    ap.add_argument("--root", default=str(BARGEIN_ROOT))
    ap.add_argument("--confirm", default="", help="the registration this run confirms, e.g. PR-003")
    ap.add_argument("--asr-latency-ms", type=float, default=None,
                    help="charge every recognizer call this much instead of what it took here")
    ap.add_argument("--refresh", action="store_true", help="run the recognizer again, ignoring the cache")
    ap.add_argument("--noise-dir", default="", help="score on a folder of real non-speech clips instead of a set")
    a = ap.parse_args()
    names = a.policies or [n for n, p in POLICIES.items() if p.baseline]
    if a.noise_dir:
        print(score_noise(names, pathlib.Path(a.noise_dir), a.asr_latency_ms, a.refresh))
    else:
        print(score_set(names, pathlib.Path(a.root), a.confirm, a.asr_latency_ms, a.refresh))