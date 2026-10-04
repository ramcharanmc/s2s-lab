"""
Error analysis: WHY does a policy interrupt the turns it interrupts?

The sweep says T1 still interrupts 67% of disfluent turns. This module looks at
each of those turns individually and asks, at the moment of the cut:

    1. What was the last word the speaker had finished?
       A FUNCTION word ("is", "the") means T1's hold fired and the pause simply
       lasted longer than the hold. A CONTENT word ("seven", "tired") means the
       hold never fired -- the 46-word list could not tell the sentence was
       unfinished.
    2. How long was the pause the cut landed in, as the VAD saw it?

WHAT THIS CAN AND CANNOT TELL YOU. The simulator places mid-turn pauses at
random word boundaries, so "most failures follow a content word" could have
been an artefact of that placement: hesitation research finds pauses
clustering just BEFORE content words ("the ... appointment"), right after a
function word, where T1's list does catch them. The real recordings in probes/
(2026-09-14) point the other way for the turns this system serves. In
dictation and read speech -- addresses, phone numbers, spelled names, lists --
35 of 36 unprompted pauses followed a CONTENT word, at a phrase or list
boundary, where T1 is blind. That is one speaker performing a task, not a
sample of disfluent speakers, but for this kind of turn the simulator does not
understate T1's problem.

Run this on EXPLORATORY data (seed 0). Anything you change because of what it
shows has to be confirmed on a seed nobody has looked at -- 1000 is now used.

PR-001's step 6 (after its confirmation, labelled exploratory) is T1b against T1 on the
confirmation's own turns:

    python -m s2slab.diagnose --policy T1b_complete_500ms --contrast T1_semantic_500ms \
        --split eval --turns sentence --seed 2000

The EVAL sentences are refused until runs/turn-taking/pr001.json exists.

main()
 ├─ make_all(n, seed)[cohort]                → 250 utterances
 └─ for T0_fixed_500ms, then T1_semantic_500ms:
     interruptions(policy, utterances)       → [Interruption, ...]
      for each turn:
       ├─ policy.decide(ctx)                 → EndpointDecision(endpoint_s, fired, reason)
       ├─ skip unless fired and endpoint_s < true_end_s
       ├─ ctx.partial_at(endpoint_s)         → e.g. "my account number is four seven"
       ├─ classify_tail(partial)             → e.g. ("seven", "content")
       └─ silent_run_ms(speech_prob, …)      → e.g. 620.0
     format_breakdown(records, …)
      └─ breakdown(records)                  → {"content": {...}, "function": {...}}

"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .policies import INCOMPLETE_TAILS, EndpointPolicy
from .simulate import Utterance

@dataclass(frozen=True)
class Interruption:
    """One turn a policy cut off, described at the moment of the cut."""

    turn: int            # index of the turn in the cohort list
    cut_s: float         # when the policy started talking
    true_end_s: float    # when the speaker actually finished
    last_word: str       # last word finished before the cut ("" if none)
    tail: str            # "function", "content" or "none"
    pause_ms: float      # length of the silence the cut landed in, per the VAD

    @property
    def early_ms(self) -> float:
        """How far ahead of the real end the policy jumped in."""
        return (self.true_end_s - self.cut_s) * 1000

def classify_tail(partial: str) -> tuple[str, str]:
    """Last word of a partial transcript, and whether T1's list would flag it.

    Takes:   a partial transcript, e.g. "my account number is four seven".
    Returns: (last_word, tail). tail is "function" if the word is in
             INCOMPLETE_TAILS, "content" otherwise, "none" if nothing was said.

    Cleans the word exactly the way SemanticThreshold._should_hold does, so the
    verdict here matches what T1 actually saw.
    """
    words = (partial or "").strip().lower().split()
    if not words:
        return "", "none"
    last = words[-1].strip(".,!?;:\"'")
    return last, ("function" if last in INCOMPLETE_TAILS else "content")

#classify_tail("my account number is")              → ("is", "function")
#classify_tail("my account number is four seven")   → ("seven", "content")
#classify_tail("")                                  → ("", "none")

def silent_run_ms(speech_prob: list[float], frame: int, frame_ms: float) -> float:
    """Length of the silence containing `frame`, as the VAD saw it.

    Takes:   per-frame speech probabilities, a frame index, the frame length.
    Returns: milliseconds from the start of that silent run to where speech
             resumes (or the buffer ends). 0.0 if `frame` is not silent.

    A policy fires PART-WAY through a pause, so the cut frame alone says nothing
    about how long the pause really was. Walk outwards in both directions to
    find its full extent.
    """
    if not 0 <= frame < len(speech_prob) or speech_prob[frame] >= 0.5:
        return 0.0
    start = frame
    while start > 0 and speech_prob[start - 1] < 0.5:
        start -= 1
    end = frame
    while end + 1 < len(speech_prob) and speech_prob[end + 1] < 0.5:
        end += 1
    return (end - start + 1) * frame_ms

def interruptions(policy: EndpointPolicy,
                  utterances: list[Utterance]) -> list[Interruption]:
    """Every turn this policy interrupted, described at the moment of the cut.

    Takes:   a policy and a cohort of utterances.
    Calls:   policy.decide(ctx), ctx.partial_at(cut), classify_tail(),
             silent_run_ms().
    Returns: one Interruption per interrupted turn, in cohort order. Turns the
             policy handled correctly, or never decided, are left out.
    """
    out: list[Interruption] = []
    for i, u in enumerate(utterances):
        ctx = u.context()
        d = policy.decide(ctx)
        if not (d.fired and d.endpoint_s < u.true_end_s):
            continue                          # same definition of "cut" as the sweep
        last, tail = classify_tail(ctx.partial_at(d.endpoint_s))
        frame = int(round(d.endpoint_s * 1000 / ctx.frame_ms)) - 1
        out.append(Interruption(
            turn=i, cut_s=d.endpoint_s, true_end_s=u.true_end_s,
            last_word=last, tail=tail,
            pause_ms=silent_run_ms(ctx.speech_prob, frame, ctx.frame_ms),
        ))
    return out

def breakdown(records: list[Interruption]) -> dict[str, dict]:
    """Group interruptions by what the speaker had just said.

    Takes:   the list interruptions() returned.
    Returns: {tail: {"n", "share", "pause_p50_ms", "top_words"}} for each of
             "content", "function", "none" that actually occurs.
    """
    total = len(records)
    out: dict[str, dict] = {}
    for tail in ("content", "function", "none"):
        group = [r for r in records if r.tail == tail]
        if not group:
            continue
        out[tail] = {
            "n": len(group),
            "share": len(group) / total,
            "pause_p50_ms": float(np.median([r.pause_ms for r in group])),
            "top_words": Counter(r.last_word for r in group).most_common(5),
        }
    return out

def format_breakdown(records: list[Interruption], policy: str, cohort: str,
                     n_turns: int) -> str:
    """Markdown summary of one policy's interruptions on one cohort."""
    lines = [f"### {policy} on {cohort}: {len(records)} of {n_turns} turns interrupted", "",
             "| last word before the cut | turns | share | median pause ms | most common |",
             "|---|---:|---:|---:|---|"]
    for tail, s in breakdown(records).items():
        words = ", ".join(f"{w} ×{c}" for w, c in s["top_words"])
        lines.append(f"| {tail} word | {s['n']} | {s['share']:.0%} "
                     f"| {s['pause_p50_ms']:.0f} | {words} |")
    return "\n".join(lines)


PR001_RESULT = "runs/turn-taking/pr001.json"


def _policy(name: str) -> EndpointPolicy:
    """A registered policy, or T1b, which needs its language model and is built on demand."""
    if name == "T1b_complete_500ms":
        from .completion import t1b
        return t1b()
    from .policies import get
    return get(name)


def main(policy: str = "T1_semantic_500ms", cohort: str = "disfluent",
         n: int = 250, seed: int = 0, split: str = "dev", turns: str = "prefix",
         contrast: str = "T0_fixed_500ms") -> None:
    """Diagnose `policy`, with `contrast` printed first.

    Uses make_all, not make_cohort, so the turns are exactly the ones the sweep
    scored for this seed, split and turn mode.
    """
    import pathlib

    from .simulate import make_all
    if split == "eval" and not pathlib.Path(PR001_RESULT).exists():
        raise SystemExit(f"the EVAL sentences are PR-001's confirmation's, and {PR001_RESULT} does not exist yet")
    utts = make_all(n, seed, split=split, turns=turns)[cohort]
    print(f"seed {seed} · {split} sentences · {turns} turns · {cohort}" + (" · exploratory" if split == "eval" else ""))
    print()
    for name in dict.fromkeys((contrast, policy)):            # skips a duplicate
        print(format_breakdown(interruptions(_policy(name), utts), name, cohort, len(utts)))
        print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="T1_semantic_500ms")
    ap.add_argument("--cohort", default="disfluent")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--turns", default="prefix", choices=["prefix", "sentence"])
    ap.add_argument("--contrast", default="T0_fixed_500ms")
    a = ap.parse_args()
    main(a.policy, a.cohort, a.n, a.seed, a.split, a.turns, a.contrast)