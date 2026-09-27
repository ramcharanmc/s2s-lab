"""
FT-2's registered test, PR-011: small.en with the LoRA merged in, against small.en itself, on the
frozen 200-clip eval set -- transcripts only, scored exactly as the ladder's rows are (s2slab.wer).

    pr011(base, lora)         the criterion from two {clip id: transcript} dicts:
                              S1  raw WER on the DISFLUENT cohort falls (LoRA minus base, in points;
                                  the 95% bootstrap interval over its 50 clips wholly below zero)
                              S2  the fillers in the disfluent references are kept, FILLER_RECALL_BAR
                                  or more of them (base small.en drops most of them)
                              S3  the guardrail: normalized WER over the OTHER 150 clips (clean,
                                  accented, slow: other speakers, fluent speech) does not rise by
                                  more than GUARD_PP points (the interval's upper end)
                              supported = S1 and S2 and S3
    format_pr011(results)     the printed verdict
    pr011_validity(results)   what would make a run not the registered one

The disfluent cohort is 50 clips of DisfluencySpeech's test split: the same single speaker the
LoRA trained on (its train split), so S1 and S2 measure same-speaker adaptation and the entry says
so; S3 is where other speakers are.

The bootstrap resamples CLIPS with replacement, 10,000 times, seed 0, and recomputes each model's
corpus WER (total errors over total reference words) on every resample; the difference's 2.5th and
97.5th percentiles are the interval. A difference in points: 100 * (lora - base).
"""

from __future__ import annotations

import collections

import numpy as np

from . import wer
from .evalset import EVAL_ROOT

REGISTRATION = "PR-011"
BASE_MODEL = "small.en"                      # R2's slow tier (r2-gpu); FT-2's base, unmodified
TARGET_COHORT = "disfluent"
GUARD_COHORTS = ("clean", "accented", "slow")
FILLER_RECALL_BAR = 0.90                     # S2: fixed at registration (26 of the cohort's 28 fillers)
GUARD_PP = 1.0                               # S3: fixed at registration
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0
# The words Whisper's EnglishTextNormalizer deletes -- the same set as ft/'s transcripts.FILLERS
# (a test in ft/ checks the two agree).
FILLERS = frozenset({"hmm", "mm", "mhm", "mmm", "uh", "um"})


def per_clip(hyps: dict[str, str], refs: dict[str, tuple[str, str]], normalizer) -> dict[str, wer.Errors]:
    """Each clip's counted errors, under one normalizer."""
    return {i: wer.wer(refs[i][1], hyps[i], normalizer) for i in hyps}


def corpus_rate(errors: dict[str, wer.Errors], ids) -> float:
    total = wer.ZERO
    for i in ids:
        total = total + errors[i]
    return total.rate


def bootstrap_points(base: dict[str, wer.Errors], lora: dict[str, wer.Errors], ids: list[str],
                     n: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """(point, low, high): 100 * (lora - base) corpus WER over `ids`, and its 95% interval."""
    ids = list(ids)
    b = np.array([[base[i].errors, base[i].n_words] for i in ids], dtype=float)
    l = np.array([[lora[i].errors, lora[i].n_words] for i in ids], dtype=float)
    point = 100 * (l[:, 0].sum() / l[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ids), size=(n, len(ids)))
    bs, ls = b[picks].sum(axis=1), l[picks].sum(axis=1)                 # (n, 2) each
    diffs = 100 * (ls[:, 0] / ls[:, 1] - bs[:, 0] / bs[:, 1])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fillers(text: str) -> collections.Counter:
    return collections.Counter(w for w in wer.light(text).split() if w in FILLERS)


def filler_recall(hyps: dict[str, str], refs: dict[str, tuple[str, str]], ids) -> tuple[int, int]:
    """(kept, in the references) over `ids`: a reference filler is kept when the hypothesis has
    the same word, up to the number the reference has."""
    kept = total = 0
    for i in ids:
        r, h = fillers(refs[i][1]), fillers(hyps[i])
        total += sum(r.values())
        kept += sum(min(k, h[w]) for w, k in r.items())
    return kept, total


def pr011(base: dict[str, str], lora: dict[str, str], root=EVAL_ROOT, n: int = BOOTSTRAP,
          seed: int = BOOTSTRAP_SEED) -> dict:
    """The registered criterion. `base` and `lora`: {clip id: transcript} over the same clips."""
    refs = wer.references(root)
    ids = sorted(i for i in base if i in lora)
    target = [i for i in ids if refs[i][0] == TARGET_COHORT]
    guard = [i for i in ids if refs[i][0] in GUARD_COHORTS]
    raw_b, raw_l = per_clip(base, refs, wer.light), per_clip(lora, refs, wer.light)
    norm_b, norm_l = per_clip(base, refs, wer.standard), per_clip(lora, refs, wer.standard)
    s1_point, s1_lo, s1_hi = bootstrap_points(raw_b, raw_l, target, n, seed)
    kept_b, total = filler_recall(base, refs, target)
    kept_l, _ = filler_recall(lora, refs, target)
    s3_point, s3_lo, s3_hi = bootstrap_points(norm_b, norm_l, guard, n, seed)
    s2_recall = kept_l / total if total else float("nan")
    s1, s2, s3 = s1_hi < 0, s2_recall >= FILLER_RECALL_BAR, s3_hi <= GUARD_PP
    return {"n_clips": len(ids), "n_target": len(target), "n_guard": len(guard),
            "s1": {"base_raw_wer": corpus_rate(raw_b, target), "lora_raw_wer": corpus_rate(raw_l, target),
                   "diff_pp": s1_point, "ci95_pp": [s1_lo, s1_hi], "pass": s1},
            "s2": {"fillers_in_refs": total, "base_kept": kept_b, "lora_kept": kept_l,
                   "base_recall": kept_b / total if total else float("nan"), "lora_recall": s2_recall,
                   "bar": FILLER_RECALL_BAR, "pass": s2},
            "s3": {"base_norm_wer": corpus_rate(norm_b, guard), "lora_norm_wer": corpus_rate(norm_l, guard),
                   "diff_pp": s3_point, "ci95_pp": [s3_lo, s3_hi], "bar_pp": GUARD_PP, "pass": s3},
            "target_norm": {"base": corpus_rate(norm_b, target), "lora": corpus_rate(norm_l, target)},
            "bootstrap": {"n": n, "seed": seed}, "supported": bool(s1 and s2 and s3)}


def pr011_validity(results: dict) -> list[str]:
    """What would make a run not the registered one."""
    problems = []
    c = results.get("criteria", {})
    if c.get("n_clips") != 200 or c.get("n_target") != 50 or c.get("n_guard") != 150:
        problems.append(f"{c.get('n_clips')} clips ({c.get('n_target')} disfluent, {c.get('n_guard')} others), "
                        "not the frozen 200 (50 + 150)")
    commit = str(results.get("commit", ""))
    if not commit or commit.endswith("-dirty") or commit == "unknown":
        problems.append(f"commit {commit!r}: not a clean committed tree")
    if not results.get("export_sha256"):
        problems.append("the exported model's SHA-256 is missing")
    if results.get("base_model") != BASE_MODEL:
        problems.append(f"the comparator is {results.get('base_model')!r}, not {BASE_MODEL}")
    return problems


def format_pr011(results: dict) -> str:
    c = results["criteria"]
    s1, s2, s3 = c["s1"], c["s2"], c["s3"]
    mark = lambda ok: "PASS" if ok else "FAIL"     # noqa: E731
    lines = [f"{REGISTRATION}, the registered criteria (docs/preregistration.md), {c['n_clips']} eval-set clips:",
             f"  S1  raw WER on the {c['n_target']} disfluent clips: base {100 * s1['base_raw_wer']:.1f}%, "
             f"LoRA {100 * s1['lora_raw_wer']:.1f}%; {s1['diff_pp']:+.1f} points, "
             f"95% interval {s1['ci95_pp'][0]:+.1f} to {s1['ci95_pp'][1]:+.1f}   {mark(s1['pass'])}",
             f"  S2  fillers kept on those clips: LoRA {s2['lora_kept']} of {s2['fillers_in_refs']} "
             f"({100 * s2['lora_recall']:.1f}%; base {s2['base_kept']}), bar {100 * s2['bar']:.0f}%   {mark(s2['pass'])}",
             f"  S3  normalized WER on the {c['n_guard']} other clips: base {100 * s3['base_norm_wer']:.1f}%, "
             f"LoRA {100 * s3['lora_norm_wer']:.1f}%; {s3['diff_pp']:+.1f} points, interval "
             f"{s3['ci95_pp'][0]:+.1f} to {s3['ci95_pp'][1]:+.1f}, allowed up to +{s3['bar_pp']:.1f}   {mark(s3['pass'])}"]
    problems = pr011_validity(results)
    if problems:
        lines.append(f"{REGISTRATION}: NOT THE REGISTERED RUN -- " + "; ".join(problems))
    else:
        failed = [k.upper() for k, v in (("s1", s1), ("s2", s2), ("s3", s3)) if not v["pass"]]
        lines.append(f"{REGISTRATION}: " + ("SUPPORTED" if c["supported"] else "NOT SUPPORTED -- failed " + ", ".join(failed)))
    return "\n".join(lines)
