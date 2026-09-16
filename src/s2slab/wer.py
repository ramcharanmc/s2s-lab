"""
Word error rate: how much of what was said the system actually heard.

WER counts the words a recognizer got wrong, as a share of the words really
spoken: (substitutions + deletions + insertions) / reference words. It can pass
100%, because a recognizer can invent more words than were said.

TWO NUMBERS, ALWAYS BOTH (v1 metric definitions).

  raw         lower case, dashes and punctuation out, spaces collapsed, and
              nothing else. "Four seven two" stays three words, "uh" stays a
              word, "don't" stays as it is. Raw WER is what the pipeline
              actually emitted, before anyone tidied it up.
  normalized  Whisper's EnglishTextNormalizer, the number every ASR paper
              quotes. It writes numbers as digits ("fourteenth" and "the 14th"
              both become "14th"), expands "Mr." to "mister" and "don't" to
              "do not" -- AND DELETES FILLERS such as "uh" and "um".

That last point is why both are reported. On the disfluent cohort the
normalized score cannot see a filler the recognizer dropped, which is exactly
the error that cohort exists to measure; the raw score can.

Reference text comes from the frozen eval set (evalset/manifest.json), so these
numbers are only as good as that set -- see evalset/README.md for what each
cohort is and is not.

    score(hyps)                 {cohort: {"raw": Errors, "normalized": Errors}}
     ├─ references()            {id: (cohort, reference)} from the manifest
     └─ wer(ref, hyp, norm)     one utterance
         ├─ normalizer(text)    raw or normalized
         └─ align(ref, hyp)     Levenshtein: equal / sub / delete / insert

Corpus WER is total errors over total reference words -- never the mean of
per-utterance rates, which would let a three-word turn count as much as a
thirty-word one.
"""

from __future__ import annotations

import collections
import functools
import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass

from .evalset import EVAL_ROOT, MANIFEST

_DASH = re.compile(r"[-\u2013\u2014_/]")          # hyphen, en dash, em dash, underscore, slash
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)      # keep letters, digits, spaces, apostrophes


def light(text: str) -> str:
    """Lower case, dashes to spaces, other punctuation gone, spaces collapsed."""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_PUNCT.sub("", _DASH.sub(" ", text)).split())


@functools.lru_cache(maxsize=1)
def _whisper_normalizer():
    """Built once: it compiles a fair number of regexes."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer({})      # {} = no British/American spelling map


def standard(text: str) -> str:
    """Whisper's EnglishTextNormalizer. Needs transformers (requirements/r0.txt)."""
    return _whisper_normalizer()(text)


NORMALIZERS = {"raw": light, "normalized": standard}

def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Line the two word sequences up, as cheaply as possible.

    Takes:   reference words, recognized words.
    Returns: [(op, reference word, recognized word), ...] in order, where op is
             "equal", "sub", "delete" (a word was missed) or "insert" (a word
             was invented). A deletion carries no recognized word, an insertion
             no reference word.
    """
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                             cost[i - 1][j] + 1,      # the reference word was missed
                             cost[i][j - 1] + 1)      # a word was invented
    ops, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("delete", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("insert", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]

@dataclass(frozen=True)
class Errors:
    """Counted errors, and the words they are counted against."""

    n_utterances: int
    n_words: int          # reference words
    sub: int
    delete: int
    insert: int

    @property
    def errors(self) -> int:
        return self.sub + self.delete + self.insert

    @property
    def rate(self) -> float:
        """Errors per reference word. NaN when the reference is empty."""
        return self.errors / self.n_words if self.n_words else float("nan")

    def __add__(self, other: "Errors") -> "Errors":
        return Errors(self.n_utterances + other.n_utterances, self.n_words + other.n_words,
                      self.sub + other.sub, self.delete + other.delete, self.insert + other.insert)

    def __str__(self) -> str:
        return (f"{100 * self.rate:5.1f}%  [{self.errors} of {self.n_words} words, "
                f"{self.n_utterances} turns: {self.sub}S {self.delete}D {self.insert}I]")


ZERO = Errors(0, 0, 0, 0, 0)


def wer(ref: str, hyp: str, normalizer=light) -> Errors:
    """One utterance: normalize both sides, align, count."""
    ops = align(normalizer(ref).split(), normalizer(hyp).split())
    counted = collections.Counter(op for op, _, _ in ops)
    return Errors(1, sum(counted[k] for k in ("equal", "sub", "delete")),
                  counted["sub"], counted["delete"], counted["insert"])


def corpus_wer(pairs, normalizer=light) -> Errors:
    """Many utterances: total errors over total reference words."""
    total = ZERO
    for ref, hyp in pairs:
        total = total + wer(ref, hyp, normalizer)
    return total


def mistakes(ref: str, hyp: str, normalizer=light) -> list[tuple[str, str | None, str | None]]:
    """Just the errors, for reading: every op except "equal"."""
    return [op for op in align(normalizer(ref).split(), normalizer(hyp).split()) if op[0] != "equal"]

def references(root=EVAL_ROOT, key: str = "text") -> dict[str, tuple[str, str]]:
    """{id: (cohort, reference text)} from the eval set's manifest -- no audio read.

    key="text_fluent" gives the disfluent cohort's tidied reference, the one
    with fillers and false starts already removed.
    """
    items = json.loads((pathlib.Path(root) / MANIFEST).read_text(encoding="utf-8"))["items"]
    return {i["id"]: (i["cohort"], i[key]) for i in items if key in i}


def score(hyps: dict[str, str], root=EVAL_ROOT, key: str = "text") -> dict[str, dict[str, Errors]]:
    """Score a run against the eval set.

    Takes:   {clip id: what the system heard}. An id the eval set does not know
             is an error, not a silent skip.
    Returns: {cohort: {"raw": Errors, "normalized": Errors}}.
    """
    refs = references(root, key)
    out: dict[str, dict[str, Errors]] = {}
    for item_id, hyp in hyps.items():
        if item_id not in refs:
            raise KeyError(f"{item_id} is not in the eval set")
        cohort, ref = refs[item_id]
        for name, normalizer in NORMALIZERS.items():
            got = out.setdefault(cohort, {}).get(name, ZERO)
            out[cohort][name] = got + wer(ref, hyp, normalizer)
    return out


def format_table(scored: dict[str, dict[str, Errors]], title: str = "") -> str:
    """Markdown: one row per cohort, raw and normalized side by side."""
    lines = [f"### {title}", ""] if title else []
    lines += ["| cohort | turns | ref words | raw WER | normalized WER |",
              "|---|---:|---:|---:|---:|"]
    total = {name: ZERO for name in NORMALIZERS}
    for cohort in sorted(scored):
        row = scored[cohort]
        for name in NORMALIZERS:
            total[name] = total[name] + row[name]
        lines.append(f"| {cohort} | {row['raw'].n_utterances} | {row['raw'].n_words} | "
                     f"{100 * row['raw'].rate:.1f}% | {100 * row['normalized'].rate:.1f}% |")
    if len(scored) > 1:
        lines.append(f"| **all** | {total['raw'].n_utterances} | {total['raw'].n_words} | "
                     f"{100 * total['raw'].rate:.1f}% | {100 * total['normalized'].rate:.1f}% |")
    return "\n".join(lines)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit('usage: PYTHONPATH=src python -m s2slab.wer hyps.json\n'
                 '       hyps.json: {"clean-00": "what the system heard", ...}')
    hyps = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(format_table(score(hyps), title=f"WER on the eval set: {sys.argv[1]}"))
    fluent = {k: v for k, v in hyps.items() if k in references(key="text_fluent")}
    if fluent:
        print()
        print(format_table(score(fluent, key="text_fluent"),
                           title="disfluent cohort against the tidied reference "
                                 "(fillers and false starts already removed)"))