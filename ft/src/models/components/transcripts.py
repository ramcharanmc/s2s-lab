"""
What a validation pass measures, from (reference, hypothesis) pairs -- pure functions.

Two WERs, as everywhere in s2s-lab (s2slab.wer): raw, which keeps fillers, and normalized, which
is Whisper's normalizer and deletes them. Plus one number the WERs blur: of the fillers in the
references, the share the hypotheses kept -- FT-2's whole point is a recognizer that writes down
the "uh"s it hears, and the raw WER alone cannot tell a kept filler from a right word elsewhere.
"""

from __future__ import annotations

import collections

from s2slab import wer

# The words Whisper's EnglishTextNormalizer deletes (transformers, english_normalizer.py):
# exactly the fillers the normalized WER cannot see.
FILLERS = frozenset({"hmm", "mm", "mhm", "mmm", "uh", "um"})


def fillers(text: str) -> collections.Counter:
    """The fillers in a text, after the raw normalizer (lower case, punctuation off)."""
    return collections.Counter(w for w in wer.light(text).split() if w in FILLERS)


def filler_recall(pairs) -> tuple[int, int]:
    """(kept, in the references): each reference filler counts as kept if the hypothesis has one too,
    word for word ("uh" for "uh"), up to the number the reference has."""
    kept = total = 0
    for ref, hyp in pairs:
        r, h = fillers(ref), fillers(hyp)
        total += sum(r.values())
        kept += sum(min(n, h[w]) for w, n in r.items())
    return kept, total


def summarize(pairs) -> dict[str, float]:
    """The validation numbers for a list of (reference, hypothesis)."""
    pairs = list(pairs)
    raw = wer.corpus_wer(pairs, wer.light)
    norm = wer.corpus_wer(pairs, wer.standard)
    kept, total = filler_recall(pairs)
    return {"wer_raw": raw.rate, "wer_normalized": norm.rate,
            "filler_recall": kept / total if total else float("nan"),
            "fillers_in_refs": float(total), "n_clips": float(raw.n_utterances), "n_words": float(raw.n_words)}
