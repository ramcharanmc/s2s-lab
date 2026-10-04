"""
Whole sentences for the simulator's sentence mode (PR-001, Deviations), drawn by a fixed rule from text
the author did not write: the transcripts of DailyTalk's dev fold (dialogue numbers ending in 1).

    PYTHONPATH=src python -m s2slab.simsentences        writes src/s2slab/sentences/whole-{dev,eval}.txt

THE RULE, fixed before either file existed:
  - the dev fold only (n % 10 == 1); never the test fold, which is PR-010's
  - the alignment words, in order, cut into sentences after every word that ends in . ? or !
  - each word lower-cased and stripped of the punctuation around it (an apostrophe inside stays)
  - a sentence is kept if it has MIN_WORDS to MAX_WORDS words, every one of them letters and
    apostrophes only (no digits, no symbols), and it is not a repeat of one already kept
  - DEV takes the dialogues with (n // 10) even (1, 21, 41, ...), EVAL the others (11, 31, ...):
    no dialogue gives sentences to both

The EVAL file is written once, its SHA-256 pinned in simulate.py, and read by no run until PR-001's
confirmation. This module prints the EVAL file's counts and hash, never its sentences.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

MIN_WORDS, MAX_WORDS = 8, 14                  # the simulator's turn lengths (simulate.COHORTS)
OUT = pathlib.Path(__file__).parent / "sentences"
WORD = re.compile(r"^[a-z]+(?:'[a-z]+)*$")


def pool(n: int) -> str:
    """Which set a dev-fold dialogue's sentences go to."""
    if n % 10 != 1:
        raise ValueError(f"dialogue {n} is not in the dev fold")
    return "dev" if (n // 10) % 2 == 0 else "eval"


def sentences(alignments: list) -> list[str]:
    """The kept sentences of one dialogue, in order: alignments are [[word, [start, end], speaker], ...]."""
    out, cur = [], []
    for w, *_ in alignments:
        cur.append(w)
        if re.search(r"[.?!][\"')]*$", w):
            words = [x.lower().strip(".,!?;:\"()-") for x in cur]
            if MIN_WORDS <= len(words) <= MAX_WORDS and all(WORD.match(x) for x in words):
                out.append(" ".join(words))
            cur = []
    return out


def extract(root: pathlib.Path) -> dict[str, list[str]]:
    """Both sets from the dev-fold transcripts under root (data/dailytalk/data_stereo)."""
    got: dict[str, list[str]] = {"dev": [], "eval": []}
    ids = sorted(int(p.stem) for p in pathlib.Path(root).glob("*.json") if p.stem.isdigit() and int(p.stem) % 10 == 1)
    for n in ids:
        a = json.loads((pathlib.Path(root) / f"{n}.json").read_text(encoding="utf-8"))["alignments"]
        for s in sentences(a):
            if s not in got["dev"] and s not in got["eval"]:
                got[pool(n)].append(s)
    return got


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main(root: str = "data/dailytalk/data_stereo") -> None:
    got = extract(pathlib.Path(root))
    OUT.mkdir(exist_ok=True)
    for split, lines in got.items():
        path = OUT / f"whole-{split}.txt"
        if path.exists():
            raise SystemExit(f"{path} exists: the sets are written once")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for split, lines in got.items():
        lens = sorted(len(s.split()) for s in lines)
        print(f"{split}: {len(lines)} sentences, {lens[0]}-{lens[-1]} words, median {lens[len(lens) // 2]}")
    print(f"whole-eval.txt sha256 {sha256(OUT / 'whole-eval.txt')} (its sentences are not printed)")


if __name__ == "__main__":
    main()
