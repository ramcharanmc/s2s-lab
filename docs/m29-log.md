# M29 run log: PR-001, a phrase-level completeness check

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m28-log.md` and those it names. The registration:
`docs/preregistration.md`, PR-001 (2026-09-11), the one registration still open.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-03 | the simulator's turns, DEV seeds 0 and 1000, laptop | 10–18% of turns are a whole sentence; 27–38% end on a T1 word: PR-001 cannot pass S3 on these turns with any real check | — | — |
| 2 | 10-03 | — | the author chose a recorded deviation, before any T1b exists | — | — |
| 3 | 10-03 | the rule over DailyTalk's dev fold, laptop | 38 DEV and 30 EVAL sentences, 8 to 14 words | `simsentences.py`, `sentences/`, `simulate.py`, `sweep.py`, tests, PR-001's Deviations | this commit |

## 1. What reading the simulator showed

PR-001 asks whether a check on whether the sentence is finished (T1b) cuts off fewer disfluent
speakers than T1's word list, with clean speakers waiting at most 50 ms longer at the median (S3).
The registration's first step, the DEV/EVAL sentence split, was done on 2026-09-12 (`485f97a`),
and `sweep.py` refuses the EVAL sentences without `--confirm`.

But every simulated turn is the first 8 to 14 words of a 14-word sentence. Measured on the DEV
seeds already used, 250 turns a cohort:

| | clean | disfluent | slow |
|---|---|---|---|
| turns that are a whole sentence | 14–16% | 16–18% | 10–16% |
| turns whose last word is on T1's list, so T1 holds at the true end | 27–30% | 31–33% | 34–38% |

A pause and the true end both fall at random word boundaries of one sentence, so whether the words
so far are a sentence says almost nothing about whether the speaker has finished. A real
completeness check would hold at most true ends: about +400 ms at the clean median, against S3's
50. Run as registered, PR-001 would fail on the simulator, not on T1b.

## 2. The choice

| | What | |
|---|---|---|
| A | run as registered | a near-certain "not supported", for a reason known in advance |
| **B, chosen** | **a recorded deviation, before any T1b exists: whole-sentence turns** | a test of the method; question, criteria, seed and procedure unchanged |
| C | withdraw PR-001, register PR-014 | the same work as B, and a fourteenth registration |

## 3. Commit 1: the turns, the sentences, the deviation

- **`simulate.py`**: `turns="sentence"`, every turn one whole sentence, its end where the sentence
  ends. `turns="prefix"` stays the default and draws in the same order as before, so every seed gives
  the turns it always gave (a test checks). The two sentence files are pinned by SHA-256, and a
  changed file is refused. A commented-out scratch block and a pasted shell transcript at the end of
  the file are removed.
- **`simsentences.py`** and **`sentences/whole-dev.txt`, `whole-eval.txt`**: the rule and what it
  drew from DailyTalk's dev fold (50 dialogues on this machine): a sentence ends after . ? or !; it is
  kept at 8 to 14 words, letters and apostrophes only, no repeats; DEV from dialogues 1, 21, 41, ...,
  EVAL from 11, 31, 51, .... 38 DEV and 30 EVAL sentences. The share of T1's words is 0.34 in
  DEV and 0.38 in EVAL (a test holds them within 0.10); 2 DEV sentences and no EVAL sentence end on
  one.
- **`sweep.py`**: `--turns sentence`, recorded in the results file; `--confirm PR-001` is refused
  without it, so the one run on seed 2000 cannot be spent on the old turns.
- **PR-001's Deviations**: the reason, the numbers, what changed and what did not, EVAL's hash, the
  confirmation command.

**Disclosures.**
- EVAL's sentences were not displayed when the file was written, to me, who builds T1b, or in
  this log: only the rule's counts and the hash. The file itself is in the repo for anyone to read.
- While looking for text on the laptop, I opened one transcript of DailyTalk's test fold, PR-010's
  locked data (dialogue 0, its first four words). PR-010 is finished, and the test fold is not used
  here: the rule refuses it.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:
the full suite, 887 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold);
and the published sweep (`runs/turn-taking/sweep.json`: seed 1000, 250 turns a cohort, 30 cells)
re-run with this commit's code, every cell the same as the file's.

## 4. Next

Commit 2: T1b, T1's list or a small pretrained language model's judgement that the sentence is not
finished, its cut-off chosen on DEV seeds 0 and 1000 with whole-sentence turns only; the test that
T1b never cuts off a turn T1 spared; then T1b frozen. Commit 3 is the author's one confirmation run.
