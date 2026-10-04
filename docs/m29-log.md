# M29 run log: PR-001, a phrase-level completeness check

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m28-log.md` and those it names. The registration:
`docs/preregistration.md`, PR-001 (2026-09-11), the one registration still open.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-03 | the simulator's turns, DEV seeds 0 and 1000, laptop | 10–18% of turns are a whole sentence; 27–38% end on a T1 word: PR-001 cannot pass S3 on these turns with any real check | — | — |
| 2 | 10-03 | — | the author chose a recorded deviation, before any T1b exists | — | — |
| 3 | 10-03 | the rule over DailyTalk's dev fold, laptop | 38 DEV and 30 EVAL sentences, 8 to 14 words | `simsentences.py`, `sentences/`, `simulate.py`, `sweep.py`, tests, PR-001's Deviations | 8eaac58 |
| 4 | 10-03 | the language model over every DEV sentence and its prefixes, laptop | P(end) separates a whole sentence from an unfinished one: AUC 0.94, 0.91 after a content word | — | — |
| 5 | 10-03 | T1 against T1b at five cut-offs, DEV seeds 0 and 1000, whole-sentence turns, laptop | the rule picks 0.01: disfluent cut-offs 63% → 46% and 76% → 50%, the clean wait unchanged; frozen | `completion.py`, `policies.py`, `sweep.py`, tests | this commit |

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

## 5. Commit 2: T1b, chosen on DEV and frozen

**The check.** T1b holds the floor 400 ms longer where T1 would (its word list), and also wherever a
small pretrained language model judges the sentence unfinished: the probability it gives to the
sentence ending right after the words so far (a next token that begins with . ? or !, or ends the
text) is below a cut-off. The model is Qwen2.5-0.5B-Instruct as published, R0 to R2's LLM;
nothing in it was fitted to these sentences. The idea is TurnGPT's (Ekstedt & Skantze, 2020).
`completion.py` scores and caches; `policies.CompletionThreshold` is the policy. It is not in the
default sweep, which must not load a model.

**A strict extension of T1 (V4).** T1b holds wherever T1 holds and only adds holds, so at no instant
is its silence requirement shorter than T1's. A test drives T1 and T1b over every DEV turn of every
cohort, in both turn modes, on both development seeds, and checks that T1b never decides earlier
than T1, so never cuts off a turn T1 spared.

**Does the signal exist?** Before any policy: the model over the 38 DEV sentences and every prefix a
pause can leave (321 texts, 267 ms each on the laptop's CPU). P(end) at a whole sentence, median
0.233; at an unfinished prefix, 0.002. A whole sentence scores above an unfinished prefix with
probability 0.94 (AUC), and 0.91 against a prefix ending on a content word, the case T1's list
cannot see.

**The cut-off, by a rule written before the tuning run** (`completion.RULE`): the largest cut-off in
GRID for which, on each DEV seed (0 and 1000, 250 turns a cohort, whole-sentence turns), T1b's clean
median wait is at most T1's + 25 ms and its clean p90 at most T1's + 50 ms: half of S3's allowances,
a margin for EVAL.

| cut-off | seed | disfluent cut-offs, T1 → T1b | slow cut-offs | clean median wait, T1b − T1 | clean p90, T1b − T1 |
|---:|---:|---|---|---:|---:|
| 0.005 | 0 | 63% → 50% | 12% → 5% | +0 ms | +0 ms |
| 0.005 | 1000 | 76% → 53% | 10% → 6% | +0 ms | +0 ms |
| 0.01 | 0 | 63% → 46% | 12% → 4% | +0 ms | +0 ms |
| 0.01 | 1000 | 76% → 50% | 10% → 4% | +0 ms | +0 ms |
| 0.02 | 0 | 63% → 41% | 12% → 4% | +2 ms | +382 ms |
| 0.02 | 1000 | 76% → 46% | 10% → 4% | +1 ms | +1 ms |
| 0.05 | 0 | 63% → 37% | 12% → 3% | +5 ms | +382 ms |
| 0.05 | 1000 | 76% → 39% | 10% → 4% | +3 ms | +382 ms |
| 0.1 | 0 | 63% → 30% | 12% → 2% | +6 ms | +382 ms |
| 0.1 | 1000 | 76% → 35% | 10% → 2% | +6 ms | +382 ms |

The rule picks **0.01**; at 0.02 one seed's clean p90 moves by 382 ms, because more than a tenth
of clean turns are then held at their true end. Frozen in `completion.T1B_THETA`; the run is
`runs/turn-taking/pr001-dev.json`. Development, not the test: DEV sentences, the development seeds.
On these turns T1 itself cuts off 63% and 76% of disfluent turns on the two seeds.

**The confirmation, now fixed in code.** `--confirm PR-001` sweeps T1 and T1b only, compares them in
each cohort (the family of 3, Holm-adjusted), computes S1 to S4 and writes
`runs/turn-taking/pr001.json`. It is refused unless the command is exactly the registered one
(`--split eval --turns sentence --seed 2000 --n 250`), and refused if that file exists.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:
the full suite, 898 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold),
and the one slow test, which loads the real model: passed;
and the confirmation path end to end on DEV (seed 5, 40 turns a cohort), from a throwaway folder:
it swept T1 and T1b, compared them, printed S1 to S4 and wrote its file. A plumbing check, not a result.
**Not run:** anything on the EVAL sentences.

## 6. Next

Commit 3, the author's: the one confirmation run, on the laptop, a few minutes:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 python -m s2slab.sweep --split eval --turns sentence --seed 2000 --n 250 --confirm PR-001
```

Then the result recorded under PR-001's Results verbatim, with this commit's hash (V2), and only
after that the error analysis of T1b's remaining cut-offs, labelled exploratory.
