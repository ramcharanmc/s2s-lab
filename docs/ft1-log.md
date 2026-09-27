# FT-1 run log (M23)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Written on 2026-09-27, after the milestone, from the record: the commits, `docs/preregistration.md`
(PR-010), `docs/cost-ledger.md`, `runs/ft1/`. Where an event is known only from the working session
and no file holds it, the entry says so. Commit times are CDT. The same log for FT-2 is
`docs/ft2-log.md`, for FT-3 `docs/ft3-log.md`.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 09-24 19:10 | — | the measure: turn gaps in two-channel audio | `src/s2slab/turngaps.py`, tests | fae0e21 |
| 2 | 09-24 19:25 | `r5_bench.py::dialogues --limit 50`, A10 | dev: base Moshi 478 ms from the reference | `src/s2slab/dailytalk.py`, the Modal entry point | 97c3a06 |
| 3 | 09-25 08:16 | — | PR-010 registered, with the Vast kit | `vast/`, the LoRA adapter in `r5.py`, `::pr010` | 4df9ef3 |
| 4 | 09-25, morning | `bash /workspace/ft1.sh`, Vast H100 | five launches; the fifth trained | — (fixed after the run, entry 8) | — |
| 5 | 09-25, to the evening | the training | 2,000 steps in 11 h 22 min, no resume | — | — |
| 6 | 09-25, evening | `::pr010 --confirm PR-010`, A10 | **crashed**: out of GPU memory | `pr010_remote`: L40S, memory recorded | 3d36f74 |
| 7 | 09-25 21:35 | `::pr010 --confirm PR-010`, L40S | **not supported, moved away** | results: evidence and text | 2f5201a |
| 8 | 09-26 07:02 | — | the Vast kit fixed by what the machine hit | `vast/ft1.sh`, `ft1_prep.py`, README | db6295e |

## 1. The measure (fae0e21)

The plan's go/no-go (page 17 §7.1) asks for overlap ratio and turn-gap distribution against a
human reference. `turngaps.py`, after Heldner & Edlund (2010): each channel's inter-pausal units,
then the offset at every change of floor (a gap if positive, an overlap if negative), and the
Wasserstein-1 distance between two offset distributions. Tested on made-up channels. 783 tests.

## 2. The reference and the dev run (97c3a06)

- **Reference:** held-out DailyTalk itself, FT-1's own training data, so the claim became "toward
  the timing of the dialogues it was trained on", not "toward human".
- **Dev run:** base Moshi answering the user channel of 50 dev dialogues. Modal A10, 1,071 s, $0.335.
- **Result:** 478 ms from the reference (interval 406 to 564); median gap 160 ms against 420; 42
  overlapping turn changes against none.
- **What it changed:** DailyTalk has no overlap at all, so the plan's overlap ratio could only be
  "closer" by falling to zero. The registered measure became the gap distribution.

## 3. PR-010 registered (4df9ef3)

Registered before any training and before the test dialogues were downloaded. The training recipe
is kyutai's example, unchanged: LoRA rank 128, learning rate 2e-6, 2,000 steps. The kit in `vast/`
had not been run on a machine. 797 tests.

**What the registration did not include, and should have:** a look at the trained adapter on dev
before the test. It went from training straight to the test (entry 7).

## 4. The Vast machine: five launches

`runs/ft1/training/ft1.log` records five launches of `ft1.sh`, at 13:29, 13:36, 13:42, 13:53 and
13:56 by the machine's clock. Each begins with the same warning: the machine cannot destroy itself.

| What broke | Evidence | Fixed in entry 8 |
|---|---|---|
| Self-destroy did not work | the warning on all five launches | reads `CONTAINER_ID` and the instance's key from PID 1 |
| The dataset download hit Hugging Face's rate limit, twice | the ledger's row | the download waits out the 429s |
| `CUDA_VISIBLE_DEVICES` was not set, so the fourth launch crashed before step 1 | PR-010, "Training, for the record"; two launch configs, identical | set for `torchrun` |
| The ssh proxy timed out | `vast/README.md` | the README gives the direct address |

Which of the first three launches failed on which cause is not in the log. The fourth and fifth
wrote their configs ("fresh start"); the fifth trained.

The machine was destroyed by hand. About one hour of it went on set-up and the copy-off.

## 5. The training

- **Run:** one H100 80 GB, on-demand, 2,000 steps, no resume, peak 39.6 GB.
- **Time:** 11 h 22 min, about 20 s a step. The plan said 3 h.
- **Loss:** 1.674 over the first 50 steps, 1.137 over the last 100.
- **Cost:** $24.29, Vast's bill. The plan's budget for all three fine-tunes on Vast was $5.86.
- **Kept:** the last three checkpoints only (`num_ckpt_keep: 3`), so steps 500, 1,000 and 1,500
  were gone when they were wanted (entry 7).
- **Would have been lost:** the kit's hard cap was 10 hours. Had self-destroy worked, it would have
  destroyed the machine near step 1,700.

## 6. The first registered run crashed (fixed in 3d36f74)

- **Failure:** base Moshi finished its 100 dialogues, then the container ran out of GPU memory
  loading Moshi + LoRA. No run file was written; no measure of either model was seen.
- **Cause, as first written:** about 5 GiB left allocated by the base pass. **This was wrong**,
  and it was written into the registration as fact without being measured. Mine.
- **Fix:** the one repeat the registration allows, on an L40S (48 GB), recording the memory the
  base pass leaves behind.
- **Cost:** about $0.6 of A10 time. Log: `results/ft1-pr010-crash1.txt`.

## 7. The registered run (2f5201a)

Modal L40S, 3,658 s, $2.011, commit 3d36f74, clean tree.

| 100 test dialogues | base Moshi | Moshi + LoRA | reference |
|---|---:|---:|---:|
| distance to the reference | 496 ms | 1,035 ms | — |
| median gap | 160 ms | 1,220 ms | 420 ms |
| overlapping turn changes | 85 | 2 | 0 |
| turn changes | 491 | 355 | 456 |

- **Verdict:** not supported, and the movement is away. Gain -539 ms, interval -685 to -402.
- **Predictions:** one right (overlap falls sharply), three wrong (the distance, the median gap,
  "the LoRA may move little").
- **The correction:** the repeat measured 16 MiB left after the base pass, and Moshi + LoRA
  peaking at 29.3 GiB on its own. No A10 run could have held it. The first diagnosis is left as
  written in the registration, with a correction entry under it.
- **From the session, not in a file:** the script that built this commit's text had a failed edit
  step and still copied its output, which put the estimate "about $24" into the repo where the
  bill, $24.29, belonged. It was caught and corrected before the commit.
- Evidence: `runs/ft1/pr010.json`, `pr010-report.md`, `training/`.

## 8. The kit fixed (db6295e)

Every fix is something the machine hit (entry 4), plus the hard cap raised from 10 to 15 hours.
**Untested on a machine**: no job has needed Vast since. 800 tests.

## What the milestone taught

- **Look at the trained model on dev before registering the test.** The 50 dev dialogues would
  have cost about $0.35 and shown the overshoot. FT-2 and FT-3 did this.
- **Keep the intermediate checkpoints.** FT-2 kept every adapter.
- **A cause that was not measured is a hypothesis.** Write it as one, with the measurement that
  would test it.
- **A plan's time estimate is not a budget.** Twenty seconds a step was knowable from the first
  ten steps; nothing checked it against the cap.

## Money

| | |
|---|---:|
| Vast, the training (billed) | $24.29 |
| Modal, the dev run | $0.335 |
| Modal, the crashed run (estimate) | about $0.6 |
| Modal, the registered run | $2.011 |
| **The milestone** | **about $27.2** |
