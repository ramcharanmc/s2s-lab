# M28 run log: the ladder on one card

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. Times are CDT. The earlier logs: `docs/m27-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 09-29 | — | the design: every rung on one card; the L40S, because R4 does not fit the A10 | — | — |
| 2 | 09-29 | the tests, laptop, scratch | R0 to R4 in one L40S container; R5's two models on the L40S; the overwrite guard honours `--tag` | `modal_app/r4_bench.py`, `r5_bench.py`, tests | this commit |

## 1. The design

Asked for by the author on 2026-09-29, from his presentation: the table of what each rung buys and
costs mixes cards. R0 to R2 ran on a T4, R3 and both R5 models on an A10, R4 on an L40S. The table's
own footnote says rows from different containers compare only to about ±20%. He asked for every
GPU result to be run again on one card, the evaluations and not the trainings.

**The card is the L40S, not the A10.** R4 (Qwen2.5-Omni-7B, bf16, through transformers) peaked at
22,518 MiB allocated in its eval-set run (`runs/r4-gpu/evalset-modal-l40s.json`). An A10 has 23,028
MiB in all, before CUDA's own share and the allocator's slack; `r5_bench.py` already calls it 22 GB,
and PR-010's first run died of memory on it. The L40S (48 GB) is the cheapest card on Modal that
holds every rung. "A10" and "A10G" in the earlier records are one card: Modal's `A10G` reports
itself as "NVIDIA A10".

| Run | What | Where |
|---|---|---|
| A | R0, R1, R2 (small.en), R2 (large-v3-turbo), R3, R4, all 200 clips | **one L40S container**, `r4_bench.py --tag ladder`: one card and one container for R0 to R4, so neither confound is left between them |
| B | Moshi, 200 calls | the L40S, `r5_bench.py --gpu l40s --tag ladder` |
| C | PersonaPlex, 200 calls | the L40S, the same, on its own image |

R5 cannot share A's container: each full-duplex model brings its own package named `moshi` and
its own torch. Their response gaps are measured in stream time and do not depend on the card; what
the card changes is their real-time factor, and so what a call costs.

**Not run again:**

| What | Why |
|---|---|
| The trainings (FT-1, FT-2, FT-3) | asked for: evaluations only |
| The registered tests: PR-002 and PR-004 (T4), PR-008 and PR-010 (L40S), PR-011 | a registered test run again on another card is not a re-test; each result stands on the card it was registered on |
| Everything that ran on a CPU: PR-003, 005, 007, 009, 012, 013, the endpointing work | no GPU in them |

**What is reported, fixed here before any run.** For every rung of run A: first audio p50 and p90,
the stage medians, WER (normalized, and by cohort), peak GPU memory, and dollars per 1,000 turns
at the L40S rate. For B and C: the response gap p50 and p90, callers answered of 200, the real-time
factor, the set-up per call, peak GPU memory. Every number is reported whatever it says. No claim is
registered: this is a measurement on one card, not a test. The earlier run files stay in `runs/`
untouched; the new ones are named `evalset-modal-l40s-ladder.json`.

**What the one card does not show.** At \\$1.95 an hour, the L40S makes the cascades faster and
dearer per turn than the T4 they would be deployed on. The table on one card compares the
architectures; the T4 rows remain the cascades' deployment numbers.

**Estimate.** A: about 50 min and \\$1.60 (R3 and R4 took 2,425 s together on this card; R0 to R2
took about 830 s on a T4). B: \\$1.00 to \\$1.55 (2,833 s on the A10). C: \\$1.60 to \\$2.45
(4,479 s on the A10). About \\$4.50 in all, at most about \\$6; about 1¼ hours on Modal with the
three run side by side.

## 2. Part 1: the code

- **`modal_app/r4_bench.py`.** R0 to R2's weights (Whisper base.en, small.en, large-v3-turbo and
  Qwen2.5-0.5B) in a layer of their own, after the image's other layers, so those stay cached. The
  image already had R0's libraries at R0's pins and R0's voice. The eval function's timeout goes
  from 90 to 120 minutes.
- **A fault, mine, found in the build:** the entry point's overwrite guard checked the untagged
  name even when `--tag` was given. The untagged R3 and R4 runs on this card are recorded, so a
  tagged run would have been refused for them, before any GPU time. It now checks the name the run
  will be saved under.
- **`modal_app/r5_bench.py`.** `--gpu l40s` runs the eval set on the L40S, on each model's own
  image; the file name carries the card, and the cost line the card's rate. On the L40S only the
  eval set runs: `--limit 8` is the smoke test.
- **Tests:** the R4 image holds R0 to R2's models; a tagged run is refused only under its own
  name; R5 on the L40S saves under its own name and uses its own functions.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 875
passed and 5 skipped (the skipped ones for audio the scratch copy does not hold; 880 in the repo),
`tests/test_modal_app.py` 33 of 33, with the Modal apps defined, which builds nothing.
**Not run:** anything on Modal.

**The runs, the author's:**

```bash
modal run modal_app/r4_bench.py --rung r0-gpu,r1-gpu,r2-gpu,r2-gpu-turbo,r3-gpu,r4-gpu --evalset --limit 8 --tag ladder
modal run modal_app/r4_bench.py --rung r0-gpu,r1-gpu,r2-gpu,r2-gpu-turbo,r3-gpu,r4-gpu --evalset --tag ladder
modal run modal_app/r5_bench.py --model r5-moshi --evalset --gpu l40s --tag ladder
modal run modal_app/r5_bench.py --model r5-personaplex --evalset --gpu l40s --tag ladder
```

The first is A's smoke test: the new image layer and six rungs on eight clips. The last two can
start at once, in other terminals: their images are built and cached, and only the card changes.
