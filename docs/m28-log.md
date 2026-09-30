# M28 run log: the ladder on one card

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. Times are CDT. The earlier logs: `docs/m27-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 09-29 | — | the design: every rung on one card; the L40S, because R4 does not fit the A10 | — | — |
| 2 | 09-29 | the tests, laptop, scratch | R0 to R4 in one L40S container; R5's two models on the L40S; the overwrite guard honours `--tag` | `modal_app/r4_bench.py`, `r5_bench.py`, tests | 13d40dc |
| 3 | 09-29, night | `modal run modal_app/r5_bench.py --model r5-moshi ...`, by the author | refused by Modal before any container: the file has three entry points and the command named none. The command was mine; so is a test, found failing once run A saved its file | the usage lines name `::main`; the test | this commit |
| 4 | 09-29, night | A's smoke test, 8 clips, by the author | the new layer built; six rungs in one L40S container; the same replies as the earlier runs on the same clips; \$0.111 | — | — |
| 5 | 09-29, night | run B: Moshi, 200 calls, the L40S | 93 of 200 answered, gap p50 140 ms; RTF 0.45; \$1.010 | — | — |
| 6 | 09-29, night | run C: PersonaPlex, 200 calls, the L40S | 190 of 200 answered, gap p50 440 ms; RTF 0.50; set-up 3.84 s; \$1.543 | — | — |
| 7 | 09-29, night | run A: R0 to R4, 200 clips each, one L40S container | every rung answered all 200; R4 6.86 s p50; this container 27% faster than R4's earlier one on the same card; \$1.212 | — | this commit |

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

## 3. The first command, mine

`13d40dc` committed and pushed; 880 tests passed on the author's machine. The first run stopped
before any container: "Specify a Modal Function or local entrypoint to run". `r5_bench.py` has
three entry points (`main`, `dialogues`, `pr010`) and `r4_bench.py` two (`main`, `pairs`); a file
with more than one needs the one named, `::main`. I copied the command from the files' own usage
lines, written before the second entry point was added, and did not check. Cost: nothing. The usage
lines of both files now name the entry point.

**A second fault, mine, found when the records were built:** part 1's test that a tagged run is
refused only under its own name checked that the name `evalset-modal-l40s-ladder.json` was free.
Run A saved that file, so the test failed from then on (1 of 880, in a scratch copy with the run
files). The test now checks that the recorded names, the untagged run's and the ladder run's, are
refused, and that a tag no run has used is free.

## 4. Run A's smoke test (at 13d40dc)

`r4_bench.py::main`, the six rungs, `--limit 8`: the new layer built (R0 to R2's weights, seconds),
all six loaded and answered 8 clean clips in one L40S container, 205 s end to end, \$0.111. On
those 8 clips every rung's replies were the length of its earlier run's, all but one turbo reply,
one word shorter; and the times matched the earlier runs within a few per cent:
R4 13.33 against 13.19 s on the first clip. R4's 12.9 s median over the 8 was the clips' long
replies, not the container. Not evidence: it is in `results/`.

## 5. The three runs (at 13d40dc)

The author ran them side by side. All three completed; no failure.

| Run | What | End to end | Cost, at \$1.95 an hour |
|---|---|---:|---:|
| A | R0, R1, R2, R2 turbo, R3, R4: 200 clips each, one container | 2,237 s | \$1.212 |
| B | Moshi, 200 calls | 1,865 s | \$1.010 |
| C | PersonaPlex, 200 calls | 2,848 s | \$1.543 |

With the smoke test, \$3.876, against the estimate of about \$4.50.

**The table on one card**, every number from the run files (`runs/<rung>/evalset-modal-l40s-ladder.json`):

| Rung | First audio p50 | p90 | WER, normalized | Peak memory | \$ per 1,000 turns |
|---|---:|---:|---:|---:|---:|
| R0 | 300 ms | 509 ms | 11.9% | 1.8 GB | 0.18 |
| R1 | 245 ms | 358 ms | 11.9% | 1.7 GB | 0.20 |
| R2, small.en | 279 ms | 398 ms | 9.6% | 1.7 GB | 0.22 |
| R2, large-v3-turbo | 275 ms | 393 ms | 8.6% | 1.7 GB | 0.23 |
| R3 | 632 ms | 1,202 ms | none in the loop (9.2% asked to transcribe) | 17.0 GB | 0.40 |
| R4 | 6,859 ms | 12,053 ms | none in the loop | 22.6 GB | 4.03 |
| R5, Moshi | gap 140 ms; 93 of 200 answered | 980 ms | none in the loop | 16.7 GB | 4.95 a 1,000 calls |
| R5, PersonaPlex | gap 440 ms; 190 of 200 answered | 760 ms | none in the loop | 18.1 GB | 7.56 a 1,000 calls |

Dollars are the mean processing time a turn at the L40S's list rate; a call is 20 s at its mean
real-time factor plus its mean set-up. **The earlier tables mixed the two statistics**: R0's \$0.12
was the mean, R3's \$0.32 and R4's \$5.0 the median (checked from their run files). This table is
the mean throughout, because a bill is a sum.

**Two containers of one card are not one machine.** R4 took 0.73 of its earlier L40S time on the
same clips (paired median), steadily through the run: 925 ms of wait a second of reply here, 1,284
there, with the same card, driver, image and code. R3 took 0.92 of its earlier time. The smoke
test's container matched the earlier speed. So a container can differ by a quarter from another of
the same card, and only rungs in one container compare to the decimal. R0 to R4 are in one; R5's
two runs are on the same card in containers of their own, and their gap is in stream time.

**What the one card changes**, against the table the author presented:

| | On the mixed cards | On one L40S container |
|---|---|---|
| R1 against R0, first audio | 26% sooner (PR-002, T4) | 18% sooner |
| R2 turbo's slow tier, against R1 | +310 ms (PR-004, T4) | +30 ms |
| R3 against R0 | 2.4 times (A10) | 2.1 times |
| R4 | 9.17 s (its own L40S container) | 6.86 s: this container |
| Moshi, real-time factor | 0.69 (A10) | 0.45 |
| PersonaPlex, set-up a call | 5.95 s (A10) | 3.84 s |

The registered results do not change: PR-002 and PR-004 were tests on a T4 and stand there. What
changes is the comparison across rungs. The largest: on a card with room, the larger recognizer's
accuracy costs R2 about 30 ms, not 310.

**A correction to what I told the author after the smoke test.** Its 8 clips ran R0 and R1 at their
A10 times, and I said the cascades run no faster on this card. Run A says otherwise: R0 took 0.68 of
its A10 time on the same clips, every stage faster (recognizer 47 to 38 ms, LLM 315 to 210, voice 80
to 56). The smoke test ran in a slower container than run A. Card and container cannot be told
apart here: the A10 row and this one are different containers too.

**The words.** WER is what it was on every rung, to the decimal (11.9, 11.9, 9.6, 8.6%); 185 to 194
of 200 cascade replies are the same length as before. The full-duplex models are not bit-identical
across cards: Moshi answered 93 callers against 96, PersonaPlex 190 against 192; each still gave
identical conversations when one clip was driven twice in the same run.

**R4 on this card fits, with room.** Peak 22,591 MB. An A10 would not have held it.

**Not changed by this run, and the author's to decide on the slides:** the T4 rows remain the
cascades' deployment numbers; on the L40S, at \$1.95 an hour, every rung costs more a turn than on
its cheapest card.
