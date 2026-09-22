# Cost ledger

One row per paid run, pasted from the run's own cost line (v1 plan §5.1). The
estimate is wall time at the GPU's list rate; Modal's dashboard is the bill.
Rates checked 2026-08-19: T4 \$0.59/h, A10 \$1.10/h.

**Rules:** add the row in the same commit as the run's results, or the next
one. A failed run is still a row, since it was still billed.

| Date | Commit | Command | GPU | Inside fn | End to end | Estimate | Result |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-15 | `d5e0eef` | `modal run modal_app/r0_bench.py` | T4 | — | — | — | failed at warm-up: no C compiler for Triton; fixed in `1ea99ae` |
| 2026-09-15 | `1ea99ae` | `modal run modal_app/r0_bench.py` | T4 | 53 s | 68 s | \$0.011 | R0 on 30 probe clips: p50 827 ms (page 20 §4) |
| 2026-09-16 | `0af142a` | `modal run modal_app/r0_bench.py --evalset --limit 8` | T4 | 45 s | 70 s | \$0.011 | smoke test, 8 clips: not evidence |
| 2026-09-16 | `0af142a` | `modal run modal_app/r0_bench.py --evalset` | T4 | 167 s | 180 s | \$0.030 | R0 on 200 eval clips: p50 655 ms, p90 1130 ms, WER 11.9% (page 20 §5.2; `runs/r0-gpu/`) |
| 2026-09-18 | `b107927` | `modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu` | T4 | 86 s | 96 s | \$0.016 | R0 and R1 on 30 probe clips, one container: R1's first GPU run, dev data |
| 2026-09-18 | `b107927` | `modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu --evalset --tag pr002` | T4 | 324 s | 334 s | \$0.055 | **the PR-002 run**: R1 first audio 491 vs R0 660 ms; not supported (`docs/preregistration.md`) |
| 2026-09-21 | `4920228-dirty` | `modal run modal_app/r0_bench.py --rung r1-gpu,r2-gpu,r2-gpu-turbo` | T4 | 151 s | 158 s | \$0.026 | R1 and both R2 slow tiers on 30 probe clips, one container, dev data: asr 65 / 123 / 376 ms, first audio 512 / 605 / 900 ms. Dirty only by the edits committed next (a stale git lock blocked that commit first) |
| 2026-09-21 | `91048b3` | `modal run modal_app/r0_bench.py --rung r1-gpu,r2-gpu,r2-gpu-turbo --evalset --tag pr004` | T4 | 666 s | 724 s | \$0.119 | **the PR-004 run**: turbo WER 8.6% vs R1 11.9%, accented 22.2% vs 26.5%, first audio 889 vs 574 ms; not supported, S2 failed (`docs/preregistration.md`). Two rows lost to earlier paste-overs (\$0.030 and \$0.055) restored in this commit |
| 2026-09-22 | `fee8c03` | `modal run modal_app/r3_probe.py --split dev` | A10 | — | — | ≈\$0.02 | failed at the first forward pass: no C compiler for Triton, as R0's first run; `gcc` added to the image |
| 2026-09-22 | `fee8c03-dirty` | `modal run modal_app/r3_probe.py --split dev` | A10 | 69 s | 124 s | \$0.038 | R3 (Qwen2-Audio) on the 21 dev pairs, four wordings, dev data: ranks 21 of 21 right, answers 6 · 5 · 5 of 7 right with the `upset` wording. Dirty only by the `gcc` fix, committed next |

**Not in the rows:** image builds on Modal's CPUs (weights download, pip
installs), billed separately at CPU rates. See the dashboard.

## Running total

| | GPU estimate | Plan budget (v2 §8) |
|---|---:|---:|
| Modal | \$0.326 + two failed runs | \$36.93 |
| Vast.ai | \$0 | \$5.86 |