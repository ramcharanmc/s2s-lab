# Cost ledger

One row per paid run, pasted from the run's own cost line (v1 plan §5.1). The
estimate is wall time at the GPU's list rate; Modal's dashboard is the bill.
Rates checked 2026-08-19: T4 \$0.59/h.

**Rules:** add the row in the same commit as the run's results, or the next
one. A failed run is still a row, since it was still billed.

| Date | Commit | Command | GPU | Inside fn | End to end | Estimate | Result |
|---|---|---|---|---:|---:|---:|---|
| 2026-09-15 | `d5e0eef` | `modal run modal_app/r0_bench.py` | T4 | — | — | — | failed at warm-up: no C compiler for Triton; fixed in `1ea99ae` |
| 2026-09-15 | `1ea99ae` | `modal run modal_app/r0_bench.py` | T4 | 53 s | 68 s | \$0.011 | R0 on 30 probe clips: p50 827 ms (page 20 §4) |
| 2026-09-16 | `0af142a` | `modal run modal_app/r0_bench.py --evalset --limit 8` | T4 | 45 s | 70 s | \$0.011 | smoke test, 8 clips: not evidence |
| 2026-09-18 | `b107927` | `modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu` | T4 | 86 s | 96 s | \$0.016 | R0 and R1 on 30 probe clips, one container: R1's first GPU run, dev data |
| 2026-09-18 | `b107927` | `modal run modal_app/r0_bench.py --rung r0-gpu,r1-gpu --evalset --tag pr002` | T4 | 324 s | 334 s | \$0.055 | **the PR-002 run**: R1 first audio 491 vs R0 660 ms; not supported (`docs/preregistration.md`) |

**Not in the rows:** image builds on Modal's CPUs (weights download, pip
installs), billed separately at CPU rates. See the dashboard.

## Running total

| | GPU estimate | Plan budget (v2 §8) |
|---|---:|---:|
| Modal | \$0.123 + one failed run | \$36.93 |
| Vast.ai | \$0 | \$5.86 |