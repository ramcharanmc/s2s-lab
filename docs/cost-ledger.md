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
| 2026-09-22 | `5c1f3aa` | `modal run modal_app/r3_probe.py --split test --confirm PR-006 --wording upset --save runs/prosody/pr006.json` | A10 | 66 s | 200 s | \$0.061 | **the PR-006 run**: R3 ranks 13 · 13 · 13 of 13 pairs right (Claim A supported), answers 11 · 11 · 9 right against the side-channel's 7 · 7 · 11 (Claim B not supported: anger, p = 0.21) |
| 2026-09-22 | `4c5905d-dirty` | `modal run modal_app/r3_probe.py::own --save runs/prosody/ownvoice-r3.json` | A10 | 42 s | 47 s | \$0.014 | the own-voice mini set, dev-only: R3 ranks 3 of 5 pairs right under the registered wording, the Yes-lean gone |
| 2026-09-23 | `f3e6982` | `modal run modal_app/r3_probe.py::evalset --rung r3-gpu --limit 8` | A10 | 60 s | 66 s | \$0.020 | R3 rung smoke test, 8 clean clips: total p50 1.63 s, peak 16.9 GB; found the transcript wrapper and the token cap (fixed in `960b491`) |
| 2026-09-23 | `960b491` | `modal run modal_app/r3_probe.py::evalset --rung r0-gpu,r1-gpu,r3-gpu` | A10 | 718 s | 724 s | \$0.221 | **the R3 row**, one container: R0 0.44 s, R1 0.33 s to first audio, R3 1.06 s (p90 2.02); heard WER 9.1% re-scored; peak 17.0 GB |
| 2026-09-23 | `b8092b9` | `modal run modal_app/r4_bench.py --rung r4-gpu` | L40S | — | — | ≈ \$0.01 (failed) | R4 smoke test: image built (Omni + R3 weights), then died at Qwen2.5-Omni's processor load -- Pillow and torchvision missing (fixed in `34a4209`) |
| 2026-09-23 | `34a4209` | `modal run modal_app/r4_bench.py --rung r4-gpu` | L40S | 328 s | 333 s | \$0.180 | R4 smoke test, 30 probe clips: 8.26 s p50, leading silence 200 ms; found that the Talker samples its voice (seeded in `339092a`) |
| 2026-09-23 | `339092a` | `modal run modal_app/r4_bench.py --rung r3-gpu,r4-gpu --evalset` | L40S | 2425 s | 2433 s | \$1.318 | **the R4 row**, one container: R3 0.69 s, R4 9.17 s p50 (p90 17.3); R4's wait = 1.33 s per second of its reply; 22.5 GB |
| 2026-09-23 | `a677969` | `modal run modal_app/r4_bench.py::pairs --split dev` | L40S | 340 s | 362 s | \$0.196 | PR-008 development: R3 and R4 answer the 21 dev prosody pairs; R4 5 wider / 11 narrower; the seeding holds |
| 2026-09-23 | `f4da0bc` | `modal run modal_app/r4_bench.py::pairs --split test --confirm PR-008 --save runs/prosody/pr008.json` | L40S | ≈ 820 s | ≈ 840 s | ≈ \$0.45 | **the PR-008 run**: R4 14 wider / 16 narrower -- not supported, no direction. Times and cost estimated from the saved turn times: the printed cost line was overwritten (PR-008 Deviations); Modal's dashboard has the exact bill |
| 2026-09-24 | `c1adf16` | `modal run modal_app/r5_bench.py` | A10 | — | — | ≪ \$0.01 (failed) | R5 smoke test: image built (torch 2.8, moshi 0.2.13, the moshiko weights), then died loading the probe clips -- Praat missing (fixed in `bef00df`) |
| 2026-09-24 | `bef00df` | `modal run modal_app/r5_bench.py` | A10 | 379 s | ≈ 386 s | \$0.118 | R5 smoke test, Moshi, 30 probe clips, no lead-in: it greets 0.4-2 s into every clip, over the caller; 24 of 30 unanswered within a 4 s tail (lead-in and 8 s tail in `236486b`). End-to-end time back-computed from the cost line; the printed output was overwritten by the next smoke test |
| 2026-09-24 | `43b747b` | `modal run modal_app/r5_bench.py` | A10 | 521 s | 529 s | \$0.162 | R5 smoke test, Moshi, 3.2 s lead-in: opens every call ("Hello, how is it going?"), 17 of 30 unanswered within 8 s; RTF 0.70, 16.6 GB. This run also built the PersonaPlex image (CPU, not in the row) |
| 2026-09-24 | `43b747b` | `modal run modal_app/r5_bench.py --model r5-personaplex` | A10 | 793 s | 806 s | \$0.246 | R5 smoke test, PersonaPlex: answers 30 of 30, gap p50 410 ms; its greeting lasts to 3.28 s, one frame into the caller (lead-in lengthened in the next commit); RTF 0.77, 18.1 GB, load 44 s |

**Not in the rows:** image builds on Modal's CPUs (weights download, pip
installs), billed separately at CPU rates. See the dashboard.

## Running total

| | GPU estimate | Plan budget (v2 §8) |
|---|---:|---:|
| Modal | \$3.312 + four failed runs | \$36.93 |
| Vast.ai | \$0 | \$5.86 |