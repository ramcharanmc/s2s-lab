Modal NVIDIA L40S, 580.95.05 · commit 3d36f74 · adapter sha256 7a48bd5823e0
peak GPU memory: base 16,677 MB, LoRA 29,962 MB; left allocated after the base pass: 16 MB

## base Moshi

100 dialogues; the answerer's floor-transfer offsets, ms (negative = overlap)

| | base Moshi | reference (the human on the left) |
|---|---:|---:|
| turn changes | 491 | 456 |
| offset p10 / p50 / p90 | -400 / 160 / 220 | 250 / 420 / 830 |
| overlapping turn changes | 85 | 0 |
| speech inside the other's turn (within) | 98 | 0 |
| overlap ratio, median over dialogues | 0.122 | 0.000 |

distance between the two offset distributions (Wasserstein-1, pooled): 496 ms

## Moshi + FT-1's LoRA

100 dialogues; the answerer's floor-transfer offsets, ms (negative = overlap)

| | Moshi + LoRA | reference (the human on the left) |
|---|---:|---:|
| turn changes | 355 | 456 |
| offset p10 / p50 / p90 | 508 / 1220 / 2752 | 250 / 420 / 830 |
| overlapping turn changes | 2 | 0 |
| speech inside the other's turn (within) | 4 | 0 |
| overlap ratio, median over dialogues | 0.030 | 0.000 |

distance between the two offset distributions (Wasserstein-1, pooled): 1035 ms

PR-010, the registered criterion (docs/preregistration.md), 100 test dialogues:
  distance to the reference: base 496 ms, LoRA 1035 ms
  gain (base - LoRA): -539 ms, 95% bootstrap interval -685 to -402
PR-010: NOT SUPPORTED -- the LoRA is not closer to the reference beyond the interval

saved -> runs/ft1/pr010.json; listen: results/ft1/listen/
3658 s inside the function, 3712 s end to end: about $2.011 at the L40S rate. Modal's dashboard has the exact bill.
