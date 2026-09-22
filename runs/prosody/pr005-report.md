39 test pairs · confirms PR-005 · "Judging by their tone of voice, is the caller upset?" · shuffled = mean of 1000 deals, seed 20260921

| condition | SAD: answered right · ranked right (of 13) | FEA: answered right · ranked right (of 13) | ANG: answered right · ranked right (of 13) |
|---|---|---|---|
| none | 0 · 3 | 0 · 5 | 1 · 8 |
| tagger | 7 · 10 | 7 · 8 | 11 · 13 |
| oracle | 13 · 13 | 13 · 13 | 13 · 13 |
| shuffled | 3.2 · 6.1 | 3.1 · 5.6 | 3.1 · 6.8 |

with no tag the LLM says Yes to 73 of 78 clips (median p_yes 0.78); it puts a median 1.00 of its probability on Yes or No
median p_yes by the word shown: neutral 0.10, sad 0.91, fearful 0.79, angry 0.82

| criterion | pairs answered right | shuffled deals at or above it | permutation p | registered bar | |
|---|---:|---:|---:|---|---|
| S1 fear | 7 of 13 | 15 of 1000 | p = 0.015 | at least 8 and p < 0.05 | FAIL |
| S2 anger | 11 of 13 | 0 of 1000 | p < 0.001 | at least 7 and p < 0.05 | PASS |
| S3 sadness | 7 of 13 | 25 of 1000 | p = 0.025 | at least 7 and p < 0.05 | PASS (reported, not deciding) |

PR-005: NOT SUPPORTED: failed fearful

| pair | neutral clip tagged | p(Yes) | loaded clip tagged | p(Yes) | answered | ranked | same transcript |
|---|---|---:|---|---:|---|---|---|
| sad-t01 | neutral | 0.08 | fearful | 0.81 | right | right | yes |
| sad-t02 | sad | 0.85 | sad | 0.85 | **wrong** | wrong | yes |
| sad-t03 | neutral | 0.11 | sad | 0.94 | right | right | yes |
| sad-t04 | angry | 0.64 | sad | 0.79 | **wrong** | right | yes |
| sad-t05 | neutral | 0.08 | sad | 0.75 | right | right | no |
| sad-t06 | angry | 0.82 | fearful | 0.82 | **wrong** | right | no |
| sad-t07 | angry | 0.57 | neutral | 0.04 | **wrong** | wrong | yes |
| sad-t08 | neutral | 0.10 | sad | 0.88 | right | right | yes |
| sad-t09 | neutral | 0.10 | sad | 0.88 | right | right | yes |
| sad-t10 | sad | 0.79 | sad | 0.79 | **wrong** | wrong | yes |
| sad-t11 | neutral | 0.11 | sad | 0.98 | right | right | no |
| sad-t12 | sad | 0.85 | sad | 0.97 | **wrong** | right | no |
| sad-t13 | neutral | 0.07 | sad | 0.91 | right | right | yes |
| fea-t01 | sad | 0.88 | fearful | 0.82 | **wrong** | wrong | no |
| fea-t02 | neutral | 0.11 | fearful | 0.90 | right | right | yes |
| fea-t03 | neutral | 0.12 | fearful | 0.91 | right | right | no |
| fea-t04 | neutral | 0.10 | fearful | 0.61 | right | right | no |
| fea-t05 | fearful | 0.74 | angry | 0.82 | **wrong** | right | yes |
| fea-t06 | sad | 0.91 | angry | 0.72 | **wrong** | wrong | yes |
| fea-t07 | neutral | 0.16 | neutral | 0.16 | **wrong** | wrong | yes |
| fea-t08 | neutral | 0.16 | neutral | 0.16 | **wrong** | wrong | no |
| fea-t09 | neutral | 0.07 | angry | 0.83 | right | right | no |
| fea-t10 | angry | 0.90 | fearful | 0.86 | **wrong** | wrong | yes |
| fea-t11 | neutral | 0.12 | angry | 0.93 | right | right | yes |
| fea-t12 | neutral | 0.12 | fearful | 0.88 | right | right | yes |
| fea-t13 | neutral | 0.07 | fearful | 0.86 | right | right | yes |
| ang-t01 | neutral | 0.08 | angry | 0.92 | right | right | yes |
| ang-t02 | angry | 0.57 | angry | 0.70 | **wrong** | right | no |
| ang-t03 | neutral | 0.04 | angry | 0.57 | right | right | yes |
| ang-t04 | fearful | 0.74 | angry | 0.90 | **wrong** | right | no |
| ang-t05 | neutral | 0.10 | angry | 0.79 | right | right | no |
| ang-t06 | neutral | 0.10 | angry | 0.90 | right | right | no |
| ang-t07 | neutral | 0.16 | angry | 0.90 | right | right | no |
| ang-t08 | neutral | 0.16 | fearful | 0.87 | right | right | no |
| ang-t09 | neutral | 0.16 | angry | 0.90 | right | right | no |
| ang-t10 | neutral | 0.07 | angry | 0.72 | right | right | yes |
| ang-t11 | neutral | 0.09 | angry | 0.82 | right | right | no |
| ang-t12 | neutral | 0.07 | angry | 0.66 | right | right | yes |
| ang-t13 | neutral | 0.12 | angry | 0.93 | right | right | yes |

the tagger alone told apart (neutral clip tagged neutral, loaded clip tagged anything else): SAD 7 of 13, FEA 7 of 13, ANG 11 of 13

saved to runs/prosody/pr005.json
