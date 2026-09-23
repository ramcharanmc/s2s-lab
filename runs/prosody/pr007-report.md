38 fresh pairs · confirms PR-007 · absolute tagger against the per-caller baseline · 268 baseline clips for 27 speakers · commit a591503

absolute tagger (tagger.json):
| emotion | pairs | both right | told apart | neutral clip tagged neutral | loaded clip tagged its emotion |
|---|---:|---:|---:|---:|---:|
| SAD | 12 | 7 | 8 | 10 | 9 |
| FEA | 13 | 5 | 8 | 11 | 6 |
| ANG | 13 | 8 | 10 | 10 | 11 |

pairs not told apart:
  sad-f03: neutral clip tagged neutral (0.94), loaded clip tagged neutral (0.79)
  sad-f05: neutral clip tagged sad (0.55), loaded clip tagged sad (0.87)
  sad-f08: neutral clip tagged sad (0.89), loaded clip tagged sad (0.83)
  sad-f10: neutral clip tagged neutral (0.98), loaded clip tagged neutral (0.98)
  fea-f01: neutral clip tagged neutral (0.88), loaded clip tagged neutral (0.95)
  fea-f02: neutral clip tagged sad (0.90), loaded clip tagged fearful (0.35)
  fea-f04: neutral clip tagged neutral (0.98), loaded clip tagged neutral (0.89)
  fea-f07: neutral clip tagged sad (1.00), loaded clip tagged sad (0.96)
  fea-f13: neutral clip tagged neutral (0.89), loaded clip tagged neutral (0.91)
  ang-f01: neutral clip tagged sad (0.85), loaded clip tagged angry (0.93)
  ang-f04: neutral clip tagged sad (0.99), loaded clip tagged angry (0.99)
  ang-f13: neutral clip tagged sad (0.55), loaded clip tagged angry (0.95)

relative tagger (tagger-relative.json), baseline = all of each speaker's baseline clips:
| emotion | pairs | both right | told apart | neutral clip tagged neutral | loaded clip tagged its emotion |
|---|---:|---:|---:|---:|---:|
| SAD | 12 | 7 | 8 | 11 | 8 |
| FEA | 13 | 8 | 9 | 11 | 8 |
| ANG | 13 | 8 | 9 | 9 | 11 |

pairs not told apart:
  sad-f03: neutral clip tagged neutral (0.85), loaded clip tagged neutral (0.70)
  sad-f05: neutral clip tagged neutral (0.96), loaded clip tagged neutral (0.41)
  sad-f08: neutral clip tagged sad (0.96), loaded clip tagged sad (0.61)
  sad-f10: neutral clip tagged neutral (0.97), loaded clip tagged neutral (0.93)
  fea-f01: neutral clip tagged neutral (0.91), loaded clip tagged neutral (0.76)
  fea-f04: neutral clip tagged sad (0.73), loaded clip tagged neutral (0.53)
  fea-f07: neutral clip tagged sad (1.00), loaded clip tagged sad (0.98)
  fea-f13: neutral clip tagged neutral (0.99), loaded clip tagged neutral (0.87)
  ang-f01: neutral clip tagged sad (0.74), loaded clip tagged angry (0.99)
  ang-f02: neutral clip tagged sad (0.61), loaded clip tagged fearful (0.96)
  ang-f11: neutral clip tagged angry (0.62), loaded clip tagged angry (1.00)
  ang-f13: neutral clip tagged sad (0.68), loaded clip tagged angry (0.94)

| criterion | absolute | relative | bar | |
|---|---:|---:|---|---|
| S1 told apart, paired | 26 of 38 | 26 of 38 | more, and one-sided sign test p < 0.05 | FAIL |
| S2 loaded clips tagged non-neutral | 33 of 38 | 32 of 38 | at least 31 | PASS |
  fixed 2: ang-f04, fea-f02
  broken 2: ang-f02, ang-f11
  exact one-sided sign test on the 4 pairs that differ: p = 0.688
  told apart per emotion (absolute -> relative): SAD 8 -> 8 of 12, FEA 8 -> 9 of 13, ANG 10 -> 9 of 13

PR-007: NOT SUPPORTED: failed S1

| pair | absolute: calm -> upset | relative: calm -> upset | absolute | relative | baseline clips |
|---|---|---|---|---|---:|
| sad-f01 | neutral (0.99) -> sad (0.98) | neutral (0.95) -> sad (1.00) | apart | apart | 9 |
| sad-f02 | neutral (0.89) -> sad (1.00) | neutral (0.99) -> sad (1.00) | apart | apart | 11 |
| sad-f03 | neutral (0.94) -> neutral (0.79) | neutral (0.85) -> neutral (0.70) | **not** | **not** | 9 |
| sad-f04 | neutral (0.42) -> sad (0.90) | neutral (0.73) -> sad (0.97) | apart | apart | 10 |
| sad-f05 | sad (0.55) -> sad (0.87) | neutral (0.96) -> neutral (0.41) | **not** | **not** | 6 |
| sad-f06 | neutral (1.00) -> sad (0.65) | neutral (1.00) -> fearful (0.44) | apart | apart | 10 |
| sad-f07 | neutral (0.99) -> sad (0.70) | neutral (0.83) -> sad (0.97) | apart | apart | 11 |
| sad-f08 | sad (0.89) -> sad (0.83) | sad (0.96) -> sad (0.61) | **not** | **not** | 10 |
| sad-f09 | neutral (0.89) -> fearful (0.73) | neutral (0.96) -> sad (0.56) | apart | apart | 9 |
| sad-f10 | neutral (0.98) -> neutral (0.98) | neutral (0.97) -> neutral (0.93) | **not** | **not** | 10 |
| sad-f11 | neutral (0.97) -> sad (0.98) | neutral (0.91) -> sad (1.00) | apart | apart | 11 |
| sad-f12 | neutral (0.96) -> sad (0.85) | neutral (0.95) -> sad (0.94) | apart | apart | 9 |
| fea-f01 | neutral (0.88) -> neutral (0.95) | neutral (0.91) -> neutral (0.76) | **not** | **not** | 11 |
| fea-f02 | sad (0.90) -> fearful (0.35) | neutral (0.86) -> fearful (0.97) | **not** | apart | 9 |
| fea-f03 | neutral (0.54) -> fearful (0.81) | neutral (0.54) -> fearful (1.00) | apart | apart | 9 |
| fea-f04 | neutral (0.98) -> neutral (0.89) | sad (0.73) -> neutral (0.53) | **not** | **not** | 10 |
| fea-f05 | neutral (0.98) -> fearful (0.97) | neutral (0.92) -> fearful (0.98) | apart | apart | 9 |
| fea-f06 | neutral (1.00) -> angry (0.93) | neutral (0.99) -> fearful (0.94) | apart | apart | 10 |
| fea-f07 | sad (1.00) -> sad (0.96) | sad (1.00) -> sad (0.98) | **not** | **not** | 10 |
| fea-f08 | neutral (0.83) -> fearful (0.92) | neutral (0.55) -> fearful (0.91) | apart | apart | 10 |
| fea-f09 | neutral (0.99) -> angry (0.97) | neutral (0.99) -> angry (0.60) | apart | apart | 11 |
| fea-f10 | neutral (0.96) -> fearful (0.74) | neutral (0.80) -> fearful (1.00) | apart | apart | 11 |
| fea-f11 | neutral (0.75) -> fearful (0.68) | neutral (0.90) -> fearful (1.00) | apart | apart | 6 |
| fea-f12 | neutral (0.95) -> sad (0.76) | neutral (0.95) -> fearful (0.98) | apart | apart | 7 |
| fea-f13 | neutral (0.89) -> neutral (0.91) | neutral (0.99) -> neutral (0.87) | **not** | **not** | 10 |
| ang-f01 | sad (0.85) -> angry (0.93) | sad (0.74) -> angry (0.99) | **not** | **not** | 10 |
| ang-f02 | neutral (0.99) -> fearful (0.58) | sad (0.61) -> fearful (0.96) | apart | **not** | 9 |
| ang-f03 | neutral (0.83) -> angry (1.00) | neutral (1.00) -> angry (1.00) | apart | apart | 10 |
| ang-f04 | sad (0.99) -> angry (0.99) | neutral (0.70) -> angry (1.00) | **not** | apart | 6 |
| ang-f05 | neutral (0.99) -> angry (0.92) | neutral (1.00) -> angry (1.00) | apart | apart | 11 |
| ang-f06 | neutral (0.82) -> angry (1.00) | neutral (0.95) -> angry (1.00) | apart | apart | 10 |
| ang-f07 | neutral (0.68) -> angry (0.87) | neutral (0.86) -> angry (0.99) | apart | apart | 10 |
| ang-f08 | neutral (0.94) -> angry (0.99) | neutral (0.78) -> angry (1.00) | apart | apart | 9 |
| ang-f09 | neutral (0.69) -> angry (0.99) | neutral (0.96) -> angry (0.99) | apart | apart | 11 |
| ang-f10 | neutral (0.77) -> angry (1.00) | neutral (0.80) -> angry (1.00) | apart | apart | 11 |
| ang-f11 | neutral (0.85) -> angry (0.95) | angry (0.62) -> angry (1.00) | apart | **not** | 10 |
| ang-f12 | neutral (0.96) -> fearful (0.77) | neutral (0.91) -> fearful (0.89) | apart | apart | 10 |
| ang-f13 | sad (0.55) -> angry (0.95) | sad (0.68) -> angry (0.94) | **not** | **not** | 11 |

saved to runs/prosody/pr007.json
