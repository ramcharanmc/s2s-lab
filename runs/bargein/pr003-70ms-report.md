20 items from probes/bargein · confirms PR-003 · recognizer time: charged at 70 ms a call

| policy | stop-long | stop-short | continue-short | continue-long | wrong | missed | false barge-in | of which noise | stop latency of the hits, ms | agent words said over the caller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0_duration_200ms | 0 | 0 | 5 | 5 | **10 of 20** | 0 | 10 | 2 of 2 | 62 / 202 / 275  (fastest / middle / slowest of 10 hits) | 0 / 0 / 1 |
| B0_duration_400ms | 0 | 0 | 5 | 5 | **10 of 20** | 0 | 10 | 2 of 2 | 262 / 402 / 475  (fastest / middle / slowest of 10 hits) | 0 / 1 / 2 |
| B0_duration_800ms | 0 | 4 | 2 | 5 | **11 of 20** | 4 | 7 | 2 of 2 | 762 / 842 / 875  (fastest / middle / slowest of 6 hits) | 2 / 2 / 5 |
| B0_duration_1500ms | 0 | 5 | 0 | 3 | **8 of 20** | 5 | 3 | 0 of 2 | 1462 / 1575 / 4962  (fastest / middle / slowest of 5 hits) | 4 / 4 / 15 |
| B1_words | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 642 / 1122 / 2662  (fastest / middle / slowest of 10 hits) | 1 / 4 / 10 |
| B1_words_min1 | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 282 / 782 / 2662  (fastest / middle / slowest of 10 hits) | 1 / 3 / 10 |
| B2_speechlike | 0 | 0 | 0 | 1 | **1 of 20** | 0 | 1 | 0 of 2 | 642 / 1122 / 3102  (fastest / middle / slowest of 10 hits) | 1 / 4 / 11 |

B0_duration_200ms: cs01 (false barge-in at +215 ms), cs02 (false barge-in at +340 ms), cs03 (false barge-in at +273 ms), cs04 (false barge-in at +240 ms), cs05 (false barge-in at +182 ms), cl01 (false barge-in at +202 ms), cl02 (false barge-in at +202 ms), cl03 (false barge-in at +254 ms), cl04 (false barge-in at +182 ms), cl05 (false barge-in at +182 ms)

B0_duration_400ms: cs01 (false barge-in at +415 ms), cs02 (false barge-in at +540 ms), cs03 (false barge-in at +513 ms), cs04 (false barge-in at +440 ms), cs05 (false barge-in at +382 ms), cl01 (false barge-in at +402 ms), cl02 (false barge-in at +402 ms), cl03 (false barge-in at +454 ms), cl04 (false barge-in at +382 ms), cl05 (false barge-in at +382 ms)

B0_duration_800ms: ss01 (missed interruption), ss03 (missed interruption), ss04 (missed interruption), ss05 (missed interruption), cs02 (false barge-in at +940 ms), cs04 (false barge-in at +840 ms), cl01 (false barge-in at +802 ms), cl02 (false barge-in at +1802 ms), cl03 (false barge-in at +874 ms), cl04 (false barge-in at +782 ms), cl05 (false barge-in at +822 ms)

B0_duration_1500ms: ss01 (missed interruption), ss02 (missed interruption), ss03 (missed interruption), ss04 (missed interruption), ss05 (missed interruption), cl01 (false barge-in at +1542 ms), cl03 (false barge-in at +1714 ms), cl05 (false barge-in at +1702 ms)

B1_words: cs02 (false barge-in at +640 ms), cs04 (false barge-in at +320 ms), cl02 (false barge-in at +1882 ms)

B1_words_min1: cs02 (false barge-in at +640 ms), cs04 (false barge-in at +320 ms), cl02 (false barge-in at +1682 ms)

B2_speechlike: cl02 (false barge-in at +2102 ms)

| item | cell | should | B0_duration_200ms | B0_duration_400ms | B0_duration_800ms | B0_duration_1500ms | B1_words | B1_words_min1 | B2_speechlike | the recognizer finally wrote |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| sl01 | stop-long | stop | +202 ms | +402 ms | +802 ms | +4962 ms | +1122 ms | +1122 ms | +1122 ms | '' (no-speech 1.00) |
| sl02 | stop-long | stop | +162 ms | +362 ms | +762 ms | +1462 ms | +642 ms | +642 ms | +642 ms | 'Sorry, can you slow down? I need to find the cable first.' (no-speech 0.04) |
| sl03 | stop-long | stop | +275 ms | +475 ms | +875 ms | +1575 ms | +955 ms | +955 ms | +955 ms | '' (no-speech 1.00) |
| sl04 | stop-long | stop | +202 ms | +402 ms | +842 ms | +1542 ms | +1102 ms | +282 ms | +1102 ms | 'Can you book that one?' (no-speech 0.02) |
| sl05 | stop-long | stop | +202 ms | +402 ms | +842 ms | +1642 ms | +2662 ms | +2662 ms | +3102 ms | 'Let me think, I removed recently so it might be the old one.' (no-speech 0.01) |
| ss01 | stop-short | stop | +162 ms | +362 ms | **held, wrong** | **held, wrong** | +882 ms | +662 ms | +882 ms | 'Hand-bond.' (no-speech 0.02) |
| ss02 | stop-short | stop | +182 ms | +382 ms | +842 ms | **held, wrong** | +1482 ms | +662 ms | +1482 ms | '' (no-speech 1.00) |
| ss03 | stop-short | stop | +202 ms | +402 ms | **held, wrong** | **held, wrong** | +1142 ms | +1142 ms | +1142 ms | 'No, wait.' (no-speech 0.03) |
| ss04 | stop-short | stop | +62 ms | +262 ms | **held, wrong** | **held, wrong** | +782 ms | +782 ms | +782 ms | '' (no-speech 1.00) |
| ss05 | stop-short | stop | +182 ms | +382 ms | **held, wrong** | **held, wrong** | +1122 ms | +482 ms | +1122 ms | '1 second.' (no-speech 0.22) |
| cs01 | continue-short | keep going | **+215 ms, wrong** | **+415 ms, wrong** | held | held | held | held | held | 'Mm-hmm' (no-speech 0.30) |
| cs02 | continue-short | keep going | **+340 ms, wrong** | **+540 ms, wrong** | **+940 ms, wrong** | held | **+640 ms, wrong** | **+640 ms, wrong** | held | 'Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. ' (no-speech 0.62) |
| cs03 | continue-short | keep going | **+273 ms, wrong** | **+513 ms, wrong** | held | held | held | held | held | 'Right.' (no-speech 0.11) |
| cs04 | continue-short | keep going | **+240 ms, wrong** | **+440 ms, wrong** | **+840 ms, wrong** | held | **+320 ms, wrong** | **+320 ms, wrong** | held | '' (no-speech 1.00) |
| cs05 | continue-short | keep going | **+182 ms, wrong** | **+382 ms, wrong** | held | held | held | held | held | 'Okay' (no-speech 0.04) |
| cl01 | continue-long | keep going | **+202 ms, wrong** | **+402 ms, wrong** | **+802 ms, wrong** | **+1542 ms, wrong** | held | held | held | 'Okay, yeah, that works for me' (no-speech 0.01) |
| cl02 | continue-long | keep going | **+202 ms, wrong** | **+402 ms, wrong** | **+1802 ms, wrong** | held | **+1882 ms, wrong** | **+1682 ms, wrong** | **+2102 ms, wrong** | 'Doing that now' (no-speech 0.07) |
| cl03 | continue-long | keep going | **+254 ms, wrong** | **+454 ms, wrong** | **+874 ms, wrong** | **+1714 ms, wrong** | held | held | held | 'Okay, yeah, that is fine. Got it.' (no-speech 0.01) |
| cl04 | continue-long | keep going | **+182 ms, wrong** | **+382 ms, wrong** | **+782 ms, wrong** | held | held | held | held | 'Okay, go ahead' (no-speech 0.03) |
| cl05 | continue-long | keep going | **+182 ms, wrong** | **+382 ms, wrong** | **+822 ms, wrong** | **+1702 ms, wrong** | held | held | held | 'Sure, yeah, no problem at all.' (no-speech 0.01) |

B0_duration_200ms: 10 of 10 interruptions stopped within 1500 ms
B0_duration_400ms: 10 of 10 interruptions stopped within 1500 ms
B0_duration_800ms: 6 of 10 interruptions stopped within 1500 ms
B0_duration_1500ms: 1 of 10 interruptions stopped within 1500 ms
B1_words: 9 of 10 interruptions stopped within 1500 ms
B1_words_min1: 9 of 10 interruptions stopped within 1500 ms
B2_speechlike: 9 of 10 interruptions stopped within 1500 ms

B2_speechlike against B0_duration_1500ms: wrong on 1 against 8 of 20
  fixed 8: cl01, cl03, cl05, ss01, ss02, ss03, ss04, ss05
  broken 1: cl02
  exact two-sided sign test on the 9 items that differ: p = 0.039

saved to runs/bargein/pr003-70ms.json
