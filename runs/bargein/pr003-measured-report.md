20 items from probes/bargein · confirms PR-003 · recognizer time: as measured on this machine

| policy | stop-long | stop-short | continue-short | continue-long | wrong | missed | false barge-in | of which noise | stop latency of the hits, ms | agent words said over the caller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0_duration_1500ms | 0 | 5 | 0 | 3 | **8 of 20** | 5 | 3 | 0 of 2 | 1462 / 1575 / 4962  (fastest / middle / slowest of 5 hits) | 4 / 4 / 15 |
| B1_words | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 1102 / 1602 / 3402  (fastest / middle / slowest of 10 hits) | 3 / 5 / 12 |
| B1_words_min1 | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 642 / 1322 / 3402  (fastest / middle / slowest of 10 hits) | 2 / 4 / 12 |
| B2_speechlike | 0 | 0 | 0 | 1 | **1 of 20** | 0 | 1 | 0 of 2 | 1102 / 1662 / 3402  (fastest / middle / slowest of 10 hits) | 3 / 5 / 12 |

B0_duration_1500ms: ss01 (missed interruption), ss02 (missed interruption), ss03 (missed interruption), ss04 (missed interruption), ss05 (missed interruption), cl01 (false barge-in at +1542 ms), cl03 (false barge-in at +1714 ms), cl05 (false barge-in at +1702 ms)

B1_words: cs02 (false barge-in at +1880 ms), cs04 (false barge-in at +780 ms), cl02 (false barge-in at +2462 ms)

B1_words_min1: cs02 (false barge-in at +1880 ms), cs04 (false barge-in at +780 ms), cl02 (false barge-in at +2462 ms)

B2_speechlike: cl02 (false barge-in at +2462 ms)

| item | cell | should | B0_duration_1500ms | B1_words | B1_words_min1 | B2_speechlike | the recognizer finally wrote |
|---|---|---|---:|---:|---:|---:|---|
| sl01 | stop-long | stop | +4962 ms | +1662 ms | +1662 ms | +1662 ms | '' (no-speech 1.00) |
| sl02 | stop-long | stop | +1462 ms | +1482 ms | +1482 ms | +2002 ms | 'Sorry, can you slow down? I need to find the cable first.' (no-speech 0.03) |
| sl03 | stop-long | stop | +1575 ms | +1255 ms | +1255 ms | +1255 ms | '' (no-speech 1.00) |
| sl04 | stop-long | stop | +1542 ms | +1102 ms | +642 ms | +1102 ms | 'Can you book that one?' (no-speech 0.02) |
| sl05 | stop-long | stop | +1642 ms | +3402 ms | +3402 ms | +3402 ms | 'Let me think, I removed recently so it might be the old one.' (no-speech 0.01) |
| ss01 | stop-short | stop | **held, wrong** | +1442 ms | +1142 ms | +1442 ms | 'Hand-bond.' (no-speech 0.02) |
| ss02 | stop-short | stop | **held, wrong** | +1902 ms | +1202 ms | +1902 ms | '' (no-speech 1.00) |
| ss03 | stop-short | stop | **held, wrong** | +1662 ms | +1662 ms | +1662 ms | 'No, wait!' (no-speech 0.03) |
| ss04 | stop-short | stop | **held, wrong** | +1322 ms | +1322 ms | +1322 ms | '' (no-speech 1.00) |
| ss05 | stop-short | stop | **held, wrong** | +1602 ms | +1242 ms | +1602 ms | '1 second.' (no-speech 0.22) |
| cs01 | continue-short | keep going | held | held | held | held | 'Mm-hmm.' (no-speech 0.26) |
| cs02 | continue-short | keep going | held | **+1880 ms, wrong** | **+1880 ms, wrong** | held | 'Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. Okay. ' (no-speech 0.62) |
| cs03 | continue-short | keep going | held | held | held | held | 'Right.' (no-speech 0.11) |
| cs04 | continue-short | keep going | held | **+780 ms, wrong** | **+780 ms, wrong** | held | '' (no-speech 1.00) |
| cs05 | continue-short | keep going | held | held | held | held | 'Okay' (no-speech 0.04) |
| cl01 | continue-long | keep going | **+1542 ms, wrong** | held | held | held | 'Okay, yeah, that works for me' (no-speech 0.01) |
| cl02 | continue-long | keep going | held | **+2462 ms, wrong** | **+2462 ms, wrong** | **+2462 ms, wrong** | 'Doing that now' (no-speech 0.06) |
| cl03 | continue-long | keep going | **+1714 ms, wrong** | held | held | held | 'Okay, yeah, that is fine. Got it.' (no-speech 0.01) |
| cl04 | continue-long | keep going | held | held | held | held | 'Okay, go ahead' (no-speech 0.03) |
| cl05 | continue-long | keep going | **+1702 ms, wrong** | held | held | held | 'Sure, yeah, no problem at all.' (no-speech 0.01) |

B0_duration_1500ms: 1 of 10 interruptions stopped within 1500 ms
B1_words: 5 of 10 interruptions stopped within 1500 ms
B1_words_min1: 7 of 10 interruptions stopped within 1500 ms
B2_speechlike: 4 of 10 interruptions stopped within 1500 ms

B2_speechlike against B0_duration_1500ms: wrong on 1 against 8 of 20
  fixed 8: cl01, cl03, cl05, ss01, ss02, ss03, ss04, ss05
  broken 1: cl02
  exact two-sided sign test on the 9 items that differ: p = 0.039

saved to runs/bargein/pr003-measured.json
