# M34 run log: R4 streamed on a right-sized cache

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m33-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-13 | — | the design: a new rung, R4 streamed with our captured loop on a cache sized to the turn (2048 places when the prompt leaves 1,024 codes of room, else 6144); today's stream kept as the baseline; the start rule written before the development run; PR-016 registered before any eval run; the new rung to take "R4 streamed" in the demo if PR-016 is supported; the author's go on both choices | — | — |
| 2 | 10-13 | the new rung's tests, laptop, the tiny Talker and the fake DiT | a turn takes the smallest cache that leaves it room, and writes generate's codes; a cache too small cuts the reply and says so; the sized stream writes today's stream's codes and the same audio, on the 2048-place cache; each cache written and its step captured at load | `r4sized.py` (`SizedTalker`, `r4-gpu-stream-sized`), `FastOmni.make_cache`, `r4sizeddev.py`, `::sizeddev`, tests | deb2294 |
| 3 | 10-13 | `modal run modal_app/r4_bench.py::sizeddev`, the author, L40S (driver 610.57.04), 30 probe clips, \$0.505 | both streams' codes `r4-gpu-fast`'s on 30 of 30, the same texts and words (2 errors in 634 each); the sized stream on 2048 places for 29 turns, 6144 for the 53 s clip, no reply cut; at a 0.5 s buffer the sized stream's first sound 2,684 ms (p50), 3,114 (p90), no stall; the stream's 3,433 and 3,816, 4 of 30 stalled; total 5,820 against 7,734 | — | — |
| 4 | 10-13 | — | the start rule applied: 0.5 s for the sized stream (0 and 0.25 s stalled every turn; 1 s the same as 0.5); none for the stream on this host, which PR-016 compares as frozen | — | — |
| 5 | 10-13 | — | PR-016 registered before any eval-set turn of the sized stream: first sound p50 ≤ 0.90 and p90 ≤ 0.92 of the stream's, total ≤ 1.00, PR-015's stall tolerance, words within a point | `pr016.py`, `::pr016`, `r4sizeddev.run` takes a frozen start buffer, tests | this commit |

## 1. The design

**What M33 found.** Half of R4's Talker step was attention over its 6144-place cache's empty places: our
captured loop wrote a code in 10.26 ms there and 6.62 ms on a 2048-place cache, the same codes.

**The rung.** `r4-gpu-stream-sized` is R4 streamed as frozen for PR-015 (the same blocks, DiT and start
buffer, until the development run chooses one), with its codes from a `SizedTalker`: our loop with its
step captured as a CUDA graph, on the smallest cache that leaves the turn 1,024 codes of room after its
prompt (about 20 s of speech, twice the longest reply on the probe clips): 2048 places, else today's
6144. One loop and one captured step a cache, both made at load. A reply that reaches its cache's end is
cut, and counted in every run. `r4-gpu-stream` stays as it is, the baseline.

**Steps.** 1, the rung on the laptop ($0). 2, the development run: `r4-gpu-fast`, `r4-gpu-stream` and
`r4-gpu-stream-sized` on the 30 probe clips, both streams with no start buffer and every buffer of 0, 0.25,
0.5 and 1 s computed from the same turns, as M31 did; each stream's codes against the fast rung's, Whisper
on every reply (about \$0.70). 3, PR-016, registered before any eval run: the sized stream at its chosen
buffer against `r4-gpu-stream` as frozen; first sound sooner at p50, with no more stalled turns and no
worse words (about \$2.60). 4, the demo: the new rung under the "R4 streamed" label, if PR-016 is
supported; a `modal serve` check and `modal deploy`, the author's.

**The start rule, written before the development run:** for each stream, the buffer with the lowest
first-sound p50 among those with no stalled turn on the 30 probe clips; a tie goes to the smaller
buffer. (M31 chose PR-015's 0.5 s by the same rule.)

**What it could buy** (M33's projection, not measured): first sound at about 1.9 s with a perfect start,
or 2.4 s under today's 0.5 s rule, against 3.08 s today on the probe clips.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 1001 passed, 2 skipped.

## 2. The development run (rows 3 and 4)

`runs/r4-gpu/sized-dev-l40s.json`, at `deb2294`, one L40S container on the slower of M31's two hosts.

| rung | first sound p50 | p90 | stalled turns | total p50 | WER | codes as the fast rung's |
|---|---:|---:|---:|---:|---:|---:|
| `r4-gpu-fast` | 6,376 ms | 11,921 | 0 of 30 | 6,166 | 0.5% | - |
| `r4-gpu-stream`, no buffer | 1,722 | 2,123 | 30 of 30 | 7,734 | 0.3% | 30 of 30 |
| `r4-gpu-stream-sized`, no buffer | 1,527 | 1,946 | 30 of 30 | 5,820 | 0.3% | 30 of 30 |

With a start buffer, computed from the same turns:

| buffer | the stream: first sound p50 / p90, stalled | the sized stream |
|---:|---|---|
| 0 and 0.25 s | 1,722 / 2,123, 30 of 30 | 1,527 / 1,946, 30 of 30 |
| 0.5 and 1 s | 3,433 / 3,816, 4 of 30 | **2,684 / 3,114, 0 of 30** |

(The first block is 24 codes, 0.48 s: any buffer under it plays the first block alone, any from it to
2.4 s waits for the second.)

**The start rule, applied** (written before the run): the sized stream, 0.5 s; the stream, none, since
every buffer stalled a turn on this host, where PR-015's host stalled none at 0.5 s. PR-016 compares the
sized stream at 0.5 s with the stream as frozen.

On development the sized stream at 0.5 s against the stream at 0.5 s: first sound p50 0.782 (95% interval
0.771 to 0.794), p90 0.816 (0.737 to 0.868), total p50 0.753 (0.720 to 0.804); the sized stream's smallest
margin before a stall 771 ms. M33 projected about 2.4 s under this rule on a faster host; here 2.68.

## 3. PR-016, registered (row 5)

`docs/preregistration.md`, PR-016: on the frozen eval set, the sized stream against the stream as frozen,
both at 0.5 s, `r4-gpu-fast` beside them. Valid only if the texts are the same on 190 of 200, the sized
stream's codes the fast rung's on 190, and at most 2 replies cut. Supported if all five hold: first sound
p50 ≤ 0.90 and p90 ≤ 0.92 of the stream's (upper ends of paired bootstrap intervals), total p50 ≤ 1.00, at
most 20 turns stalled and 10 for more than 200 ms, Whisper's word error rate within a point. Then the run,
once: `modal run modal_app/r4_bench.py::pr016 --confirm PR-016` (about \$2.60).

