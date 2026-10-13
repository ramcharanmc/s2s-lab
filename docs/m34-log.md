# M34 run log: R4 streamed on a right-sized cache

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m33-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-13 | — | the design: a new rung, R4 streamed with our captured loop on a cache sized to the turn (2048 places when the prompt leaves 1,024 codes of room, else 6144); today's stream kept as the baseline; the start rule written before the development run; PR-016 registered before any eval run; the new rung to take "R4 streamed" in the demo if PR-016 is supported; the author's go on both choices | — | — |
| 2 | 10-13 | the new rung's tests, laptop, the tiny Talker and the fake DiT | a turn takes the smallest cache that leaves it room, and writes generate's codes; a cache too small cuts the reply and says so; the sized stream writes today's stream's codes and the same audio, on the 2048-place cache; each cache written and its step captured at load | `r4sized.py` (`SizedTalker`, `r4-gpu-stream-sized`), `FastOmni.make_cache`, `r4sizeddev.py`, `::sizeddev`, tests | this commit |

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
