# M35 run log: a cheaper L40S wake

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m34-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-14 | — | the design: only the 2048-place cache prepared at the L40S's load, the 6144-place step captured on the first turn that needs it (none in the demo: its turns are 15 s at most, about 485 places); the load timed by phase; the author's go | — | — |
| 2 | 10-14 | the tests, laptop, the tiny Talker | a cache not prepared is written on its first turn (our loop, eager) and its step captured on the next, generate's codes both times; the rung prepares only 2048 at load and times its phases; the demo prints each rung's load and phases | `r4sized` (`PREPARED`, `prepare(sizes)`, a first turn on an unwritten cache eager), `r4stream` (`load_phases`), `demo.Live.load_report`, `modal_app/demo.py` prints it, tests | this commit |

## 1. The design

**Why.** M34 lengthened the L40S's wake from 153 to 167 s to 251 s: at load the new rung wrote both its
caches and captured both steps, four throwaway turns and a compile of the decoder for each cache size.
The demo cuts a recording at 15 s (`MAX_CLIP_S`), whose prompt takes about 485 places, so no demo turn
can take the 6144-place cache.

**The change.** `r4-gpu-stream-sized` prepares the caches of `PREPARED` (2048) at load; the 6144-place
cache's step is captured on the first turn that needs it, which pays about 40 s once (the 6144 cache is
the fast rung's own, written by R4's throwaway turn; a cache never written takes one eager turn first).
The load is timed by phase (the model, the throwaway turn, the DiT's windows, each cache prepared) and
the demo's container log prints every rung's load and its phases.

**Expected:** the wake about 40 to 45 s shorter, about 205 s. The bigger lever, a compile cache kept on
a Modal volume so later wakes skip most compiles, is left for after the phases have been read.

**Then** a `modal serve` session with one cold L40S turn (the author's, about \$0.30): the wake and its
phases from the log; then `modal deploy`, the author's.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 1009 passed, 2 skipped.
