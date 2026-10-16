# M37 run log: a start rule timed from the turn

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m36-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-16 | — | the design (below); the author's go | — | — |
| 2 | 10-16 | the tests, laptop: 1028 passed, 2 skipped (13 new) | a start checked at each code would have gained nothing: the planned start falls inside the second window's decode (about 520 ms, longer than the first block's 460), when the Talker's thread writes no code. So a timer hands the held pieces on, from a thread of its own; the simulation scores it as such, and a test shows the two agree | `r4stream` (the codes', pieces' and windows' times; `TimedStart`, `planned_start`), `r4streamdev.heard`, `r4timed.py`, `r4timeddev.py`, `::timeddev`, tests | fe65b94 |
| 3 | 10-16 | `modal run modal_app/r4_bench.py::timeddev`, the author, at fe65b94 | one L40S (driver 610.57.04, the slower host), 407 s in the function, \$0.224; 29 turns on the 2,048-place cache, 1 on 6,144, none cut. The plan with no margin short by 12 ms (p50), 17 at most, and every turn stalled; **the margin 50 ms**, the smallest with no stalled turn: first sound 2,248 ms against the buffer's 2,670 (p50), 423 ms sooner. **GO** by the rule | — | — |
| 4 | 10-16 | — | the margin frozen; the rung `r4-gpu-stream-timed` registered; **PR-017 registered** before any eval-set turn of it; PR-016's row, missing from the registry's table since M34, added | `r4timed.MARGIN_S`, `rungs`, `r4stream` (the plan's time kept), `pr017.py`, `::pr017`, `docs/preregistration.md`, tests | this commit |

## 1. The design

**Why.** On M34's development turns (`runs/r4-gpu/sized-dev-l40s.json`, no start buffer) the second block came
a nearly fixed time after the first: 1,153 to 1,169 ms on the 29 turns on the 2,048-place cache, 1,528 on the
one on 6,144 (n04). That is 96 codes of the Talker (6.6 or 10.5 ms each, as M33 measured) and one decode of
the second block's window (about 520 ms). Later blocks are made faster than they play (1.92 s of audio in 1.2
to 1.5 s), so only the second can run playback dry. The 0.5 s start buffer always waits for it (the first
block is 0.46 s): playback starts about 1.16 s after the first block, where about 0.70 would do (p50 697 ms,
p90 703, max 1,068 on n04). About 0.46 s lost on every turn; on the eval set R4 streamed is heard after
2.62 s (PR-016, p50).

**The rule.** When the first block exists, at t1, playback starts at

    t1 + max(0, 96 × (this turn's ms a code) + (the second window's decode) − (the first block's length) + m)

The ms a code is the turn's own, over the codes written so far (36): it carries the cache the turn took and
the host's speed. The second window's decode (132 codes) is timed at load, on a replay of its captured graph,
in each container. A fixed wait could not do this: on those 30 turns it would need about 1,070 ms, for n04,
barely better than today. The held pieces are handed on by a timer of their own (row 2); a reply that ends
first plays at its end, as today.

**The development run.** `modal run modal_app/r4_bench.py::timeddev`, the author's: one L40S, the 30 probe
clips, `r4-gpu-stream-sized` as deployed with no hold, every turn's times kept (each code's, each piece's as
decoded, each window's decode) and the load's window decodes. Every rule is scored afterwards from those
times on the same turns (`r4timed.simulate`; today's 0.5 s buffer by `r4streamdev.buffered`): a start rule
changes when audio is handed out, never what is made. Expected about \$0.25 to \$0.35.

**The rules, written before the run** (`s2slab.r4timeddev`):

- **The margin:** m is the smallest of 0, 50, …, 500 ms with no stalled turn on the 30 probe clips (any
  stall over 0 ms, as in M34).
- **The stop:** no margin stall-free, or the chosen margin's first-sound p50 less than **150 ms** sooner than
  today's 0.5 s buffer on the same turns: M37 stops there, nothing registered, as M32b did.
- **Alongside, deciding nothing:** a fixed wait after the first block, the smallest of 0, 50, …, 2,000 ms
  with no stalled turn: what timing from the turn adds.

**Then**, if the rules say go: the margin frozen in a new rung, `r4-gpu-stream-timed`, and **PR-017**
registered before any eval run. A sketch, its thresholds fixed after the development run: on the 200 eval
clips, the timed rung's first-sound p50 at most about 0.90 of today's rule's, the 95% interval's upper bound
under 1; its p90 no worse; at most 2 of 200 turns stalled; the same codes and words. Paired and half PR-016's
cost (about \$1.30): one live pass of the timed rung, today's rule computed from the same turns' pieces.
Then the demo, only if PR-017 is supported.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 1028 passed, 2 skipped.

## 2. The development run (row 3)

`runs/r4-gpu/timed-dev-l40s.json`: the sized stream with no hold on the 30 probe clips, every rule scored on the
same turns.

| start rule | first sound p50 ms | p90 | stalled turns |
|---|---:|---:|---:|
| today: the 0.5 s buffer | 2,670 | 3,092 | 0 |
| timed, margin 0 ms | 2,198 | 2,618 | 30 |
| **timed, margin 50 ms** (the rule's choice) | **2,248** | **2,668** | **0** |
| timed, margin 100 ms | 2,298 | 2,718 | 0 |
| a fixed wait after the first block, 1,100 ms (the smallest stall-free) | 2,612 | 3,032 | 0 |

**What the plan rests on.** The second window (132 codes) decoded in 513.5 ms at load and 526 (p50), 532 at most,
in the turns: about 12 ms slower, so the plan with no margin started every turn a little early and every turn
stalled, by 15 ms at p90. The turn's own ms a code was 6.6 (p50) and 10.5 on the 6,144-place turn, which the plan
followed: that turn needed 1,071 ms after its first block, the others about 698. With 50 ms every turn kept at
least 33 ms in hand. Paired ratios against the buffer: 0.842 (95% interval 0.834 to 0.850) at p50, 0.863 (0.854
to 0.932) at p90. **Timing from the turn is the gain**: a fixed wait had to cover the slow turn and gained 58 ms.

The first window (36 codes) took 251 ms at load; the plan does not use it, counting from when the first block
exists.

**Then:** PR-017 registered (row 4), its run the author's: `modal run modal_app/r4_bench.py::pr017 --confirm
PR-017`, one L40S, about \$0.80.
