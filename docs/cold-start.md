# Cold starts on Modal: what a wake takes and what it costs

Both plans asked for "cold-start economics" (v1 §4, P3, weeks 7 and 8; v2 §8, week 8). This is it, written after
M40 from runs already made: **no new run, \$0**. Every measured number names the log line it comes from. Numbers
worked out from Modal's list rates are marked **arithmetic**; none of them was measured.

The demo (`modal_app/demo.py`) is the workload: a CPU container for the page, and four GPU machines, each woken
only when a visitor's recording asks for its rungs, each asleep again after an idle window.

| Machine | Rungs | Rate (checked 2026-08-19, `modal_app/demo.py` lines 58 and 62) | Idle window |
|---|---|---:|---:|
| T4 | R0, R1, R2, R2 verbatim | \$0.59 an hour | 300 s |
| A10 | R3 | \$1.10 | 120 s |
| L40S | R4, R4 streamed | \$1.95 | 120 s |
| a second A10 | R5, Moshi | \$1.10 | 120 s |

## 1. What a wake is, and how it is measured

A wake is everything between a request reaching a sleeping machine and that machine starting to answer: the
container starting, the models loading from weights baked into the image (fetched once, on a CPU, when the image
was built; `HF_HUB_OFFLINE=1` at run time), and whatever each rung compiles or warms up at load.

The demo measures it on every cold request as **the request's wall time less the turns' own time**
(`src/s2slab/demoweb.py`, lines 150 to 158), and shows the last one on the page. A request is cold when it is the
first a container has answered (`src/s2slab/demo.py`, `Live.turns`).

## 2. Each machine's wake, over time

**T4** (four rungs):

| When | Wake | What it was | Source |
|---|---:|---|---|
| 09-28, the first run on Modal | 72 s | 62.8 s of it the four rungs loading | `docs/m27-log.md`, line 292 |
| 09-28, later sessions | 47 s, 59 s, 54 s, 49 s, 52 s | loads of 40.8 to 50.1 s | `docs/m27-log.md`, lines 486, 586, 587, 750, 848 |
| 09-29 | 49.6 s | 41.8 s loading, the request 55.2 s | `docs/m27-log.md`, line 1175 |
| 10-18 (M39) | the cold Send 48.8 s | the four loaded in 37.7 s, R1 in 4.8 | `docs/m39-log.md`, row 3 |

About 50 s, nearly all of it loading the rungs, and steady over three weeks.

**A10, R3** (Qwen2-Audio, 17 GB): 37 s, 33 s, then 31.8 s with 24.4 s of loading (`docs/m27-log.md`, lines 750,
852, 1176).

**Second A10, Moshi** (16.6 GB): 82 s the first time (`docs/m27-log.md`, line 854), then 50.5 s, after a throwaway
call at load was added (line 1178). No log says why it fell.

**L40S, R4 and later R4 streamed beside it** (one set of 22.5 GB weights): the wake that changed most, and the one
four milestones worked on.

| When | Wake or load | What changed | Source |
|---|---:|---|---|
| 09-28 | wake 51 s, 50 s | R4 only, eager | `docs/m27-log.md`, lines 750, 853 |
| 09-29 | wake 66.5 s, 27.9 s of it loading | most of it before the model began to load | `docs/m27-log.md`, lines 1177, 1181 |
| 10-05 (M32a) | wake 159 s, 152.7 s of it loading | R4 streamed added, its Talker compiled at load | `docs/m32-log.md`, line 93 |
| 10-13 (M34) | load 251.0 s; wake 260.8 s on the deployed page | the sized cache: both caches written and both steps captured at load | `docs/m34-log.md`, lines 17, 18, 118 |
| 10-14 (M35) | load 185.7 s; wake 194.1 s deployed | only the 2048-place cache prepared at load | `docs/m35-log.md`, rows 3 and 4 |
| 10-15 (M36) | load 93.1 s and 71.8 s | the compile cache on a Modal volume, loaded after R4 | `docs/m36-log.md`, row 5 |
| 10-17, 10-18 (M37) | load 73.5 s, 92.3 s | the compile cache hit | `docs/m37-log.md`, rows 7 and 13 |

**The L40S's load, phase by phase** (`docs/m35-log.md`, section 2; `docs/m36-log.md`, section 3):

| Phase | M35, no cache | M36, cache, first wake | M36, second wake |
|---|---:|---:|---:|
| R4: its weights and its throwaway turn, which compiles the Talker's layers | 61.2 s | 59.7 | 45.5 |
| R4 streamed: the DiT compiled and captured for its three window lengths | 65.2 | 12.4 | 10.4 |
| R4 streamed: the 2048-place cache's decoder compiled, its step captured | 56.6 | 16.6 | 13.4 |
| R4 streamed: its throwaway turn | 2.7 | 2.6 | 2.3 |
| **in all** | **185.7** | **93.1** | **71.8** |

The weights themselves take about 4 s to load (`docs/m35-log.md`, section 2). **On the L40S a wake is mostly
compiling**, and a compile cache on a volume turned most of that into a read: R4 streamed's compiles went from
about 120 s to about 25 (`docs/m36-log.md`, section 3). What is left is R4's own Talker compile (about 30 s),
which the cache does not hold: loading the cache before R4 once made R4's load 288 s instead of 57, unexplained
(`docs/m36-log.md`, section 2).

## 3. What a wake costs

**Arithmetic**, from the list rates: a machine woken for one turn is billed for its wake, its turn, and its idle
window after the turn. The wake is the last measured; the turn's own seconds are left out (a few seconds on the
cascades, more for R4's long replies and for a Moshi call).

| Machine | Wake | Its cost | Idle window | Its cost | A cold turn, about |
|---|---:|---:|---:|---:|---:|
| T4 | 49.6 s | \$0.008 | 300 s | \$0.049 | **\$0.06** |
| A10, R3 | 31.8 s | \$0.010 | 120 s | \$0.037 | **\$0.05** |
| L40S, after M36 | 72 to 93 s | \$0.04 to 0.05 | 120 s | \$0.065 | **\$0.10 to 0.11** |
| L40S, in M34 | 260.8 s | \$0.14 | 120 s | \$0.065 | \$0.21 |
| second A10, Moshi | 50.5 s | \$0.015 | 120 s | \$0.037 | **\$0.05** |

**The idle window, not the wake, is most of what a cold turn costs**, on every machine since M36. M34's own
arithmetic (`docs/m34-log.md`, line 121): "a cold wake costs about \$0.14 of the day's cap instead of \$0.09".

**What the app counted** (`demo.Budget`: the request's wall time, which holds the wake, plus the window): \$0.0223 for
the first T4 turn on Modal, with a 60 s window then (`docs/m27-log.md`, line 306; Modal's dashboard showed \$0.03 for
the app, CPU and memory included: `docs/cost-ledger.md`, line 62); \$0.0182, \$0.0200 and \$0.0193 for later T4 wakes
(`docs/m27-log.md`, lines 490 and 589); about \$0.27 for a turn that woke all four machines (line 1245). With all four
up, their windows ran at \$4.74 an hour together (line 1247).

## 4. Cold against warm

**Arithmetic.** Keeping a machine up all day (`min_containers=1`, never used here) against waking it per visit:

| Machine | Warm all day | A cold turn | Cold turns a day before warm is cheaper |
|---|---:|---:|---:|
| T4 | \$14.16 | \$0.06 | about 250 |
| A10, R3 | \$26.40 (v1 §5.1, rule 4: "\$26.40/day") | \$0.05 | about 570 |
| L40S | \$46.80 | \$0.10 to 0.11 | about 400 to 450 |
| second A10, Moshi | \$26.40 | \$0.05 | about 510 |
| all four | \$113.76 | | |

The demo's own cap is \$3 a day (`modal_app/demo.py`, line 59): about 50 cold T4 turns, or the T4 kept warm for about
five hours. **For a portfolio demo that a few people open, scale-to-zero is cheaper by two orders of magnitude**,
and the price is the visitor's wait: about 50 s for the cascades and about a minute and a half for R4 since M36
(`docs/m36-log.md`, section 3), four and a half minutes in M34 (`docs/m34-log.md`, line 120).

## 5. The idle window

A window too short makes every turn a wake: at the batch runs' 60 s, both turns of a session woke the T4, a visitor
listening to four replies and recording again in more than a minute (`docs/m27-log.md`, entry 17, line 591). So the
T4 waits 300 s, the author's choice; the A10s and the L40S 120 s, since their idle time costs two and three times the
T4's (`modal_app/demo.py`, lines 60 to 63). **Arithmetic** (`docs/m27-log.md`, lines 646 to 652): a T4 window of 60 s
costs \$0.01 and lets about 50 bursts into a dollar; 300 s costs \$0.05 and lets about 17. The batch runs keep
`scaledown_window=60` (v2 §5.1, rule 3; every batch and training function in `modal_app/`): they are one call each, with nothing after it.

## 6. What is not measured

- **The CPU container's cost.** No CPU rate is recorded anywhere; `cpu=1.0` for the page's container was judged "a
  few cents a day while the page is used, outside the day's cap" (`docs/m32-log.md`, lines 112 and 113).
- **Image builds**, billed apart at CPU rates (`docs/cost-ledger.md`, "Not in the rows").
- **The estimates against the bill.** Checked once: \$0.03 on the dashboard for the first T4 session, against the
  app's \$0.0223 for the GPU (`docs/cost-ledger.md`, line 62). Every demo row since M34 is an estimate from the
  rates, the wakes and the windows.
- **A warm container.** The table in section 4 is arithmetic; no machine here was ever kept up.
- **Modal's 150 s limit on a web request.** Moshi's wake and call once came to 124 s of one request
  (`docs/m27-log.md`, line 871), and one request per machine fixed it. Later cold streams from the L40S ran 164 to
  357 s (`docs/m32-log.md`, line 95; `docs/cost-ledger.md`, rows of M32 to M36) without failing; no log says whether
  the limit applies to a streamed response, or why it did not stop them.
- **Hosts.** Two cold wakes on one card can differ by tens of seconds (`docs/m36-log.md`, section 2: "hosts differ").
  Each wake above is one sample.
