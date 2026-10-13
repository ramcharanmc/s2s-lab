# M33 run log: where a Talker step's time goes

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m32-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-12 | — | the design: the Talker's size and floors, its captured step on caches of 6144, 2048 and 1024 places, one turn's kernels at 6144 and 2048; the follow-up line written before the run (2048 places at 8 ms a code or less, no turn cut); exploratory, the probe clips only; the author's go | — | — |
| 2 | 10-12 | the profile's tests, laptop, the tiny Talker of M32b | each cache size timed, the places each turn filled counted, a cache too small cutting the reply and saying so; the breakdown by kind of operation; a fault found in reading before any run: a new cache's tensors are made on its first write, so a capture on an unwritten cache would read the wrong ones | `r4stepprofile.py`, `::stepprofile`, tests; `talkerloop` refuses a capture on an unwritten cache, and the profile writes each new cache once (our loop, eager and capped) before any capture | a4502d2 |
| 3 | 10-13 | `modal run modal_app/r4_bench.py::stepprofile`, the author, L40S, 10 probe clips, \$0.309 | the Talker 1,351 M parameters, 2.64 GB read a code: 3.06 ms at best; captured, 10.26 ms a code at 6144 places, **6.62 at 2048**, codes the 6144 cache's on 10 of 10, no turn cut; 1024 places failed on the 53 s probe clip (a prompt of 1,422 places); eager, attention 5.10 ms a code at 6144 and 1.75 at 2048, the matrix products 4.15 at both | — | — |
| 4 | 10-13 | — | the follow-up line met (2048 places at 6.62 ms a code, under 8, no turn cut): a design note for a right-sized cache in R4 streamed follows. M33 done | — | this commit |

## 1. The design

**The question.** M32b found R4's Talker at about 10 ms of GPU work a code, even with the whole step
captured as one CUDA graph. Where does it go? Its static cache holds 6144 places (room for a 60 s turn
and 4096 codes) and every step attends over all of them, masked, while a turn fills far fewer.

**What the probe measures** (exploratory: no registration; the 10 probe clips of M31 and M32b; nothing of
the eval set):

1. the Talker's size from its configuration: layers, widths, parameters, the bytes a code reads, and the
   time an L40S needs to read them once (864 GB/s): the floor a step cannot go under; the cache's keys and
   values at each size, and theirs;
2. one clip's Talker turn, our loop run eagerly under the PyTorch profiler at 6144 and 2048 places: GPU
   time a code by kind of kernel (attention, matrix products, sampling, the rest), and the ten longest;
3. our loop with its step captured, on caches of 6144, 2048 and 1024 places, the 10 clips: ms a code,
   whether the codes are the 6144-place cache's (masked places add nothing in principle, but a kernel may
   sum in another order), and how many places each turn filled. A cache too small for a turn caps its
   reply at the room left, and the turn is counted as cut.

**Worth a follow-up, written before the run:** the 2048-place cache at **8 ms a code or less** (p50), no
turn cut. Then a design note for a right-sized cache in R4 streamed, and a registration: by M32b's
projections the stall-free start would move from about 2.7 to 3.1 s to about 2.0. If not, the latency
work on R4 ends here, the breakdown its reason.

**The limit a smaller cache brings**, for that note, not this run: 2048 places hold a prompt of about 25
places a second of audio and 110 of text, and a reply at 50 codes a second, far short of today's 60 s
turns. A guard would be needed: a cache sized to each turn, or two.

**Cost:** one L40S container, about 8 minutes: about \$0.25.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 995 passed, 2 skipped.

## 2. The result (row 3)

`runs/r4-gpu/step-profile-l40s.json`, at `a4502d2`, one L40S container (driver 610.57.04, the slower of the
two hosts M31 met).

**The Talker** is bigger than the design notes of M32b and M33 assumed: 24 layers, hidden 896, but an
intermediate width of 18,944, so 1,351 M parameters. A code reads 2.64 GB of weights: 3.06 ms on an L40S
at best. Its cache's keys and values: 302 MB at 6144 places (0.35 ms to read), 101 MB at 2048.

**The captured step on three caches:**

| cache places | ms a code p50 | p90 | codes the 6144 cache's | turns cut |
|---:|---:|---:|---|---:|
| 6144 (today's) | 10.26 | 10.28 | (the reference) | 0 |
| 2048 | **6.62** | 6.63 | 10 of 10 | 0 |
| 1024 | failed | | | |

The 1024-place cache failed on the tenth clip, the 53 s probe clip, whose prompt alone takes 1,422
places; the others take 239 to 304, and their replies 233 to 503 codes. (Its row's places-filled figures
take in its warm-up turns, so they are not read.)

**One turn's kernels**, the loop eager (11.77 and 7.86 ms a code in all, against 10.26 and 6.62 captured):

| kind | 6144 places | 2048 places |
|---|---:|---:|
| attention (one memory-efficient attention kernel) | 5.10 ms | 1.75 ms |
| matrix products (matrix-vector kernels) | 4.15 | 4.14 |
| the rest (copies, element-wise, norms, the cache's writes) | 2.45 | 1.90 |
| sampling | 0.07 | 0.07 |

So M32b's lead was right where its premise was wrong: the attention over the cache's empty places was
the biggest single cost, about half the step; the matrix products, 4.1 ms, are within 1.4 of the
weights' read time.

**The follow-up line, written before the run: 2048 places at 8 ms a code or less, no turn cut. Met:
6.62 ms.** A design note for a right-sized cache in R4 streamed follows; M33 is done.

**What it could buy** (a projection from M31's DEV block timings, only the Talker's time a code changed,
from 12.7 to 6.62 ms: not measured): a block of 96 codes made in about 1.08 s instead of 1.66, the first
block at about 1.28 s instead of 1.43; playback could start without a stall at about 1.9 s, or 2.4 s
under today's start rule (hold until 0.5 s of audio exists), against 3.08 s today.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 995 passed, 2 skipped.
