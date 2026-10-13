# M33 run log: where a Talker step's time goes

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m32-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-12 | — | the design: the Talker's size and floors, its captured step on caches of 6144, 2048 and 1024 places, one turn's kernels at 6144 and 2048; the follow-up line written before the run (2048 places at 8 ms a code or less, no turn cut); exploratory, the probe clips only; the author's go | — | — |
| 2 | 10-12 | the profile's tests, laptop, the tiny Talker of M32b | each cache size timed, the places each turn filled counted, a cache too small cutting the reply and saying so; the breakdown by kind of operation; a fault found in reading before any run: a new cache's tensors are made on its first write, so a capture on an unwritten cache would read the wrong ones | `r4stepprofile.py`, `::stepprofile`, tests; `talkerloop` refuses a capture on an unwritten cache, and the profile writes each new cache once (our loop, eager and capped) before any capture | this commit |

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
