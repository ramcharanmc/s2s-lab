# M36 run log: a compile cache for the L40S

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m35-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-15 | — | the design: what the L40S compiles kept on a Modal volume between wakes; the author's go, M36 before M37 (a timed start rule, with a registration). Option 2 of M35's candidates, R4 streamed from the Thinker's first words, analysed and set aside (below) | — | — |
| 2 | 10-15 | the tests, laptop, torch.compiler's save and load replaced by recorders | no file: nothing loaded, the first wake's blob written; a later wake loads it and writes only when it compiled more; a broken file costs only the cache and is replaced; the L40S's load reads the cache before anything compiles and writes it after | `compilecache.py`, `modal_app/demo.py` (the volume `s2s-lab-compile-cache` on `Speaks`), tests | this commit |

## 1. The design

**Why.** M35's phases: the L40S wakes in 186 s, most of it compiling (the DiT's windows 65 s, the sized
cache's decoder 57 s, most of R4's 61 s the Talker's layers); its weights load in about 4 s. Each container
starts empty, so every wake compiles everything again.

**The change.** `s2slab.compilecache`: before the load, the blob `torch.compiler.save_cache_artifacts`
wrote at an earlier wake is loaded into the process (`load_cache_artifacts`: inductor's graphs,
AOTAutograd's, Triton's kernels, the autotuning results); after it, the process's blob is written back when
there was none, when it failed to load, or when it grew (new graphs compiled). On a volume,
`s2s-lab-compile-cache`, committed when written. Dynamo still traces and the CUDA graphs are still captured
at every wake. PyTorch keys each entry by its version and the graph's source: a change compiles afresh.

**Expected.** The first wake after this fills the cache and costs what it did; later wakes perhaps 70 to
110 s. Measured, not promised: two cold wakes in one `modal serve` session (about \$0.40), the phases
printed as in M35.

**Option 2 set aside** (R4 streamed from the Thinker's first words). The Talker writes code *k* only after
the Thinker's reply token *k*; playback that never stalls must wait for the second block (code 120), less
the first block's 0.46 s, under any start rule. On one GPU the time to code 120 is the Thinker's whole reply
(replies are capped at 60 tokens) plus 120 Talker steps, both bound by reading weights (the Talker's
matrix work near the L40S's floor, M33; the Thinker about 20 ms a token against a floor of about 18):
interleaving them changes their order, not their sum. So about no gain at first sound, unless the two
models ran on two GPUs. Not built; not measured (a \$0.30 check was offered and not needed).

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 1014 passed, 2 skipped.
