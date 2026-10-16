# M36 run log: a compile cache for the L40S

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m35-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-15 | — | the design: what the L40S compiles kept on a Modal volume between wakes; the author's go, M36 before M37 (a timed start rule, with a registration). Option 2 of M35's candidates, R4 streamed from the Thinker's first words, analysed and set aside (below) | — | — |
| 2 | 10-15 | the tests, laptop, torch.compiler's save and load replaced by recorders | no file: nothing loaded, the first wake's blob written; a later wake loads it and writes only when it compiled more; a broken file costs only the cache and is replaced; the L40S's load reads the cache before anything compiles and writes it after | `compilecache.py`, `modal_app/demo.py` (the volume `s2s-lab-compile-cache` on `Speaks`), tests | f7e5bef |
| 3 | 10-15 | `modal serve modal_app/demo.py`, the author, two cold L40S wakes | the first filled the cache (32 MB): loaded in 160.4 s (R4 57.4, the DiT's windows 54.4, the 2048 cache prepared 46.2); the second loaded the cache first (19.8 s) and took 311.4 s: R4 streamed's compiles fell from 103 s to 23 (the DiT's windows 9.8, the cache 11.0), but **R4's own load rose from 57 s to 288**, and its weights loaded at 62 files a second, not 400 to 900 | — | — |
| 4 | 10-15 | — | R4's 288 s not explained by one sample: a slow host, or the loaded cache sending R4's compile (its layers in PyTorch's CUDA-graph mode) down a slower path. So: R4's load timed by phase, and the cache loaded after R4 and before R4 streamed, where it was shown to work; not deployed meanwhile | `r4stream` (`FastSpeechToSpeech` load phases), `demo.Live.load_all(before)`, `modal_app/demo.py`, tests | this commit |

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

## 2. The first session (rows 3 and 4)

| the L40S's load | first wake (no cache; filled after) | second wake (the cache loaded first, 19.8 s) |
|---|---:|---:|
| R4 (its weights, and its throwaway turn compiling the Talker) | 57.4 s | **288.3 s** |
| R4 streamed: the DiT's windows | 54.4 | **9.8** |
| R4 streamed: the 2048-place cache prepared | 46.2 | **11.0** |
| R4 streamed: the throwaway turn | 2.3 | 2.2 |
| in all | **160.4 s** | **311.4 s** |

The cache did what it was for on R4 streamed's compiles (103 s to 23). R4's own load is the unexplained
part: its weights came slowly (14 s, not 2 to 4) but that is not 230 s; R4 compiles its layers in PyTorch's
CUDA-graph mode, which the loaded cache may have sent down a slower path. One sample cannot tell. The first
wake, with no cache, was itself faster than M35's 186 s: hosts differ.

**The change (row 4).** R4's load is now timed by phase (its model, its throwaway turn); the cache is loaded
after R4 has loaded and before R4 streamed, where it worked, so R4 compiles as it did before M36. Not
deployed until a session shows the wake shorter: expected about 57 + 23 s plus the cache's load.
