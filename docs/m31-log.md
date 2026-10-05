# M31 run log: R4 streams its reply

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m30-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design: a streaming R4 beside R4, as R1 is to R0; the profile first; the author's go | — | — |
| 2 | 10-04 | the stage profile's tests, laptop, fake model | the clock times each stage and leaves the reply as it was | `r4profile.py`, tests, `r4_bench.py::profile` | 911fea3 |
| 3 | 10-04 | `r4_bench.py::profile`, the author, L40S, 30 probe clips, \$0.278 | the Talker is 75% of R4's time and runs at real time (19.8 ms a code, 0.99); token2wav about 890 ms for any block of 25 to 100 codes; streamed as designed, a first block at about 2.1 to 3.5 s but production 1.4 to 2.8 times slower than playback | — | the run file, this commit |
| 4 | 10-04 | — | the author's choice (A of three): before any streaming, can the Talker and the DiT run faster with standard tools, unchanged in what they make | `r4speed.py`, tests, `r4_bench.py::speed`; `OmniS2S.speak` takes extra arguments for generate | f7e8d99 |
| 5 | 10-04 | `r4_bench.py::speed`, the author, L40S, 10 probe clips, \$0.164 | the DiT compiled 4.2, 3.2 and 2.1 times faster on 25, 50 and 100 codes, its audio within 0.003 to 0.06 of eager's; both static Talker variants failed at once: `TypeError` (`int` minus `dict`) | — | the run file, this commit |
| 6 | 10-04 | the failure reproduced on the laptop, a tiny random Talker | with a static cache, generate hands the Talker a dict of masks; given its 2-D mask back it runs, codes identical to eager's; compiled, its layers need no mask on one-code steps: one graph, no breaks, codes identical | `keep_2d_mask`; `compile_layers` without the mask; the variants eager, static, static+layers; `::speed --tag` | 567e7f2 |
| 7 | 10-04 | `r4_bench.py::speed --tag mask`, the author, L40S, the same 10 probe clips, \$0.287 | both fixes held on the real Talker: static+layers 12.7 ms a code, 0.63 of real time (eager 20.2, static alone 23.6), no graph breaks; its codes differ from eager's on 10 of 10, from code 8 to 200 on; the DiT as in the first run | — | the run file, this commit |
| 8 | 10-04 | the stream's tests, laptop, a fake model; then `transformers`' own Talker and token2wav, tiny and random | a block decoder that reproduces a whole reply sample for sample; the stream's codes the fast rung's; the real Talker path writes eager's codes; 480 samples a code; my cache reset outside inference mode caught and fixed | `r4stream.py` (`r4-gpu-fast`, `r4-gpu-stream`), `r4streamdev.py`, tests, `::streamdev` | ceae735 |
| 9 | 10-04 | `r4_bench.py::streamdev`, the author, L40S, 30 probe clips, \$0.685 | the stream's first sound 1,629 ms with no buffer, but every turn stalled; with 0.5 s held, 3,284 against R4's 5,865 and no stall; its codes the fast rung's on 30 of 30; Whisper 9 errors in 622 words against R4's 6; `r4-gpu-fast` no faster than R4 in this container | — | the run file, this commit |
| 10 | 10-04 | the author listens to the development replies | "edges sounded fine"; the author's choice (A of three): register now on the stall-free setting | the setting frozen; `pr015.py`, tests, `::pr015`; PR-015 registered | e2a4b4e |
| 11 | 10-04 | `r4_bench.py::pr015 --confirm PR-015`, the author, L40S, the 200 eval clips, once, \$2.592 | **PR-015 supported**: first sound 3,331 ms against R4's 6,806 (0.49), p90 3,728 against 11,784 (0.32); no stall; Whisper 1.1% against 1.3%; total 1.14 times R4's; the same text and codes on 200 of 200 | — | the run file, PR-015's Results, this commit |

## 1. The design

**Where R4's wait comes from.** R4 (`src/s2slab/r4.py`) calls Qwen2.5-Omni's `generate` once. Inside it
three stages run one after another, each to its end before the next begins: the Thinker writes the
whole text reply (greedy, at most 60 tokens); the Talker writes every speech code for it (sampled,
seeded); token2wav turns all the codes into audio, a DiT making a mel spectrogram in 10 solver steps
and BigVGAN the 24 kHz waveform. The first sample exists only when the last stage ends. On the L40S in
one container (M28, `runs/r4-gpu/evalset-modal-l40s-ladder.json`): first audio 6,859 ms at p50, 11,974
at p90, replies 7.4 s at the median, and **0.93 s of wait for every second of reply** (0.86 to 1.14
over the 200 turns). Every other rung's first audio is under 700 ms.

**The idea.** A new rung, `r4-gpu-stream`: the same model, prompt, greedy text, voice and seed, with
the audio handed out (`Rung.emit`) as it is made. R4 stays as measured, so the difference is what
streaming alone buys, as R1 is to R0. The rungs already record first audio, first sound and stalls for
a streaming rung.

| step | what streams | first audio becomes | work |
|---|---|---|---|
| S1 | token2wav decodes in blocks, after the Talker has finished | Thinker + Talker + one block | small |
| **S2** | **the Talker's codes go to a streamer; token2wav decodes each block as it fills** | **Thinker + one block of codes + its decode** | **medium** |
| S3 | the Talker also starts while the Thinker writes (the model is built for it; `transformers` does not do it) | a few tokens + one block | large: `generate` rewritten |

The aim is S2. The model's DiT already attends by blocks (one back, one ahead in some layers), so
decoding a block at a time is how it was built to run. S3 stays a later option.

**The risk is stalls, not first audio.** The whole call runs at 0.93 of real time. If the Thinker takes
a second of that, the Talker and token2wav run near 0.8, and decoding by blocks re-reads each block's
neighbours. Below real time, playback starts early and then stops; starting early and stalling is not
a win (R1's rule), so every result reports both, and the remedy, if needed, is a start buffer: N
seconds of audio made before playback begins, a measured trade.

**Quality.** The streamed audio cannot equal R4's: each block starts from its own diffusion noise. So
against R4 on the same clips: the words heard (Whisper on the reply), the reply's length, clicks at
block edges, and the author's listening.

**Plan.** (1) The profile: each stage timed on the 30 probe clips on an L40S, about \$0.30; it settles
whether S2 can keep up and sets the block size. (2) `r4-gpu-stream`, S2, tested with a fake model, then
tried on development clips on Modal, about \$0.50. (3) A registration, PR-015, before the eval set, its
bars from step 2. (4) R4 and `r4-gpu-stream` on the 200 eval clips in one L40S container, about \$2. In
all about \$2.50 to \$3. Left out: Qwen3-Omni, which streams by design; it changes the model, not only
the delivery.

## 2. Step 1, the stage profile (built)

`src/s2slab/r4profile.py` wraps the model's stages with timers that synchronise the GPU before and
after each call: the Thinker's and the Talker's `generate`, token2wav, and token2wav's DiT and BigVGAN
inside it. The wrappers call the originals and return their outputs unchanged, so the replies are R4's
own: the same `OmniS2S.speak`, settings and seed. After each turn, token2wav decodes the reply's first
25, 50 and 100 codes alone, twice each (the first call at a new length can include one-off kernel
set-up), with the turn's own speaker conditioning.

From those times, per turn, arithmetic and not a measurement of a streaming rung:

- **codes per second** of reply audio;
- **the Talker's and token2wav's real-time factors** (milliseconds of work per millisecond of reply);
- for each block size b: **when a first block could exist** (the Thinker, plus b codes' share of the
  Talker, plus token2wav on b codes), and **whether a stream could keep up** (the Talker's factor plus
  token2wav's on blocks of b; below 1 it keeps up, before a block's neighbours are counted).

```
modal run modal_app/r4_bench.py::profile        one L40S; saves runs/r4-gpu/profile-l40s.json
```

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 929 passed, 2 skipped.

## 3. Step 1, the profile: the Talker runs at real time (at 911fea3)

Run by the author: `modal run modal_app/r4_bench.py::profile`, one L40S, 506 s in the function, 513 s
end to end, \$0.278. The 30 probe clips; `runs/r4-gpu/profile-l40s.json`.

| stage | p50 ms | p90 ms | share of all the time |
|---|---:|---:|---:|
| the Thinker | 658 | 1,072 | 7% |
| **the Talker** | **6,469** | **11,851** | **75%** |
| token2wav (its DiT 1,241 and BigVGAN 65 at p50) | 1,308 | 3,003 | 16% |
| R4's whole call | 8,622 | 16,199 | |

Replies: 25 text tokens, 326 codes and 6.5 s at p50; 50 codes a second of audio on every clip.

- **The Talker runs at real time.** 19.8 ms a code at p50 (19.3 to 32.1), 50 codes a second: a
  real-time factor of 0.99. However early playback began, the Talker alone would only just keep pace.
- **token2wav costs the same per call, whatever the length.** The first 25, 50 or 100 codes took 884 to
  898 ms at p50, the first call and the repeat alike; the whole reply took 902 ms for 207 codes and
  4,419 ms for 762. Decoding in blocks after the Talker (S1) costs more than decoding once: S1 is out.
- **Streamed as designed (S2), by arithmetic:** a first block could exist at about 2,058 ms (25 codes),
  2,552 (50) or 3,534 (100), against R4's 8,622 on these clips; but making the audio would take 2.78,
  1.88 or 1.43 times as long as playing it, before a block's neighbours are counted. Playback would
  stall throughout.
- **Why it looks fixable.** The Talker, a small model, takes about as long a step as the 7B Thinker
  (19.8 against 25.8 ms a token), and the DiT costs the same for 25 codes as for 100: both look bound
  by launching many small GPU operations, not by arithmetic. The standard remedy is a static KV cache
  and a compiled step replayed as a CUDA graph.

**What I had wrong in the design.** I wrote that the Talker and token2wav together would run "near 0.8"
of real time. The Talker alone runs at 0.99 and token2wav adds 0.21. And the whole call here took 1.32 s
a second of reply (per turn, p50), not M28's 0.93: R4's first L40S run took 1.28, so the M28 container
was the fast one (M28 found same-card containers up to 27% apart).

## 4. Step 1b, the speed probe (built)

The author chose A: before any streaming, find out whether the two slow parts can run faster with
standard tools and still make the same thing. `src/s2slab/r4speed.py`, on every third probe clip (10 of
30), each through R4's own call with R4's seed:

| Talker variant | what changes |
|---|---|
| eager | nothing: R4 as it runs |
| static+compile | `transformers`' own route: a static KV cache for the Talker only (`talker_cache_implementation="static"`), with which `generate` compiles the Talker's call for decoding. The Talker is not marked fully compilable in `transformers` 5.17, and its decoding step branches on the cache's length, so this may break into pieces or fail |
| static+layers | the same static cache, `transformers`' compile turned off, and only the Talker's decoder layers compiled (`mode="reduce-overhead"`, CUDA graphs) for one-code steps; the prefill stays eager |

For each: milliseconds per code, the real-time factor, the warm-up (compile) time, the graph breaks,
and whether its codes are identical to eager's on the same clip and seed. A sampled Talker can take
another code from a last-bit difference in a logit, so "identical" is reported as found.

**The DiT:** its network compiled (`reduce-overhead`; `default` if that fails), each output copied at
once (the Runge-Kutta solver keeps four alive, and a CUDA-graph replay would overwrite the last). Then
token2wav on the first 25, 50 and 100 codes of eager's replies, eager and compiled, each with the same
noise seed: milliseconds, and the largest sample difference.

A variant that fails is recorded with its error and the next one runs. The estimates use the fastest
Talker variant that ran. `OmniS2S.speak` now passes extra keyword arguments to `generate`; R4 passes
none, so R4 is unchanged.

```
modal run modal_app/r4_bench.py::speed          one L40S, about 12 min, about $0.40; saves runs/r4-gpu/speed-l40s.json
```

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 936 passed, 2 skipped.

## 5. Step 1b, the first run: the DiT runs 2 to 4 times faster compiled; the Talker's static cache failed (at f7e8d99)

Run by the author: `modal run modal_app/r4_bench.py::speed`, one L40S, 298 s in the function, 303 s
end to end, \$0.164. Every third probe clip (10); `runs/r4-gpu/speed-l40s.json`.

| token2wav on a reply's first codes | eager ms p50 | compiled ms p50 | faster | largest sample difference, 10 replies |
|---:|---:|---:|---:|---:|
| 25 codes | 866 | 206 | 4.2 times | 0.0027 |
| 50 codes | 849 | 266 | 3.2 times | 0.0215 |
| 100 codes | 854 | 412 | 2.1 times | 0.0605 |

- **The DiT, compiled with CUDA graphs** (`reduce-overhead`, the first mode tried), now costs more for
  more codes (206, 266, 412 ms): what remains is the arithmetic, not the launching. A block of 50 codes
  (one second of audio) takes 266 ms.
- **It is not the same audio.** From the same noise seed, compiled and eager differ by up to 0.0027 on
  25 codes and 0.06 on 100, growing with the length: a different order of floating-point operations in
  the compiled kernels, carried through 10 solver steps. Whether that can be heard, or changes the
  words, is not measured yet; the streaming rung's checks (the words heard, the author's listening)
  will cover it.
- **The Talker, eager:** 19.3 ms a code at p50 (19.6 at p90), a real-time factor of 1.0, as in step 1.
- **Both static Talker variants failed at once**, before any timed turn: `TypeError: unsupported operand
  type(s) for -: 'int' and 'dict'`.
- **With the eager Talker and the compiled DiT, by arithmetic:** a first block at about 1,337 ms (25
  codes), 1,878 (50) or 2,987 (100); a stream at 1.38, 1.23 or 1.17 times the length of the audio. The
  DiT no longer stops a stream; the Talker, at real time, still does.

**My display error:** the printed table showed the 100-code difference as "0.1", rounded to one place;
the file holds 0.0605. The table now prints four places.

**The cause, reproduced on the laptop** (entry 6) with a tiny random Talker built from `transformers`'
own configuration class (2 layers, the real code): with a static cache, `generate` prepares the
attention masks itself and hands the model a dict of them, one per attention type; the Talker's
forward computes its rotary positions from the plain 2-D mask (`1 - attention_mask`) and fails on the
dict. A mistake in `transformers` 5.17's Talker, not in R4's settings. Two fixes, both in
`r4speed.py`:

- `keep_2d_mask` gives the Talker its 2-D mask back. On the tiny Talker: 40 codes identical to eager's.
- Compiled that way, the decoder layers saw a mask one longer at every step, recompiled up to torch's
  limit (8) and then ran eager. `compile_layers` now passes no mask on one-code steps: with one reply
  and no padding the decoder builds the same causal mask from the cache position. It refuses a padded
  mask. On the tiny Talker, on the laptop's CPU (compile mode "default": CUDA graphs need a GPU): one
  graph, no breaks, no recompiles, 40 codes identical to eager's. The real Talker on the L40S is the
  second run.

The variants are now eager, static (the cache with the 2-D mask) and static+layers (that, with the
layers compiled). `transformers`' own whole-call compile is dropped: it would see the same growing
mask.

## 6. Step 1b, the second run (built)

The same probe with the fixes, on the same 10 clips, saved under its own name:

```
modal run modal_app/r4_bench.py::speed --tag mask      one L40S, about 5 min, about $0.17; saves runs/r4-gpu/speed-l40s-mask.json
```

What it settles: whether the real Talker, compiled, runs well under real time, and whether its codes
stay identical to eager's. Below about 0.7 (with the DiT's 0.27 for blocks of 50 codes), a stream keeps
up before a block's neighbours are counted.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 938 passed, 2 skipped.

## 7. Step 1b, the second run: the Talker at 0.63 of real time (at 567e7f2)

Run by the author: `modal run modal_app/r4_bench.py::speed --tag mask`, one L40S, 521 s in the function,
530 s end to end, \$0.287. The same 10 clips; `runs/r4-gpu/speed-l40s-mask.json`.

| Talker | ms a code p50 | p90 | real-time factor | warm-up s | codes identical to eager's |
|---|---:|---:|---:|---|---:|
| eager (R4) | 20.2 | 20.4 | 1.0 | - | - |
| static (the cache, its 2-D mask) | 23.6 | 24.1 | 1.2 | 8.2, 8.6 | 0 of 10 |
| **static+layers** (and the layers compiled) | **12.7** | **13.0** | **0.63** | 41.7, 5.2 | 0 of 10 |

- **Both fixes held on the real Talker**: no error, no graph break. Compiled, it takes 12.7 ms a code,
  1.6 times eager's speed; the static cache alone is slower than eager (each step attends over the
  whole cache, the prompt and 4,096 places). Where the remaining time goes, not measured: most likely
  `generate`'s own work around each step (sampling, its stopping checks), which is not compiled.
- **The codes are not eager's.** In both static variants they part from eager's on all 10 clips, from
  code 8 to code 200 on (the first codes always agree: the prompt and the positions are right). The
  static cache already does it, without any compile: a last-bit difference in a logit, and a sampled
  Talker takes another code from there. The replies run from 12% shorter to 22% longer than eager's.
  So a fast R4 speaks another sample of the same reply; the words come from the Thinker, untouched,
  and whether the speech still says them is what the next step measures.
- **The DiT, as in the first run:** compiled 219, 273 and 421 ms on 25, 50 and 100 codes; the largest
  sample difference 0.0027, 0.0211 and 0.0737.
- **By arithmetic, with the compiled Talker and DiT:** a first block at about 1,176 ms (25 codes),
  1,549 (50) or 2,332 (100); a stream at 1.08, 0.91 or 0.85 of real time, before a block's neighbours
  are counted. R4 waits 6.9 s on the eval set (M28) and 8.6 s on these clips.

The author chose A (of three): build the streaming rung on this path now; a faster Talker (its whole
step compiled, in a loop of our own) can follow if the stream stalls too much.

## 8. Step 2, the fast and streamed rungs (built)

`src/s2slab/r4stream.py`, two rungs on R4's model, prompt, greedy text, voice and seed:

| rung | what changes from R4 |
|---|---|
| `r4-gpu-fast` | the Talker with a static cache of its own (6,144 places, made once, reset each turn), its 2-D mask, and its layers compiled; the reply decoded whole, as R4's |
| `r4-gpu-stream` | the same Talker; its codes handed over one by one (`generate`'s streamer) and decoded a block at a time by the compiled DiT, each block handed out (`Rung.emit`) as soon as it exists |

So R4 to `r4-gpu-fast` is what compiling the Talker buys, and `r4-gpu-fast` to `r4-gpu-stream` what
streaming buys, as R1 is to R0.

- **A fixed cache.** `transformers` sizes a static cache to each prompt plus 4,096; every longer prompt
  would make a new cache and recompile the layers. One cache of 6,144, reset each turn, holds a turn of
  up to 60 s (about 25 prompt tokens a second of audio, plus about 110 of text) and R4's own cap of
  4,096 codes; a longer turn is refused. The longest probe clip is 53.4 s (my first size, for 30 s,
  would have stopped the development run on it; caught before the hand-over), the eval set's 17.0 s.
- **The blocks** (`BlockDecoder`). The DiT attends within blocks of `block_size` mel frames (12 codes in
  `transformers`' default configuration; the rung reads the checkpoint's own at load), one block back in
  two of its layers and one ahead in one: a frame depends on at most two blocks back and one ahead. With
  blocks of 12 codes, a block of codes [s, e) is decoded in the window [s - 24, e + 12), aligned to the
  DiT's blocks, and only [s, e) kept: the attention it sees is the attention the whole reply would give it.
  The noise is its own, so neighbouring blocks are crossfaded over one code (20 ms). A block is decoded
  once 12 more codes exist, or the reply has ended. The DiT runs compiled on the steady windows and
  eager on the last, shorter one.
- **The Talker is not disturbed.** Each block is decoded inside the Talker's own `generate` call, when
  the streamer hands over a code, inside `torch.random.fork_rng`: the DiT's noise does not move the
  Talker's sampling, so the stream writes the fast rung's codes on the same clip and seed. A test
  checks it with a fake Talker that draws each code as it streams.
- **A start buffer** (`StartBuffer`), if one is needed: the first pieces held back until that much audio
  exists, then everything played on.
- **The blocks must be whole DiT blocks.** The rung, and the development run before any turn, refuse a
  setting that is not a multiple of the DiT's block (read from the checkpoint at load).

**Checked on the laptop before the hand-over**, with `transformers`' own code and tiny random weights:

- the Talker through the stream's path (our cache reset between turns, the 2-D mask, the layers
  compiled, on the CPU, and the streamer) wrote eager's codes on four turns of different prompt
  lengths: one graph, no breaks, no recompiles. That check caught a mistake of mine: the cache's
  tensors are made inside `generate`'s inference mode, and resetting them outside it fails, so every
  turn after the first would have crashed on Modal. The reset now runs inside it.
- the real token2wav code (a small DiT): 480 samples a code from a window, the DiT's context read as 24
  codes back and 12 ahead, and 150 codes streamed in four pieces to exactly 72,000 samples, every
  window starting on a DiT block.

**Step 2's development run** (`src/s2slab/r4streamdev.py`), one container, one set of weights, the 30
probe clips: R4, then `r4-gpu-fast`, then the stream with its first block and later blocks at (24, 96),
(24, 192) and (48, 96) codes, no start buffer. For each turn: first sound, stalls, total, the reply's
length; the stream's codes against the fast rung's; and Whisper small.en on the reply audio against the
Thinker's text (the same text for all three), so a stream that garbles its seams shows. What a start
buffer of 0.5 to 3 s would do is computed afterwards from each turn's pieces. Three replies of each rung
are written to `results/` for the author to listen to.

```
modal run modal_app/r4_bench.py::streamdev      one L40S, about 25 min, about $0.80; saves runs/r4-gpu/stream-dev-l40s.json
```

From it: the setting and the start buffer, then PR-015, registered before the eval set.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 957 passed, 2 skipped.

## 9. Step 2's development run: sooner, if playback waits for the second block (at ceae735)

Run by the author: `modal run modal_app/r4_bench.py::streamdev`, one L40S, 1,256 s in the function, 1,264
s end to end, \$0.685. The 30 probe clips; `runs/r4-gpu/stream-dev-l40s.json`; 15 replies in
`results/r4-gpu/streamdev/` (not committed).

| rung | first sound p50 ms | p90 | stalled turns | total p50 ms | reply s p50 | Whisper's errors |
|---|---:|---:|---:|---:|---:|---:|
| `r4-gpu` (R4) | 5,865 | 10,831 | 0 of 30 | 5,675 | 6.2 | 6 of 622 words |
| `r4-gpu-fast` | 6,216 | 10,290 | 0 | 6,006 | 6.6 | 10 |
| stream, 24 + 96, no buffer | 1,629 | 2,002 | **30** (p90 1,234 ms) | 6,931 | 6.6 | 9 |
| **stream, 24 + 96, 0.5 s held** | **3,284** | **3,654** | **0** | | | |
| stream, 48 + 96, 1 s held | 3,592 | 3,965 | 0 | 6,860 | 6.6 | 8 |
| stream, 24 + 192, 0.5 s held | 4,641 | 5,136 | 0 | 6,250 | 6.6 | 9 |

(The held rows are computed from the same turns' pieces: `r4streamdev.buffered`, which the tests show
gives what `StartBuffer` gives.)

- **The stream works.** Its codes were the fast rung's on 30 of 30 turns in every setting: decoding the
  blocks inside the Talker's call did not move its sampling. Whisper heard 8 or 9 errors in 622 words,
  against R4's 6. The author listened and heard nothing at the edges.
- **But with no buffer every turn stalls.** The first block is 0.46 s of audio, and the next arrives
  1.65 s later (p50): the stream makes its audio at 1.03 of real time, so playback catches up at once.
  Held until 0.5 s exists, playback starts with the second block, and no turn stalled; any hold from
  0.5 to 2 s gives the same, because the second block arrives whole.
- **`r4-gpu-fast` was no faster than R4 here** (paired, 0.998 of its time a second of reply). This
  container was a fast one: R4 took 0.91 s of wait a second of reply, against 1.32 in step 1's, as in
  M28. The compiled Talker runs at about 13 ms a code on both hosts; the eager one at 13 to 20 by host.
  Probably `generate`'s own loop around each step, run by the CPU, is the floor both reach, and a
  faster host lifts only the eager one. Not measured stage by stage in this container.
- **What the start depends on.** The stream's first sound grows with the reply's length (r = 0.98): the
  Thinker writes its whole text before the Talker starts.

On the development data, by the same paired bootstrap PR-002 used: first-sound p50 ratio 0.56 (95%
interval 0.46 to 0.61), p90 0.34 (0.31 to 0.43), total p50 1.22 (1.08 to 1.24); the smallest margin
before a stall, on one turn, 102 ms (median 725).

## 10. PR-015, registered

The author's choice (A of three: register now; B, a Talker loop of our own first; C, stop), and his
listening: "edges sounded fine". The setting frozen by a rule written after the development run, and
said so in the entry: the lowest first-sound p50 among the settings and buffers with no stalled turn,
(24, 96) held to 0.5 s. PR-015 (`docs/preregistration.md`) asks, on the eval set, whether the stream's
first sound is at most 0.70 of R4's at the median and 0.55 at p90, with its total at most 1.40, at most
20 turns stalling (10 for more than 200 ms), and Whisper's word error rate within 2 points of R4's; and
it is a valid test only if the stream says R4's text and the fast rung's codes on 190 of 200 turns.
PR-002's correlation criterion is not used: the Thinker still comes first, and that is said too.

`src/s2slab/pr015.py` runs R4, `r4-gpu-fast` and the stream in one container (`r4streamdev.run` with the
frozen setting, its buffer in the rung) and computes the criteria.

```
modal run modal_app/r4_bench.py::pr015 --confirm PR-015      once; one L40S, about 1.5 to 2 h, about $3 to $4; saves runs/r4-gpu/pr015.json
```

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 965 passed, 2 skipped.

## 11. PR-015: supported (at e2a4b4e)

Run once by the author: `modal run modal_app/r4_bench.py::pr015 --confirm PR-015`, one L40S, 4,778 s in
the function, 4,785 s end to end, \$2.592. `runs/r4-gpu/pr015.json`; the analysis's output is recorded
verbatim under PR-015's *Results*.

| on the 200 eval clips | `r4-gpu-stream` | R4 | `r4-gpu-fast` |
|---|---:|---:|---:|
| first sound p50 / p90, ms | **3,331 / 3,728** | 6,806 / 11,784 | 7,010 / 12,439 |
| total p50 / p90, ms | 7,530 / 12,638 | 6,602 / 11,584 | 6,810 / 12,239 |
| stalled turns | **0 of 200** | — | — |
| Whisper's word error rate on the reply | 1.1% | 1.3% | 1.1% |

All five criteria held, with room: first sound 0.489 of R4's at the median (interval 0.471 to 0.504,
bar 0.70) and 0.316 at p90 (bar 0.55); the total 1.141 (bar 1.40). The stream said R4's text and the
fast rung's codes on 200 of 200 turns. Every prediction came close: 0.50, 0.33 and 1.2 predicted.

Exploratory: no turn stalled, but four came within 200 ms of it (the closest 44 ms); the gain is the
same in every cohort (3.3 to 3.4 s against 6.8 to 7.0); the stream's first sound still grows with the
reply's length, the Thinker writing its whole text first, but within 2.2 to 4.1 s against R4's up to
18.4; and the compiled Talker alone bought nothing in this container (1.03 of R4's time).

## 12. M31, closed

What M31 found and built, in order:

1. **Where R4's wait goes** (step 1): the Talker, 75%, at real time; token2wav a fixed ~890 ms a call.
2. **What standard tools buy** (step 1b): the DiT compiled, 2 to 4 times faster; the Talker compiled
   after two fixes for a mistake in `transformers` 5.17 (the static cache hands it a dict of masks), 1.6
   times faster on a slow host and no faster on a fast one, where probably `generate`'s own loop is the
   floor (not timed stage by stage).
3. **R4 streamed** (step 2): codes decoded a block at a time inside the Talker's call, the DiT's
   context kept, its noise kept from the Talker's sampling; on development, every turn stalled without
   a start buffer, none with 0.5 s.
4. **PR-015** (step 3): on the eval set, the caller hears R4's reply in half the time at the median and
   a third at p90, with no stall and the same words.

Cost on Modal, M31: \$0.278 + \$0.164 + \$0.287 + \$0.685 + \$2.592 = **\$4.006**. My errors along the way,
each caught before it cost a run but one: the display that rounded 0.06 to "0.1"; a cache sized for
30 s turns when a probe clip ran 53.4 s; the cache reset outside inference mode; the first speed run's
Talker variants, which failed on Modal (\$0.164) because I had not run `transformers`' static path on
the laptop first.

**Left, if wanted** (the backlog): a Talker decode loop of our own, its whole step compiled, so a stream
can start from its first block without stalling (about 1.6 s on development); a stream that starts from
the Thinker's first words; the compiled Talker timed stage by stage on a slow and a fast host.
