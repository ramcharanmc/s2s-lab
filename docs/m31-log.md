# M31 run log: R4 streams its reply

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m30-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design: a streaming R4 beside R4, as R1 is to R0; the profile first; the author's go | — | — |
| 2 | 10-04 | the stage profile's tests, laptop, fake model | the clock times each stage and leaves the reply as it was | `r4profile.py`, tests, `r4_bench.py::profile` | 911fea3 |
| 3 | 10-04 | `r4_bench.py::profile`, the author, L40S, 30 probe clips, \$0.278 | the Talker is 75% of R4's time and runs at real time (19.8 ms a code, 0.99); token2wav about 890 ms for any block of 25 to 100 codes; streamed as designed, a first block at about 2.1 to 3.5 s but production 1.4 to 2.8 times slower than playback | — | the run file, this commit |
| 4 | 10-04 | — | the author's choice (A of three): before any streaming, can the Talker and the DiT run faster with standard tools, unchanged in what they make | `r4speed.py`, tests, `r4_bench.py::speed`; `OmniS2S.speak` takes extra arguments for generate | this commit |

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
