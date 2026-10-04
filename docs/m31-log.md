# M31 run log: R4 streams its reply

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m30-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design: a streaming R4 beside R4, as R1 is to R0; the profile first; the author's go | — | — |
| 2 | 10-04 | the stage profile's tests, laptop, fake model | the clock times each stage and leaves the reply as it was | `r4profile.py`, tests, `r4_bench.py::profile` | this commit |

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
