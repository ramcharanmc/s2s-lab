# M26 run log: the quantization ladder with a prosody guardrail

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen, from the milestone's first day. Times are CDT. The earlier logs:
`docs/ft1-log.md`, `docs/ft2-log.md`, `docs/ft3-log.md`.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 09-27, afternoon | — | the design, five decisions | — | — |
| 2 | 09-27 15:11 | checks before any download | no install needed; the inputs match | — | — |
| 3 | 09-27 15:24 | three model files, four voices downloaded | all arrived, 581 MB | — | — |
| 4 | 09-27, afternoon | the pilot, scratch | fp16 returned no audio for one sentence of two | — | — |
| 5 | 09-27, afternoon | the dev run, scratch, 24 renders a rung | the 8-bit copy speaks lower and flatter | `src/s2slab/quantladder.py`, tests | this commit |

## 1. The design

v1's P3 claims that as a synthesizer is compressed, its prosody degrades at a rung where its
intelligibility is still flat. Decided with the author on 2026-09-27:

| Decision | Chosen |
|---|---|
| The model | Kokoro-82M, the TTS of rungs R0 to R3, through onnx-community's published ONNX copies (Apache-2.0) |
| The rungs | the kokoro package (torch), ONNX fp32, ONNX fp16, ONNX 8-bit |
| The measures | severity's four (pitch level, range, rate, pauses) against fp32; WER of base.en on the audio; size and speed |
| The text | dev: twelve spent sentences; test: fifty of the eval set's clean references, locked |
| Where | the laptop CPU, $0; a GPU run is optional and not decided |

Out of scope, in writing: TensorRT. Nothing here quantizes a model; the copies are the publisher's.

## 2. Checks before any download

| Check | Result |
|---|---|
| `onnxruntime` | already in the environment, 1.23.2 (it comes with `faster-whisper`) |
| The download command's form | run on two small files into a folder outside the repo |
| Phoneme ids | the ONNX repo's vocabulary is the package's, plus the pad id |
| Voices | the ONNX repo's `af_heart` file equals the package's voice pack, difference 0.0 |

So a difference between the torch rung and ONNX fp32 comes from the export, not from the inputs.

## 3. The downloads

Revision `1939ad2a8e416c0acfeecc08a694d14ef25f2231`, one file a command, into `data/m26/kokoro-onnx/`
(ignored by git):

    mkdir -p data/m26/kokoro-onnx
    hf download onnx-community/Kokoro-82M-v1.0-ONNX onnx/model.onnx --revision 1939ad2a8e416c0acfeecc08a694d14ef25f2231 --local-dir data/m26/kokoro-onnx
    hf download onnx-community/Kokoro-82M-v1.0-ONNX onnx/model_fp16.onnx --revision 1939ad2a8e416c0acfeecc08a694d14ef25f2231 --local-dir data/m26/kokoro-onnx
    hf download onnx-community/Kokoro-82M-v1.0-ONNX onnx/model_quantized.onnx --revision 1939ad2a8e416c0acfeecc08a694d14ef25f2231 --local-dir data/m26/kokoro-onnx
    hf download onnx-community/Kokoro-82M-v1.0-ONNX voices/af_heart.bin voices/am_michael.bin voices/af_sarah.bin voices/am_adam.bin --revision 1939ad2a8e416c0acfeecc08a694d14ef25f2231 --local-dir data/m26/kokoro-onnx

| File | Bytes | SHA-256 |
|---|---:|---|
| `onnx/model.onnx` | 325,532,232 | `8fbea51e…21a34cb` |
| `onnx/model_fp16.onnx` | 163,234,740 | `ba4527a8…c35c334a` |
| `onnx/model_quantized.onnx` | 92,361,116 | `fbae9257…028a1478` |

In full in `quantladder.RUNGS`, which refuses a file that differs.

## 4. The pilot

Two dev sentences, one voice, each rung rendered twice (fp16 six times). Scratch only.

| Question from the design | Answer |
|---|---|
| Does the ONNX model take the package's ids? | yes |
| Is ONNX Runtime on a CPU deterministic? | yes: every rung gave the same samples twice |
| Is the torch package? | **no**: two renders of one sentence differ, by up to 9% in pitch range |
| Does fp16 run on a CPU? | **not always**: every sample of one sentence was not a number, six times of six; the other sentence was fine |
| Speed | fp32 and fp16 about 0.33 of real time; **the 8-bit copy 1.3**: four times slower, and slower than speech |

Two changes to the design followed:

- **The noise floor is torch against torch.** ONNX repeats itself exactly, so re-rendering it says
  nothing. The torch package's own run-to-run difference is the measure of "a change that means
  nothing": five renders of every sentence, each against the first.
- **A render that is not a number is its own outcome.** It is counted, kept out of every measure,
  and not scored as speech.

Why fp16 fails on that sentence was not looked into. Not tested: the same file on a GPU.

## 5. Commit 1 and the dev run

`src/s2slab/quantladder.py`: the two speakers, the comparison, the run, the floor, the table.
Seven tests on a synthetic voice and fake speakers. **Run before the hand-over:** the tests, and the
whole dev run on the real models, in a scratch copy of the tree.

The dev run, scratch, 2 voices x 12 sentences, 560 s on the laptop CPU. Dev data, not evidence.
Within = PR-009's tolerance (3 Hz for the offset, 10% for the others).

| Rung | Size | Audio | Pitch offset | Pitch range | Rate | Pauses | WER raw / norm | Real-time factor |
|---|---:|---:|---|---|---|---|---:|---:|
| ONNX fp32 (the reference) | 325.5 MB | 24 of 24 | — | — | — | — | 2.1% / 1.1% | 0.33 |
| ONNX fp16 | 163.2 MB | **23 of 24** | 23 of 23 | 23 of 23 | 23 of 23 | 23 of 23 | 2.2% / 1.1% | 0.34 |
| ONNX 8-bit | 92.4 MB | 24 of 24 | **9 of 24** | **17 of 24** | 24 of 24 | 23 of 24 | 2.6% / 1.6% | **1.32** |
| torch package | — | 24 of 24 | 24 of 24 | 23 of 24 | 24 of 24 | 23 of 24 | 2.1% / 1.1% | 0.38 |

| Noise floor, 96 pairs | median | 95th percentile | max |
|---|---:|---:|---:|
| pitch offset, Hz | 0.30 | 0.99 | 1.79 |
| pitch range | 0.011 | 0.040 | 0.099 |
| rate | 0.000 | 0.006 | 0.021 |
| pauses | 0.000 | 0.028 | 0.113 |

- **The 8-bit copy speaks lower and flatter.** Its median pitch is lower than fp32's on 23 of 24
  renders, by 3.3 Hz on average and up to 7.9; its range is 7% narrower on average. Both are well
  outside the noise floor's 95th percentile (1.0 Hz; 4%).
- **Its words barely change.** 10 errors in 380 words against 8 (raw), 6 against 4 (normalized).
- **fp16 is fp32 when it speaks**, within the noise floor on every measure, and it returned no
  audio for one render of 24 (`af_heart`, d1, the pilot's sentence).
- **The export is faithful.** The torch package against ONNX fp32 sits at the noise floor.
- **Smaller is slower here.** The 8-bit file is 28% of fp32's size and takes four times as long.

This is the pattern v1's P3 predicts, on dev data. The registration comes next.

## Open, for the registration

- How a rung that returns no audio counts.
- Whether the GPU run is added, now that fp16 fails on a CPU.
- The bars: the noise floor's size sets how small a change can be called a change.
