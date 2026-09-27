# FT-3 run log (M25)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Times are UTC (the machine's commits are CDT, UTC-5). Costs marked "about" are computed from wall time
at Modal's A10 rate ($1.10/h); the dashboard has the bill. The ledger is `docs/cost-ledger.md`; the
registration is PR-012 in `docs/preregistration.md`. The same log for FT-1 is `docs/ft1-log.md`,
for FT-2 `docs/ft2-log.md`.

Logs of each run are under `results/` (not in git); the run files under `results/ft3/`.

| # | When (UTC) | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | committed 09-27 03:42 | — | FT-3's control and renderer, laptop env | `ft/src/ft3/control.py`, `synth.py`, tests | 680a246 |
| 2 | before 09-27 13:36 | `synth`, laptop, 3 attempts | 2 failures, then a render | `synth.py` (entry 2) | d4dfa4b |
| 3 | committed 09-27 13:36 | dev sweep, laptop | dev decisions fixed | `control.py`, `sweep.py`, Modal app | d4dfa4b |
| 4 | before 09-27 13:48 | `::data` | **failed**: image build | `ft3_train.py` image | 5e65d85 |
| 5 | 09-27 13:56 | `::data` | ran; pitch statistics unusable | `ft3_train.py` data stage | 12bc015 |
| 6 | 09-27 14:07 | `::data` | ran; statistics used | — | — |
| 7 | 09-27 14:09 | `::train` | trained 1,000 steps; **packaging failed** | `ft3_train.py` packaging | 3e6be7f |
| 8 | 09-27 14:39 | `::package` | model saved; **then failed** on the machine line | `ft3_train.py` `_machine` | 84cf350 |
| 9 | 09-27 14:43 | `::vocoder` | **failed** on the first clip | `ft/src/ft3/mels.py` (new) | 84cf350 |
| 10 | 09-27 14:50 | review and dry run, laptop | vocoder stage passed end to end on the CPU | `ft3_train.py`, tests | 84cf350 |
| 11 | 09-27 15:07 | `::vocoder` | trained 1,000 steps, packaged | — | — |
| 12 | 09-27, afternoon | models downloaded | first download unusable; second, file by file, verified | download command in `ft3_train.py`, README | 89edd05 |
| 13 | 09-27 17:46 | dev sweep, fine-tuned voice, laptop | pitch control worse than the base voice's | — | — |
| 14 | 09-27 18:38 | PR-012 registered | predicted: not supported | `ft/src/ft3/pr012.py`, tests, the entry | 89edd05 |
| 15 | 09-27 18:39 | the PR-012 run, laptop | **not supported**, all five criteria failed | — (results commit: evidence and text only) | — |

## 1. The control and the renderer (680a246)

`ft/src/ft3/control.py` turns page 09's severity targets into FastPitch's own inputs: per-token
durations divided by the rate, pauses of 150 ms or more scaled, pitch scaled and shifted in the
model's normalized units. Pure functions on arrays, tested without NeMo. `synth.py` renders with
NeMo FastPitch + HiFi-GAN. A separate env (`~/.venvs/s2slab-ft3`), because `nemo-toolkit` pins
`hydra-core`, `omegaconf` and `lightning` below what `ft/` uses.

## 2. First renders on the laptop: two API failures (fixed in d4dfa4b)

| Attempt | Failure | Cause | Fix in `synth.py` |
|---|---|---|---|
| 1 | `TypeError` in `infer(pitch=None)` | `infer()` adds the given pitch to the prediction; `None` cannot be added | predict through `forward()` instead |
| 2 | NeMo typecheck: "3 mandatory inputs" | `forward()` must be called with `text`, `durs`, `pitch` as keywords | `v.fastpitch(text=tokens, durs=None, pitch=None)` |
| 3 | rendered; pitch offset missed (-21 Hz for -16) | the range was scaled about the model's mean, so scale and offset interacted | scale about the neutral render's measured median |

Also fixed: the output file stem used Python's salted `hash()` (different every process) -> sha1.

## 3. Dev sweep and the two dev decisions (d4dfa4b)

Base LJSpeech voice, PR-009's four dev sentences, both conditions, six scores, 172 s on the CPU.

| Setting | Offset | Range | Rate | Pauses |
|---|---|---|---|---|
| baseline (audio median, gain 1.0) | 34/48 | 26/48 | 48/48 | 44/48 |
| token-median centre | 30/48 | 23/48 | — | — |
| **audio median, gain 1.2 (kept)** | 33/48 | 31/48 | 48/48 | 45/48 |

28 of 32 feature sweeps in order. The model answers a per-token range scale with about 0.83 of it,
hence the gain. Decisions are written as a comment block in `control.py`.

## 4. `::data`, first attempt: the image did not build (fixed in 5e65d85)

- **Failure:** `pip install nemo-toolkit[tts]==3.0.0` could not build wheels for `pyopenjtalk` and
  `cdifflib`: `No such file or directory: 'clang'`.
- **Cause:** both have no wheel for Python 3.12 and compile; Modal's added Python calls `clang`; the
  image had `gcc`/`g++` only.
- **Fix:** `apt_install` gains `clang`, `cmake`, `build-essential`. Only `clang` was named by the
  error; the other two are a precaution.
- **Cost:** image build only, no function ran. Log: `results/ft3-data-failed-build.txt`.

## 5. `::data`, second attempt: ran, but with the wrong pitch range (fixed in 12bc015)

- **Result:** 76 training clips (290.6 s), 2 validation. Pitch mean 158.1 Hz, std 90.9 Hz, max 2,022 Hz.
- **What was wrong:** `extract_sup_data.py` ran with its config's default pitch range (65-2093 Hz).
  The tutorial trains with 30-512 Hz. The extract caches one pitch file per clip, and the training
  LOADS cached files before computing anything (`TTSDataset`, NeMo v3.0.0), so `model.pitch_fmax=512`
  would have had no effect: the model would have trained on pitch with octave errors.
- **Whose mistake:** mine; the stage was written without reading how the cache is used.
- **Fix:** `PITCH_FMIN, PITCH_FMAX = 30, 512` feed both the extract and the training; the extract
  writes `sup_data_f30_512/` and `stats_f30_512.json`; the first extract stays on the volume, unread.
  One test holds the two stages to one range and one directory.
- **Cost:** CPU, 122 s. Log: `results/ft3-data-first-extract.txt`; run file `modal-data-20260927-135825.json`.

## 6. `::data`, third attempt: the statistics the training used

| | First extract | This extract | Tutorial |
|---|---|---|---|
| Range | 65-2093 Hz | 30-512 Hz | 30-512 Hz |
| Mean | 158.1 Hz | 162.3 Hz | 152.3 Hz |
| Std | 90.9 Hz | 75.1 Hz | 64.0 Hz |
| Min / max | 65.4 / 2,021.7 Hz | 30.0 / 511.5 Hz | — |

- My prediction (about 152 / 64) was wrong. The minimum sits on the 30 Hz floor, and librosa warns
  that 30 Hz is too low for a 1,024-sample frame: low-end tracking errors remain. The tutorial's
  range was kept.
- **Cost:** CPU, 118 s. Log: `results/ft3-data.txt`; run file `modal-data-20260927-140907.json`.

## 7. `::train`: trained, then failed to package (fixed in 3e6be7f)

- **Run:** `fastpitch-12bc015-20260927-140956`, A10, 1,000 steps (250 epochs of 4 batches),
  14:09:56 to about 14:24:20.
- **Validation loss:** best 1.7385 at step 900; step 1,000 not in the top 3. Two validation clips:
  the ranking is mostly noise. The last checkpoint is used, as in the tutorial.
- **Failure:** `RuntimeError: no *last.ckpt under .../FastPitch/checkpoints`.
- **Cause:** NeMo's exp_manager writes `FastPitch/<date_time>/checkpoints/`; the code looked in
  `FastPitch/checkpoints/`. The layout had not been checked against a real run.
- **State:** the dated directory is on the volume (`modal volume ls`, 2026-09-27). No `.nemo` file yet.
- **Fix:** `_last_ckpt` searches below the model's directory; `::package --name <run>`
  packages a finished run on a CPU; both training stages commit the volume before packaging and
  refuse to overwrite a model file.
- **Cost:** about $0.26 (14 min 15 s between the run's name stamp and the last checkpoint, at the
  A10 rate; the run record was lost with the failure). Log: `results/ft3-train-packaging-failed.txt`.

## 8. `::package`: saved the model, then failed (fixed in 84cf350)

- **Result:** `models/fastpitch_9017.nemo` on the volume, 178.4 MiB (`modal volume ls`).
- **Failure:** `FileNotFoundError: 'nvidia-smi'`, in the line that builds the run record, AFTER the
  model was saved and the volume committed.
- **Cause:** `_machine()` was written for GPU functions and reused on a CPU one. Mine.
- **Lost:** the run record (`modal-package-*.json`) was never written. The packaged checkpoint is the
  run's `*last.ckpt` by construction; its file name was not recorded.
- **Fix:** `_machine()` returns "Modal (no GPU)" when `nvidia-smi` is missing; a test holds it.
- **Cost:** CPU. Log: `results/ft3-package.txt`.

## 9. `::vocoder`, first attempt: failed on the first clip (fixed in 84cf350)

- **Failure:** `RuntimeError: The size of tensor a (90) must match the size of tensor b (89)`.
- **Cause:** the tutorial builds the alignment prior from the length the preprocessor STATES; in
  NeMo 3.0.0 the spectrogram has one frame more than that (90 and 89 for 22,784 samples), and the
  aligner refuses the mismatch. Every clip fails. Reproduced on the laptop with the base voice.
- **Fix:** the prior is built for the spectrogram's shape (`mels.prior_frames`), as the training
  itself does (`TTSDataset`). The mel loop moved from the Modal app to `ft/src/ft3/mels.py`, which
  runs on the laptop.
- **Cost:** about a minute of A10 time. Log: `results/ft3-vocoder-prior-failed.txt`.

## 10. The review: why five failures, and the dry run

Asked for after entry 9. The cause of four of the five failures (entries 5, 7, 8, 9): the Modal app
was written from NeMo's documentation and never executed before the hand-over; its tests checked
constants, not behaviour. Entry 4 (clang) was Modal-specific.

The vocoder stage was then run end to end on the laptop's CPU (base voice, two dev renders as
stand-in clips): mels, `hifigan_finetune.py` with the stage's own overrides for 6 steps under both
the tutorial's `ddp` strategy and `auto`, the last checkpoint found, packaged, restored, audio
rendered. Not covered by it: the GPU, the real clips, the fine-tuned FastPitch.

Also added: the stage counts the clips `hifigan.yaml` would prune (under 0.75 s for training, 3 s
for validation) and stops before loading a model if a whole split would go.

Learned from the dry run: HiFi-GAN's step counter advances 2 per batch (generator and
discriminator), so `max_steps=1000` is 500 generator updates. The tutorial's recipe, unchanged.

Rule from here on: a new Modal stage is run on the laptop first, and the hand-over says what ran.

## 11. `::vocoder`, second attempt: trained and packaged

- **Run:** `hifigan-84cf350-20260927-150730`, A10, 1,000 steps (500 generator updates), 3 batches
  per epoch, stopped in epoch 166.
- **Validation:** both clips kept (6.28 s and 4.74 s). Loss 0.4317 at step 960; best 0.4236.
- **Packaged from:** `HifiGan--val_loss=0.4317-epoch=167-last.ckpt` -> `models/hifigan_9017.nemo`.
- **Cost:** 644 s in the function, 669 s end to end, about $0.204. Log: `results/ft3-vocoder.txt`;
  run file `modal-vocoder-20260927-151839.json`.
- **Noticed afterwards:** the manifest's `text`, which both trainings read, is lowercase without
  punctuation; the punctuated text is in `text_normalized`. Hypothesis, to be measured by the dev
  sweep: the fine-tuned voice pauses less at punctuation than the base voice.

## 12. The download: two hours for a file that was two files

- **Command given:** `modal volume get ft3 models data/ft3/models` (printed by the vocoder stage and
  written in the README). Mine, and never run before.
- **What happened:** the destination folder did not exist, so both models were written to ONE file
  named `models`: 339,210,240 bytes, the vocoder's size, beginning with FastPitch's archive. The
  first attempt ran 1 h 48 min at 145 to 406 kB/s and was aborted at 86%; `--force` then finished.
- **Fix:** the mixed file removed, the folder made, one `modal volume get` per file. Both checked
  by size and by listing the archive: `fastpitch_9017.nemo` 187,043,840 bytes, `hifigan_9017.nemo`
  339,210,240 bytes. Their SHA-256 are in `ft/src/ft3/pr012.py`. The command is corrected in the
  stage's hint and in the README, and a test holds its form.
- **Seen in the archive:** the pronunciation dictionary and the heteronyms are packed inside the
  `.nemo`, so the model loads on the laptop without the volume's paths.

## 13. The dev sweep on the fine-tuned voice

Laptop CPU, 168 s, PR-009's four dev sentences, the settings fixed on the base voice. Before it,
the `.nemo`-path branch of `synth.load`, never run, was run with the cached base models as files.

| PR-009's counts | bar | base voice | fine-tuned 9017 |
|---|---|---:|---:|
| S1 offset, rate, pauses in order | all | 23 of 24 | 22 of 24 |
| S2 range in order | 28 of 32 | 5 of 8 | 0 of 8 |
| S3 rate and pauses within 10% | all | 93 of 96 | 87 of 96 |
| S4 pitch offset | 90% | 33 of 48 | 23 of 48 |
| S5 pitch range | 80% | 31 of 48 | 11 of 48 |

- Rate: 48 of 48 for both.
- The fine-tuned voice's range moves before any range is asked for: 1.27, 1.03, 0.80, 1.02 at
  depression 0, where the base voice gave 1.06, 0.94, 1.05, 1.04.
- Its neutral renders' F0 tracks hold 4 to 8 jumps of more than 40% between neighbouring frames.
- **The punctuation hypothesis of entry 11 was not supported:** the fine-tuned voice has 3, 3, 3
  and 2 pauses on the four sentences, the base voice 3, 2, 2 and 2.
- Why the pitch is worse was not measured. Hypotheses only: tracking errors in the training pitch;
  too little data and too few steps; the vocoder's 500 generator updates.
- Log: `results/ft3-sweep-9017-dev.txt`; run file `results/ft3/sweep-fastpitch_9017-dev.json`.

## 14. PR-012 registered

Two voices, eight new sentences, PR-009's bars unchanged, the settings frozen, the four model
files held by SHA-256. Predicted: not supported, all five criteria fail. The run's code was run
first on the dev sentences (`python -m src.ft3.pr012 dry`). The entry is in
`docs/preregistration.md`.

## 15. The PR-012 run

Laptop CPU, 490 s, commit 89edd05, clean tree, first and only run. $0.

| 32 sweeps | bar | pooled | base voice | fine-tuned 9017 |
|---|---|---:|---:|---:|
| S1 offset, rate, pauses in order | all | 92 of 96 | 47 of 48 | 45 of 48 |
| S2 range in order | 28 of 32 | 14 | 11 of 16 | 3 of 16 |
| S3 rate and pauses within 10% | all | 357 of 384 | 175 of 192 | 182 of 192 |
| S4 pitch offset | 90% | 109 of 192 (56.8%) | 67 of 96 | 42 of 96 |
| S5 pitch range | 80% | 115 of 192 (59.9%) | 78 of 96 | 37 of 96 |

- **Predictions:** two wrong, the rest right. Wrong: S5 came out at 59.9%, above the predicted 35 to
  55% (the base voice did better on these eight sentences than on the four dev ones); and "every
  pause miss is in mania" -- 8 of the 27 pause misses are in depression, all on the base voice's
  u7 and u8.
- **Not explained:** the base voice on u8, where one pause was found; in mania at 40 and 60 the
  measured pause is 0.00, and in depression it overshoots (5.45 for 4.00).
- **The report file:** NeMo printed six lines to standard output above the report, with local
  paths in them; `runs/ft3/pr012-report.md` leaves them out and is otherwise verbatim.
- Evidence: `runs/ft3/`; the ledger's eight FT-3 rows; the entry's Results in
  `docs/preregistration.md`.

## Money

Modal, FT-3: about $0.26 (FastPitch, timed from its log) + $0.204 (HiFi-GAN) + at most $0.04
(the failed vocoder run) + the CPU stages: about $0.52. With FT-2's $0.25, about $0.77 of the
$10 cap for the two milestones.
