# FT-2 run log (M24)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Written on 2026-09-27, after the milestone, from the record: the commits, `docs/preregistration.md`
(PR-011), `docs/cost-ledger.md`, `runs/ft2/`. Where an event is known only from the working session
and no file holds it, the entry says so. Commit times are CDT; all of it happened on 2026-09-26.
The same log for FT-1 is `docs/ft1-log.md`, for FT-3 `docs/ft3-log.md`.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 17:01 | tests, laptop | `ft/`: the project, the model, the export | `ft/` (45 files) | 1cec361 |
| 2 | 18:19 | — | the Modal app, and Modal chosen over Vast | `modal_app/ft2_train.py`, tests, README | 0b89a54 |
| 3 | after 18:19 | `::dev`, A10 | **failed** in its first second | `configs/trainer/default.yaml` | df227b0 |
| 4 | after 18:28 | `::dev`, A10 | 30 steps: raw WER 7.9% to 4.2% | — | — |
| 5 | 18:50 | — | the criterion and the bench | `src/s2slab/ft2.py`, `::bench`, eval on the GPU | f37d0df |
| 6 | 18:52 | `::train`, A10 | 900 steps, 545 s | — | — |
| 7 | evening | `::evaluate`, base, A10 | dev look: 10.9% raw, 8 of 176 fillers | — | — |
| 8 | evening | `::evaluate --adapter`, A10 | **failed**: no adapter at that path | the module's output folder | 65c833c |
| 9 | evening | `::evaluate --adapter`, A10 | dev look: 2.1% raw, 174 of 176 fillers | — | — |
| 10 | evening | `::export`, CPU | merged, converted to CTranslate2 | — | — |
| 11 | evening | `::bench --limit 8`, T4 | smoke test, base model only | — | — |
| 12 | 22:01 | — | PR-011 registered | the entry; S2's bar raised | 65c833c |
| 13 | 22:03 | `::bench --confirm PR-011`, T4 | **supported** | — | — |
| 14 | 22:16 | — | results | evidence and text | 29c2285 |

## 1. The project (1cec361)

`ft/` is its own uv project on the Lightning-Hydra template (MIT, credited in
`ft/LICENSE-template`), with its own environment. whisper-small.en with a PEFT LoRA, rank 32 on the
q and v projections: 3,538,944 of 245,273,088 parameters, 1.44%. Data: DisfluencySpeech's verbatim
`transcript_a`; the test split cannot be reached from the code. 18 tests of its own, on a fake
dataset and a tiny random Whisper.

**From the session, not in a file.** Three of those tests failed when first written and were fixed
before the commit: the merge test saved its base model from inside PEFT's wrapper; transformers 5
no longer writes `preprocessor_config.json` with the model, so the export saves the feature
extractor and the tokenizer on their own; and `eval.yaml` listed `_self_` in the wrong place.

## 2. The Modal app, and the platform decision (0b89a54)

- **The app:** `ft/`'s own commands run as subprocesses in an A10 image pinned to `ft/uv.lock`; a
  test holds the pins. The dataset is fetched once, on a CPU, onto the volume `ft2`. Every function
  has a dollar cap.
- **The decision:** the plan put all three fine-tunes on Vast. FT-2 went to Modal by the plan's own
  rule (under an hour of GPU is bursty). Vast's fixed overhead on FT-1 was about an hour; the cost
  was the same either way; Vast stood at $24.29 against $5.86. Full text: `ft/README.md`, "Where it
  runs".
- **Not run before the hand-over:** the app. Its first execution was entry 3.

## 3. The first paid run failed (fixed in df227b0)

- **Failure:** Hydra refused the override `trainer.max_time`.
- **Cause:** the key was not declared in the trainer's config, and Hydra's strict mode accepts only
  declared keys. Mine: the dollar cap was added to the Modal app without running the command.
- **Fix:** the key declared, `max_time: null`; a test holds it.
- **Cost:** about $0.01. The image was built and the dataset fetched, both on CPU.

## 4. The smoke run

30 steps of the real experiment on the real data, 8 validation clips: raw WER 7.9% to 4.2%,
fillers kept 40% to 100%. 45 s, $0.015. Dev data, not evidence.

**From the session:** the run's summary showed the loss as `nan`. The validation loss was not being
written with each validation; entry 5 added it.

## 5. The criterion and the bench (f37d0df)

- **`src/s2slab/ft2.py`:** PR-011's three criteria, tested on made-up transcripts.
- **`::bench`:** `small.en` and the exported model through the rungs' own recognizer class, in one
  T4 container. The exported model meets the eval set only under `--confirm PR-011`.
- **Two fixes in the same commit:** the dev look ran on a CPU trainer (`eval.yaml` composed one),
  now on the GPU; the validation loss is kept with each validation's transcripts.

## 6. The training

| | |
|---|---|
| Run | `train-f37d0df-20260926-235222`, Modal A10 |
| Steps | 900, batch 16, AdamW 2e-4, bf16 |
| Time and cost | 545 s, $0.168 |
| Validation, 64 clips | raw WER 2.4% at step 150, 2.0% at 900 |
| Fillers kept | 100% at every validation |
| Kept | all six adapters, one every 150 steps |

It had learned the target by the first validation.

## 7 to 9. The dev look, and a wrong path

| 250 validation clips | base `small.en` | + LoRA, step 900 |
|---|---:|---:|
| raw WER | 10.9% | 2.1% |
| normalized WER | 7.8% | 2.1% |
| fillers kept, of 176 | 8 | 174 |

- **Entry 8, the failure:** the adapter was not where the command looked. The module saved under the
  logger's folder, `csv/version_0/adapters/`, not at the run's root. It stopped before loading
  anything; about $0.005.
- **Cause:** the module wrote to `trainer.log_dir`, to which the CSV logger adds its own folders.
  Mine.
- **Entry 9:** the same command with the real path. The run's adapters stay where they were saved,
  which is why the export's name contains `csv_version_0`.
- **Fix, for later runs** (in 65c833c): the module writes to the run's root.

Each dev look: about 45 s, $0.014.

## 10 and 11. The export and the smoke test

- **Export:** the step-900 adapter merged into `small.en` and converted to CTranslate2 float16, the
  format the rungs load. CPU, 44 s. `ct2/model.bin` SHA-256 `b31833fe...a51cff`.
- **Bench smoke test:** the base model alone on 8 clean clips, raw WER 5.1%. $0.002. The exported
  model had still met none of the eval set's clips.

## 12. PR-011 registered (65c833c)

Registered after the dev look, with the adapter fixed at step 900 and the export's SHA-256 in the
entry. S2's bar was raised from 80% to 90% before the run, because the dev look had shown 98.9%.

**From the session:** the entry was first inserted near the top of the file, by an anchor that
matched PR-000, and was moved to the end before the commit.

## 13. The registered run

Modal T4, one container, 200 clips, 67 s, $0.012, first and only run.

| Criterion | Result | |
|---|---|---|
| S1 raw WER, disfluent cohort | 9.9% to 2.3%; -7.6 points, interval -9.6 to -5.7 | pass |
| S2 fillers kept | 29 of 29 (the base kept 1) | pass |
| S3 normalized WER, the other 150 clips | 10.4% to 8.0%; -2.3 points, where +1.0 was allowed | pass |

**Supported.** The base model reproduced PR-004's numbers.

## 14. Results (29c2285)

- **Two deviations recorded.** The entry said the references hold 28 fillers; the code counts 29.
  And the entry called the other 150 clips "fluent speech"; EdAcc's references are verbatim
  conversation, so S3 also measured the verbatim target.
- **Predictions:** S1, S2 and the timing right. S3 wrong, in the good direction, for a reason the
  entry should have checked: what the guardrail cohorts' references contain.
- **From the session:** the volume's files were brought back in two attempts; the first paste
  showed an earlier command's output, and the dev looks' files were one folder deeper than the
  command expected.
- Evidence: `runs/ft2/` (20 files). Nine ledger rows.

## What the milestone taught

- **The dev look before the registration worked.** It set S2's bar and the S1 prediction; both held.
- **Check what a guardrail's data contains** before writing what it guards.
- **Both failed runs were commands never run before the hand-over.** They cost one and a half
  cents together. The same habit cost FT-3 five runs (`docs/ft3-log.md`).
- **The same person's held-out clips are not other speakers.** The entry said so; the result is
  same-speaker adaptation.

## Money

| | |
|---|---:|
| The training | $0.168 |
| The smoke run, two dev looks, the export, the bench smoke test, the test | $0.067 |
| Two failed runs | about $0.015 |
| **The milestone, all on Modal** | **about $0.25** |
