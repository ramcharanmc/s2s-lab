# ft/ — the fine-tunes, on the Lightning-Hydra template

s2s-lab's training runs (v2 plan, page 17 §7): **FT-2**, a LoRA on Whisper for disfluent speech,
and **FT-3**, a TTS backend for the severity render. FT-1 (Moshi + LoRA) was trained with kyutai's
own script and lives in `vast/`. This is a separate uv project inside the repo, built from the
[Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template) (MIT; the uv
variant by ibatalov): `src/utils/` and the `configs/` skeleton are the template's, unchanged.
Everything measured goes through s2s-lab's own code -- `s2slab.wer`, the eval set's readers --
so the numbers here are the ladder's numbers.

## Rules, from FT-1 (page 20 §22)

1. **Look at the trained model on dev before registering a test.** `src/eval.py` is that look.
2. **Keep every checkpoint.** `callbacks/default.yaml` keeps all of them; each holds the adapter only.
3. **A dollar cap on every paid run**, in the run's own config; the ledger gets a row either way.
4. The frozen test sets are never read here: the registered test is s2s-lab's bench on the exported
   model, so it is measured exactly as the rungs are.

## Set-up (laptop, CPU: tests and one-batch runs)

The env goes on WSL's own disk, not under `/mnt/c` (20x slower imports):

    cd ft
    UV_PROJECT_ENVIRONMENT=~/.venvs/s2slab-ft uv sync
    UV_PROJECT_ENVIRONMENT=~/.venvs/s2slab-ft uv run pytest -q

`uv sync` installs CPU torch, Lightning, transformers, peft and the parent package `s2slab`
(editable, from `..`). The tests fetch `openai/whisper-tiny.en`'s tokenizer files (a few MB) on
the first run; no model weights, no dataset.

## FT-2 — Whisper LoRA, verbatim transcription of disfluent speech

**Question.** `whisper-small.en` -- R2's slow tier (`r2-gpu`), and the recognizer of `r0-gpu-bigger`
-- cleans speech up: on the eval set's disfluent cohort it scores 9.9% raw against 7.6% normalized
(PR-004's run), and the gap is the fillers and repetitions it drops. Does a LoRA trained on verbatim transcripts make it write down what was said, without
hurting the other cohorts?

**Data.** [amaai-lab/DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)
(Apache-2.0), `transcript_a` = verbatim, revision `b7da294` (the eval set's pin). One speaker
re-performing Switchboard utterances: the eval set's disfluent cohort is 50 clips of its TEST
split, so a result on that cohort is **same-speaker adaptation**, and is reported as such. The
test split is never read here (`src/data/disfluencyspeech_datamodule.py`).

**Result (PR-011, 2026-09-26, `runs/ft2/`): supported.** On the eval set's disfluent cohort raw WER
fell from 9.9% to 2.3% and the LoRA kept 29 of 29 fillers (base: 1); on the other 150 clips
normalized WER fell 2.3 points where a rise of 1.0 was allowed -- most of it on EdAcc, whose
references are verbatim too. Same-speaker adaptation on the disfluent cohort, as registered;
`docs/preregistration.md` has the entry, the predictions scored and what the registration missed.

**Recipe.** `configs/experiment/ft2_whisper_verbatim.yaml`: PEFT LoRA r=32, α=64 on q and v
(PEFT's own Whisper recipe, ~1.5% of the parameters), AdamW 2e-4 with 50 warm-up steps and linear
decay, batch 16, 900 steps (≈3.2 epochs), validation on 64 clips every 150 steps.

    uv run src/train.py experiment=ft2_whisper_verbatim trainer=cpu debug=fast   # one batch, laptop
    uv run src/train.py experiment=ft2_whisper_verbatim                          # the run (GPU)
    uv run src/eval.py adapter_dir=null                                          # dev look: base
    uv run src/eval.py adapter_dir=logs/train/runs/<run>/adapters/step-000900    # dev look: LoRA
    uv run src/export.py --adapter logs/train/runs/<run>/adapters/step-000900 --out ../data/ft/ft2-export

The export merges the adapter into the base and converts to CTranslate2, the format
faster-whisper loads; s2s-lab's bench then runs it beside the unmodified `small.en`.

**Where it runs: Modal, not Vast.ai -- the decision (2026-09-26).** The plan's week 7 says
"FT-1 / FT-2 / FT-3 on Vast spot", and its platform rule (page 17 §5) says: GPU busy for more than
an hour -> Vast; bursty -> Modal. FT-1 was 11 hours of training and went to Vast. FT-2 is about
900 steps of whisper-small with a LoRA, some 15-20 minutes of GPU, and by the plan's own rule
that is Modal's case. What decided it:

1. *Vast's overhead is fixed, and it swamps a short job.* FT-1's machine ran 12.3 h for 11.4 h of
   training; the rest was renting, ssh, installing, the dataset download (rate-limited twice) and
   the copy-off, plus the two things that broke (`CUDA_VISIBLE_DEVICES`, the ssh proxy). For a
   20-minute job that overhead is two to three times the training, in money and in attention.
2. *Cost is a wash; reproducibility is not.* Modal A10 at \$1.10/h for about 0.3 h is about
   \$0.35 (plus a one-off CPU image build); a Vast 4090 at about \$0.35/h for 0.3 h of training
   and 0.75 h of set-up is much the same. But the Modal image is pinned in code and a test holds it
   to `ft/uv.lock`, the dataset sits on a volume fetched once on a CPU, every run is one command
   with a hard dollar cap, and there is no machine to destroy and no key of ours on a rented box.
3. *Budgets.* Vast stands at \$24.29 against the plan's \$5.86; Modal at \$8.04 against \$36.93.
   The small jobs go where the headroom is.

The case for Vast is the portfolio story -- week 7 was to show the training platform -- and the
`vast/` kit's fixes from FT-1 (self-destroy read from PID 1, the 15 h cap) are untested on a real
machine. Both wait for a job that needs Vast: this project runs anywhere `uv sync` does, so FT-2
on a rented 4090 would be `uv sync`, `uv run src/train.py experiment=ft2_whisper_verbatim`, copy
`logs/` off. On Modal the same `src/train.py` command runs as a subprocess, on the volume `ft2`,
with a dollar cap on every function:

    modal run modal_app/ft2_train.py::dev                                   # 30 steps on the real data, ~$0.10
    modal run modal_app/ft2_train.py::train                                 # the experiment, ~$0.40
    modal run modal_app/ft2_train.py::evaluate                              # dev look: base, all 250 val clips
    modal run modal_app/ft2_train.py::evaluate --adapter runs/<run>/adapters/step-000900
    modal run modal_app/ft2_train.py::export --adapter runs/<run>/adapters/step-000900
    modal volume get ft2 runs/<run> results/ft2/                            # the run's files, back here

## FT-3 — a TTS with native pitch and duration control

The severity render (`s2slab.severity`, P2c) edits Kokoro's speech in Praat after the fact; PR-009
found its pitch offset leaning upward. FT-3 gives the same four targets to a TTS that takes pitch
and duration as **inputs**: NVIDIA FastPitch + HiFi-GAN (NeMo; Apache-2.0 code, the LJSpeech
checkpoints CC-BY-4.0), fine-tuned to a Hi-Fi TTS speaker (CC BY 4.0) as the plan's FT-3 asks.

**How the targets become inputs** (`src/ft3/control.py`, pure functions; `src/ft3/synth.py`, the
model): FastPitch predicts a duration in frames and a pitch value (normalized: (Hz − mean) / std)
per text token, and its `forward()` takes both back as inputs. Speech tokens' durations are
divided by the rate and their pitch scaled about the speaker's mean and shifted by offset / std;
a *pause* — a run of punctuation and space tokens predicted to last 150 ms or more, the Praat
render's own rule — has its durations multiplied by the pause scale. The pauses' positions in the
output come from the frame counts, so `severity.achieved` measures a FastPitch render exactly as it
measured the Praat one. The neutral reference goes through the same forward pass with the
model's own predictions.

**Environment.** NeMo cannot share `ft/`'s env (it pins `hydra-core` ≤ 1.3.2, `omegaconf` ≤ 2.3 and
the `lightning` package), so it has its own, from `requirements-ft3.txt` (the commands are in
that file); CPU is enough for synthesis, and the control tests run in `ft/`'s env without NeMo.

    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.synth "The train leaves at nine. I bought a ticket." --condition depression --score 15

writes the neutral and the controlled render under `results/ft3/dev/` and prints the four measures
against their targets.

**The dev sweep** (`src/ft3/sweep.py`, the base LJSpeech voice, the four dev sentences of PR-009,
both conditions, six scores; 172 s on the CPU) produces `severity.run`'s exact results shape, so
`severity.report` prints it and PR-009's `severity.criteria` scores it unchanged:

    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.sweep --sentences dev [--listen]

Dev result, with the two decisions fixed in `control.py` (the range scaled about the neutral
render's measured median; a range gain of 1.2, because the model answers a per-token scale with
about 0.83 of it): rate within 10% on 48 of 48 checks, pauses on 45 of 48 (the misses all at
YMRS 60, ×0.30), the pitch offset within 3 Hz or 10% on 33 of 48, the range within 10% on 31 of 48;
28 of 32 feature sweeps in order. Rate and pauses are as exact as the Praat render's; the pitch is
not -- FastPitch's pitch is a per-token input the model interprets, and its answer is damped and
noisy on the range.

**The fine-tuned voice on the same sentences** (2026-09-27): rate 48 of 48; pauses within 10% on 39
of 48 (every miss in mania); the pitch offset on 23 of 48; the range on 11 of 48, and in order on
none of its 8 sweeps. Its range moves before any range is asked for (0.80 to 1.27 at depression
0, where the base voice gave 0.94 to 1.06). The cause was not measured. Nothing was tuned on it.

**PR-012** (`src/ft3/pr012.py`, `docs/preregistration.md`) registers PR-009's five criteria on
both voices over eight new sentences, with the prediction that all five fail:

    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.pr012 dry                     # dev sentences, no lock
    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.pr012 test --confirm PR-012   # once

**Result (2026-09-27, `runs/ft3/`): not supported, and all five criteria failed, as predicted.**

| 32 sweeps | bar | pooled | base voice | fine-tuned 9017 | PR-009 (Praat on Kokoro) |
|---|---|---:|---:|---:|---:|
| S1 offset, rate, pauses in order | all | 92 of 96 | 47 of 48 | 45 of 48 | 96 of 96 |
| S2 range in order | 28 of 32 | 14 | 11 of 16 | 3 of 16 | 29 |
| S3 rate and pauses within 10% | all | 357 of 384 | 175 of 192 | 182 of 192 | 384 of 384 |
| S4 pitch offset within 3 Hz or 10% | 90% | 56.8% | 67 of 96 | 42 of 96 | 88.0% |
| S5 pitch range within 10% | 80% | 59.9% | 78 of 96 | 37 of 96 | 92.2% |

Rate is exact (192 of 192): a duration is an input the model obeys. Pitch is an input the model
interprets: the base voice reached the offset on 70% of its checks and the range on 81%, the
5-minute fine-tune on 44% and 39%. The two checkpoints differ in speaker as well as in training,
and the cause was not measured. PR-009's Praat edit came closer to page 09's targets on every
count, on other voices and sentences.

What happened on the way, run by run and failure by failure: `docs/ft3-log.md`.

**Training** (`modal_app/ft3_train.py`, the tutorial's recipe on an A10): `::data` (the 5-minute
Hi-Fi TTS speaker 9017, CPU), `::train` (FastPitch, 1,000 steps), `::vocoder` (HiFi-GAN on the
fine-tuned model's mels, 1,000 steps). The two `.nemo` files come back one file a call, into a
folder that exists (given the folder, `modal volume get` wrote both models into ONE file):

    mkdir -p data/ft3/models && for m in fastpitch hifigan; do modal volume get ft3 models/${m}_9017.nemo data/ft3/models/; done

`--fastpitch`/`--hifigan` point `synth` and `sweep` at them. The Modal stages' code that can run
without a GPU (`src/ft3/mels.py`) is run on the laptop first.

## Licences

Template MIT (`LICENSE-template`). Whisper weights MIT. DisfluencySpeech Apache-2.0. PEFT,
transformers, Lightning, Hydra: Apache-2.0 / MIT.
