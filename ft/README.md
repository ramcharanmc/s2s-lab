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

**Question.** R1's recognizer (`whisper-small.en`) cleans speech up: on the eval set's disfluent
cohort it scores 12.1% raw against 9.7% normalized, and the gap is the fillers and repetitions it
drops. Does a LoRA trained on verbatim transcripts make it write down what was said, without
hurting the other cohorts?

**Data.** [amaai-lab/DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech)
(Apache-2.0), `transcript_a` = verbatim, revision `b7da294` (the eval set's pin). One speaker
re-performing Switchboard utterances: the eval set's disfluent cohort is 50 clips of its TEST
split, so a result on that cohort is **same-speaker adaptation**, and is reported as such. The
test split is never read here (`src/data/disfluencyspeech_datamodule.py`).

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

**Where it runs.** The GPU run goes to Modal (`modal_app/`, next commit): each run is well under
an hour, and Vast's set-up overhead was most of an hour on FT-1.

## FT-3 — a TTS with native pitch and duration control

Planned: NVIDIA FastPitch + HiFi-GAN (Apache-2.0 code, CC-BY-4.0 checkpoint), fine-tuned to a
Hi-Fi TTS speaker (CC BY 4.0), as the synthesis backend P2c's severity render asks for. Not started.

## Licences

Template MIT (`LICENSE-template`). Whisper weights MIT. DisfluencySpeech Apache-2.0. PEFT,
transformers, Lightning, Hydra: Apache-2.0 / MIT.
