# runs/ — committed results

Every file here is the complete output of one full run: the numbers on the
Notion results pages come from these files, and anyone can re-score them.

**Rule:** a run lands here only if it covers the whole frozen set it names.
Smoke tests, `--limit` runs and reply WAVs go to `results/`, which is ignored.

| File | What | Commit | Machine | Notion |
|---|---|---|---|---|
| `turn-taking/sweep.json` | T0–T3 endpointing sweep, 250 per cohort, seed 1000 | `6a0741e` | laptop CPU | page 18 |
| `r0/evalset-laptop-cpu.json` | R0 over the 200-clip eval set | `eb57bf9` | laptop CPU (WSL, 16 threads) | page 20 §5 |
| `r0-gpu/evalset-modal-t4.json` | R0 (fp16) over the 200-clip eval set | `0af142a` | Modal Tesla T4 | page 20 §5 |
| `r0-gpu/evalset-modal-t4-pr002.json` | R0 re-run for PR-002, same container as the row below | `b107927` | Modal Tesla T4 | page 20 §9 |
| `r1-gpu/evalset-modal-t4-pr002.json` | R1 streaming over the 200-clip eval set, the PR-002 run | `b107927` | Modal Tesla T4 | page 20 §9 |
| `bargein/pr003-70ms.json` | PR-003, the registered run: seven barge-in policies on the frozen 20, recognizer time charged at 70 ms a call | `19d3ad7` | laptop CPU (WSL) | page 20 §11, page 18 |
| `bargein/pr003-70ms-report.md` | the same run's complete printed output, with the per-item table | `19d3ad7` | laptop CPU (WSL) | page 20 §11 |
| `bargein/pr003-measured.json` | PR-003, descriptive: the same set with recognizer time as measured on the laptop; not deterministic | `19d3ad7` | laptop CPU (WSL) | page 20 §11 |
| `r1-gpu/evalset-modal-t4-pr004.json` | R1 re-run for PR-004, same container as the two rows below; 17% slower than its PR-002 run, same words | `91048b3` | Modal Tesla T4 | page 20 §12 |
| `r2-gpu/evalset-modal-t4-pr004.json` | R2 with small.en as the slow tier, over the 200-clip eval set; described, not tested | `91048b3` | Modal Tesla T4 | page 20 §12 |
| `r2-gpu-turbo/evalset-modal-t4-pr004.json` | R2 with large-v3-turbo as the slow tier, the PR-004 run | `91048b3` | Modal Tesla T4 | page 20 §12 |
| `prosody/pr005.json` | PR-005, the registered run: the affect side-channel on the 39 locked prosody pairs, four conditions | `4b50017` | laptop CPU (WSL) | page 20 §14 |
| `prosody/pr005-report.md` | the same run's complete printed output, with the per-pair table | `4b50017` | laptop CPU (WSL) | page 20 §14 |
| `prosody/pr006.json` | PR-006, the registered run: R3 (Qwen2-Audio) hears the 39 locked prosody pairs, `upset` wording | `5c1f3aa` | Modal NVIDIA A10 | page 20 §15 |
| `prosody/pr006-report.md` | the same run's complete printed output, with both per-pair tables | `5c1f3aa` | Modal NVIDIA A10 | page 20 §15 |
| `prosody/ownvoice-r3.json` | the own-voice mini set (dev-only, not registered): R3 on the author's five pairs, every take windowed and as recorded, every wording | `4c5905d-dirty` | Modal NVIDIA A10 | page 20 §16 |
| `prosody/ownvoice-r3-report.md` | the same run's complete printed output, both per-pair tables | `4c5905d-dirty` | Modal NVIDIA A10 | page 20 §16 |
| `prosody/pr007.json` | PR-007, the registered run: the absolute tagger against the per-caller baseline on the 38 fresh pairs, every tag and probability | `a591503` | laptop CPU | page 20 §17 |
| `prosody/pr007-report.md` | the same run's complete printed output, with the per-pair table | `a591503` | laptop CPU | page 20 §17 |

## What an eval-set run holds

Provenance first: `rung`, `description`, `commit`, `machine`, `ran`, `load_ms`,
and `evalset` (build date, seed, turn count). Then `turns`, one entry per clip:
`cohort`, `transcript`, `reply_text`, `stages_ms`, `untimed_ms`, `total_ms`,
`reply_s`, `leading_silence_ms`.

## What a prosody run holds

`run` (the heading the side-channel printed), `question` and `system` (the
exact prompt), `seed`, `n_shuffles`; `pairs` (the 39 pairs as in
`probes/prosody/pairs.tsv`); `transcripts` (R1's recognizer, per clip); `tags`
(the tagger's word, per clip); `p_yes` (per clip, the LLM's p(Yes) under no
tag and under each of the four words, plus the no-tag prompt's probability
mass on Yes-or-No); `result` (per condition and emotion: pairs answered right,
ranked right, pairs); and `shuffled_deal_counts` (per emotion, the 1000 deals'
counts, from which the permutation p is the share at or above the tagger's
count). No audio and no reference text from CREMA-D is stored; the transcripts
are a model's output.

Re-score without a GPU or any model (needs the eval set's manifest, not its audio):

    PYTHONPATH=src python -m s2slab.evalrun --score runs/r0-gpu/evalset-modal-t4.json

## What a barge-in run holds

`run` (the heading the scorer printed: set, registration, how recognizer time
was counted); `outcomes`, per policy one entry per item: `item`, `cell`, `kind`,
`should_stop`, `stop_s`, `reason`, `latency_ms`, `overrun_words`, `correct`;
and `partials`, per item every transcript the policies were shown, as
`[delivered at (s), text, final, start of the sound (s), no-speech score]`.
The transcripts are a model's, of the scripted lines in
`probes/bargein/script.tsv`; no audio is stored here.

Re-count without the recordings or any model:

    python -c "import json; d = json.load(open('runs/bargein/pr003-70ms.json', encoding='utf-8')); print({p: sum(not o['correct'] for o in os) for p, os in d['outcomes'].items()})"
## Sources

`transcript` is a model's output from eval-set audio; no source audio or
reference text is stored here. Datasets, pinned revisions and licences are in
`evalset/README.md`.


## R3 as a rung — written before the run (2026-09-23)

The R3 row of the one table comes from `modal run modal_app/r3_probe.py::evalset
--rung r0-gpu,r1-gpu,r3-gpu`: one A10 container, the frozen 200-clip eval set,
R0 and R1 run beside R3 so the row is read against them on that day's
hardware (M13's rule). No registration -- this is the plan's week-4 row, not a
hypothesis -- but the expectations are written down first, to be scored:

- R3 total p50 **1.5–2.5 s** on the A10 (a 7B prefill of a 3–15 s clip, about
  40 new tokens, then Kokoro); first audio = total, since R3 is batch. R0-gpu
  on the same A10 about 0.5 s, R1-gpu about 0.4 s to first audio: R3 waits
  **3–5× R0** -- the "inherits R0's input-side latency, and then some" number.
- R3's replies longer than the 0.5B model's under the same 60-token cap:
  median words up by a third or more.
- `heard` (the same model asked to transcribe, untimed) **5–7% normalized
  WER** overall, accented under 20%: better than turbo's 8.6%.
- Peak GPU memory: R3 about 17–18 GB; R0 and R1 2–3 GB.
- About \$0.25 for the run, plus a `--limit 8` smoke test (about \$0.05) and
  one CPU image rebuild. The plan budgeted \$5.51 for R3.
