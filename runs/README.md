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

## What an eval-set run holds

Provenance first: `rung`, `description`, `commit`, `machine`, `ran`, `load_ms`,
and `evalset` (build date, seed, turn count). Then `turns`, one entry per clip:
`cohort`, `transcript`, `reply_text`, `stages_ms`, `untimed_ms`, `total_ms`,
`reply_s`, `leading_silence_ms`.

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