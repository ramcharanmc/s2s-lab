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

## What an eval-set run holds

Provenance first: `rung`, `description`, `commit`, `machine`, `ran`, `load_ms`,
and `evalset` (build date, seed, turn count). Then `turns`, one entry per clip:
`cohort`, `transcript`, `reply_text`, `stages_ms`, `untimed_ms`, `total_ms`,
`reply_s`, `leading_silence_ms`.

Re-score without a GPU or any model (needs the eval set's manifest, not its audio):

    PYTHONPATH=src python -m s2slab.evalrun --score runs/r0-gpu/evalset-modal-t4.json

## Sources

`transcript` is a model's output from eval-set audio; no source audio or
reference text is stored here. Datasets, pinned revisions and licences are in
`evalset/README.md`.