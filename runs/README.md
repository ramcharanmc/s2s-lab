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
| `r0-gpu/evalset-modal-a10.json` | R0 on the A10, the same-container comparator for the R3 row | `960b491` | Modal NVIDIA A10, one container: r0-gpu then r1-gpu then r3-gpu | page 20 §1, §18 |
| `r1-gpu/evalset-modal-a10.json` | R1 on the A10, the same container | `960b491` | Modal NVIDIA A10, same container | page 20 §1, §18 |
| `r3-gpu/evalset-modal-a10.json` · `evalset-modal-a10-report.md` | **the R3 row**: Qwen2-Audio hears each turn, no transcript in the loop; `heard` = the same model asked to transcribe, untimed; peak GPU memory; the report is the run's printed output (its `heard` table predates the wrapper rule -- `--score` the file for the current number) | `960b491` | Modal NVIDIA A10, same container | page 20 §1, §18 |

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

**Scored (2026-09-23, run `960b491`, one A10 container, R0 then R1 then R3):**

| A10, 200 clips | R0-gpu | R1-gpu | R3-gpu |
|---|---:|---:|---:|
| total p50 / p90 | 441 / 748 ms | 451 / 754 | **1055 / 2016** |
| first audio p50 / p90 | 441 / 748 | **333 / 483** | 1055 / 2016 (batch) |
| first sound p50 / p90 | 760 / 1063 | 651 / 787 | 1363 / 2318 |
| LLM stage p50 | 315 | 263 | **971** |
| peak GPU memory | 1,851 MB | 1,684 MB | **16,993 MB** |
| reply words, median / p90 | 14 / 26 | 14 / 26 | **19 / 41** |
| WER in the loop, normalized | 12.0% | 12.0% (identical transcripts and replies) | none: no transcript exists |
| asked to transcribe, outside the loop | | | **9.1%** re-scored (15.5% as the run printed it) |

- Total p50 1.5–2.5 s -> **1.06 s, under**; 3–5x R0 -> **2.4x R0's total, 3.2x
  R1's first audio: under**. R3's 971 ms LLM stage is the input-side wait the
  plan warned about; Kokoro adds 87 ms and 300 ms of leading silence.
- Replies a third longer -> **+36% (median 14 -> 19 words), right**; the 7B
  model also asks a question back in 34 of 200 replies against R0's 8.
- Heard WER 5–7% -> **9.1%, wrong** -- but only after re-scoring: Qwen2-Audio
  wrapped 35 of 200 transcripts in seven different sentences and the run's
  code stripped two, so the run printed 15.5%. The saved file keeps what the
  model said; `evalrun` now scores it through the wrapper rule as it stands.
  Re-scored: clean 1.8, slow 1.9, disfluent 8.8, accented 23.2 (predicted under
  20: wrong). Better than base.en in the loop (12.0), a hair behind turbo
  (8.6); eight clips come back short, one code-switched into Chinese.
- Memory 17–18 GB -> 17.0 GB, right; R0/R1 2–3 GB -> 1.7–1.9, under.
- Cost about \$0.25 -> \$0.221, right; the smoke test \$0.020.
- Same hardware matters, again: R0 measured 0.44 s here against 0.66 s on the
  T4 of 09-16 (transcripts 198 of 200 identical, WER 12.0 against 11.9).

## R4 as a rung — written before the eval-set run (2026-09-23)

`modal run modal_app/r4_bench.py --rung r3-gpu,r4-gpu --evalset`: one L40S
container, R3 re-measured beside R4. Written after the probe-clip smoke test
(`34a4209`, 30 clips, L40S, \$0.18), which already showed R4 at **8.26 s p50**
-- the design note's 2–4 s was wrong before the eval set was touched: the
Talker generates the reply's speech tokens one by one after the Thinker has
written it, then token2wav renders them. The smoke test also showed the Talker
*samples* by default; from `SEED` on, every turn is seeded, so the eval-set
audio is a function of the clip alone. Expectations, to be scored:

- R4 total p50 **7–10 s** on the 200 clips (replies capped at 60 text tokens,
  so the eval set's longer clips should not lengthen the wait much); p90
  under 14 s. R3 on the same L40S **0.8–1.1 s**: R4 waits about **8×** R3.
- R4 leading silence about 200 ms (smoke: 200) against Kokoro's 300.
- R4 peak GPU memory **22–26 GB**; R3 17 GB as on the A10.
- R4's written replies about as long as R3's (median 16–22 words): the same
  60-token cap, a same-family 7B Thinker.
- About \$1.2 for the run (≈ 35 min at \$1.95/h).

