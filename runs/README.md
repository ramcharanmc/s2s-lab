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
| `r3-gpu/evalset-modal-l40s.json` | R3 re-measured on the L40S, the same-container comparator for the R4 row | `339092a` | Modal NVIDIA L40S, one container: r3-gpu then r4-gpu | page 20 §1, §19 |
| `r4-gpu/evalset-modal-l40s.json` · `evalset-modal-l40s-report.md` | **the R4 row**: Qwen2.5-Omni hears each turn and speaks the reply itself (no transcript, no TTS), every turn seeded; peak GPU memory; the report is both rungs' printed output | `339092a` | Modal NVIDIA L40S, same container | page 20 §1, §19 |
| `prosody/pr008.json` | PR-008, the registered run: R3 and R4 answer the 39 locked prosody pairs as a caller's turn; every reply's text and measures, every pair's direction | `f4da0bc` | Modal NVIDIA L40S | page 20 §19 |
| `r5-moshi/evalset-modal-a10.json` · `evalset-modal-a10-report.md` | **the R5 row, Moshi**: full duplex, one conversation per clip (4.8 s lead-in, the clip, 8 s tail), every step's text piece and voiced run, seeded; the report is the run's printed output | `6723cb4` | Modal NVIDIA A10 | page 20 §1, §20 |
| `r5-personaplex/evalset-modal-a10.json` · `evalset-modal-a10-report.md` | **the R5 row, PersonaPlex**: the same driver, R0's instruction as its role, voice NATF2; set-up time per call | `6723cb4` | Modal NVIDIA **A10G** (the file name says a10) | page 20 §1, §20 |
| `severity/pr009.json` · `pr009-report.md` | **PR-009, the registered run**: the severity render on the held-out voices and sentences -- every target and measure, each neutral clip's SHA-256, the criteria; the report is the run's printed output. Not supported (S4) | `6c43d5e` | laptop CPU (WSL), Kokoro-82M, \$0 | page 20 §21 |
| `prosody/pr008-report.md` | the same run's report, regenerated from the run file by the frozen code (the printed output was overwritten; PR-008 Deviations) | `f4da0bc` | Modal NVIDIA L40S | page 20 §19 |
| `ft1/pr010.json` · `pr010-report.md` | **PR-010, the registered run**: base Moshi, then Moshi + FT-1's LoRA, answering the user channel of the 100 DailyTalk test dialogues, one container; every dialogue's offsets and text, the reference, the adapter's SHA-256 and config, the criterion. Not supported: moved away | `3d36f74` | Modal NVIDIA L40S | page 20 §22 |
| `ft1/training/` | FT-1's training evidence, as it came off the machine: the config run (both launches, identical), the training log (every step's loss), the script's log, the training list and SHA256SUMS. Not the adapter (740 MB, `data/ft1/`, ignored) | `4df9ef3` | Vast.ai, one H100 80 GB | page 20 §22 |
| `ft2/pr011.json` · `pr011-report.md` | **PR-011, the registered run**: small.en, then small.en with FT-2's LoRA merged in, transcribing the frozen 200-clip eval set in one container -- every transcript, every clip's recognizer time, the criteria, the exported model's SHA-256. Supported | `65c833c` | Modal Tesla T4 | page 20 §23 |
| `ft2/training/` · `ft2/dev-look/` · `ft2/summaries/` | FT-2's training evidence from the volume `ft2`: the six validations' transcripts and metrics, `metrics.csv`, the hyperparameters and hydra config; the two dev looks over the 250 validation clips (base, the LoRA); the training, export and smoke-test summaries. Not the adapter or the merged model (`ft2` volume) | `f37d0df` | Modal NVIDIA A10 | page 20 §23 |

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
## What a full-duplex run holds

`rung`, `description`, `machine`, `commit`, `versions`, `load_s`,
`peak_vram_mb`; `repeat` (the first two clips driven twice: the largest output
difference, 0 when identical); and `turns`, one entry per clip: `cohort`,
`stream_s`, `wall_s`, `rtf`, `setup_s`, `runs` (every voiced utterance of the
model, `[start, end]` in stream time), `pieces` (every text piece with its
80 ms step), `caller_end_s` (with the lead-in), `gap_ms`, `overlap`,
`backchannels`, `interruptions`, `voiced_during_caller`, `opened`,
`reply_text`. The definitions are in `src/s2slab/duplex.py`; no audio is
stored.

Re-count the answers without any model:

    python -c "import json; t = json.load(open('runs/r5-personaplex/evalset-modal-a10.json', encoding='utf-8'))['turns']; print(sum(v['gap_ms'] is not None for v in t.values()), 'of', len(t))"

## What PR-010's run holds

`registration`, `commit`, `ids`, `adapter_sha256`, `adapter_config`, `machine`, `versions`,
`tail_s`, `gpu_mb_left_after_base`, `criteria`; `reference`, one row per dialogue (the human on
the left answering the right); and `base` and `lora`, each with `rung`, `load_s`, `peak_vram_mb`
and `rows`, one per dialogue. A row is `turngaps.summary(user, answerer)`: `n_transitions`,
`fto_ms` (every floor-transfer offset), `overlapped`, `within`, `overlap_ratio`, `speech_s`
(user, answerer); a model's row adds `rtf`, `setup_s` and `text`. The definitions are in
`src/s2slab/turngaps.py` and `src/s2slab/dailytalk.py`; no audio is stored.

Re-score the criterion without any audio or model (10,000 resamples, a few seconds):

    PYTHONPATH=src python -c "import json; from s2slab.dailytalk import pr010; d = json.load(open('runs/ft1/pr010.json', encoding='utf-8')); k = lambda r: {int(i): v for i, v in r.items()}; print(pr010(k(d['base']['rows']), k(d['lora']['rows']), k(d['reference'])))"

## What PR-011's run holds

`registration`, `commit`, `base_model`, `export_name`, `export_sha256`, `machine`, `n_clips`,
`criteria` (S1, S2, S3 with their intervals and passes), and `models`: for `small.en` and `ft2`,
the model's `path`, `load_s` and `turns`, one per clip -- `cohort`, `text` (the transcript) and
`asr_ms`. The references are the eval set's (`evalset/manifest.json`); no audio is stored.

Re-score the criteria without any audio or model:

    PYTHONPATH=src python -c "import json; from s2slab import ft2; d = json.load(open('runs/ft2/pr011.json', encoding='utf-8')); h = {m: {i: t['text'] for i, t in d['models'][m]['turns'].items()} for m in d['models']}; d['criteria'] = ft2.pr011(h['small.en'], h['ft2']); print(ft2.format_pr011(d))"

## What PR-012's run holds

`runs/ft3/pr012.json` is a severity run (next section) with two voices that are FastPitch
checkpoints, and: `settings` (the centre of the pitch scale, the range gain, the shortest pause),
`model_files` and `model_sha256` (the four `.nemo` files), `criteria` for the 32 pooled sweeps,
`criteria_by_voice`, `voice_info` (each checkpoint's pitch statistics and hop), and in each row
`centre_hz` (the neutral render's median F0) and `pauses` (how many the control found).
`pr012-report.md` is the run's printed report without the six lines NeMo wrote above it.

`training/`: the data stage's two run files (the first extract, at the wrong pitch range, is kept
and named so), the vocoder stage's run file, and each training's validation losses as printed.
FastPitch's own run file does not exist: the run failed after the training (docs/ft3-log.md,
entry 7). `dev/`: the fine-tuned voice's dev sweep, and `pr012-dry.json`, the registered run's
code on the four dev sentences for both voices.

Re-count the criteria without any audio or model (the `ft` project's environment):

    cd ft && PYTHONPATH=.:../src uv run python -c "import json; from src.ft3 import pr012; print(pr012.format_criteria(json.load(open('../runs/ft3/pr012.json', encoding='utf-8'))))"

## What a severity run holds

`registration`, `commit`, `ran`, `voices`, `sentences`, `scores`, `sr`,
`tolerance`, `criteria`; and `rows`, one per voice x sentence x condition:
`neutral_s`, `neutral_sha256` (of the clip's float32 samples), `monotone`, and
`steps`, one per score: `target` and `achieved` for `f0_offset_hz`,
`f0_dev_scale`, `rate`, `pause_scale`, and the misses. The definitions are in
`src/s2slab/severity.py`. The renders and the exact neutral clips stay in
`results/severity/pr009/` (ignored); from those clips, any render can be made
again. Every file is synthetic speech and says so.

Re-count the criteria without any audio or model:

    PYTHONPATH=src python -c "import json; from s2slab.severity import format_criteria; print(format_criteria(json.load(open('runs/severity/pr009.json', encoding='utf-8'))))"

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

**Scored (2026-09-23, run `339092a`, one L40S container, R3 then R4, 200 clips, \$1.318):**

| L40S, 200 clips | R3 (Qwen2-Audio -> Kokoro) | R4 (Qwen2.5-Omni speaks) |
|---|---:|---:|
| total = first audio, p50 / p90 | 688 / 1286 ms | **9166 / 17325 ms** |
| first sound, p50 | 991 ms | 9366 ms |
| leading silence, p50 | 300 ms (Kokoro) | **200 ms** |
| peak GPU memory | 16,986 MB | **22,518 MB** |
| reply words, median / p90 | 18.5 / 39 | 22 / 37 |
| reply audio, median / p90 | 6.3 / 15.1 s | 7.2 / 13.3 s |
| replies that ask a question back | 32 of 200 | **180 of 200** |
| asked to transcribe, outside the loop | 9.1% (as on the A10) | not asked |

- **R4's wait is its own speech, rendered slower than real time:** total =
  1.33 s per second of reply audio (r = 0.997 over 200 turns). The Talker and
  token2wav make speech at about 0.75x real time, after the Thinker has
  written the text. A streaming Talker would change first audio entirely;
  transformers does not expose one.
- The two models behave differently as agents: R4 asks the caller a question
  back in 180 of 200 turns; R3 often restates the clip (on `clean-00` it
  repeats the LibriSpeech sentence word for word).
- **R3 is not bit-reproducible across cards:** greedy decoding, yet 157 of 200
  replies match between the A10 and the L40S (bf16 arithmetic differs); its
  transcription WER does not move (9.1%), and it is faster here (688 vs 1055
  ms). Rows are compared within one container, as ever.
- Predictions: R4 p50 7–10 s -> 9.17 s, **right**; p90 under 14 s -> 17.3,
  **wrong**; R3 0.8–1.1 s -> 0.69, **under**; about 8x -> **13x, wrong**;
  leading silence about 200 ms -> 200, **right**; memory 22–26 GB -> 22.5,
  **right**; words 16–22 -> 22, **right at the edge**; about \$1.2 -> \$1.32,
  **about right**.

## R5 as a rung — full duplex (2026-09-24)

`modal run modal_app/r5_bench.py --evalset` and the same with `--model
r5-personaplex`. Written after the runs: unlike R3 and R4, no expectations
were written down before the eval set was touched, so none are scored. The
plan's own two claims are checked instead -- "Verify the ~200 ms and
5-per-A100 claims with your own numbers" (v2, R5).

Two models behind one driver (`s2slab.duplex`), each in its own image because
each ships a package named `moshi`: the same-container rule cannot hold
between them. The stream-time measures do not depend on the card; the RTF
does. Every conversation is 4.8 s of silence (the model opens the call), the
clip, then 8 s of silence, seeded. Both settings were decided on the 30 probe
clips, in two dev fixes (`236486b`, `6723cb4`); `src/s2slab/duplex.py` gives
the reasons.

**Measured (2026-09-24, commit `6723cb4`, 200 clips each, \$0.867 + \$1.372):**

| 200 clips | Moshi (moshiko) | PersonaPlex (R0's instruction as its role, voice NATF2) |
|---|---:|---:|
| card | A10 | A10G |
| answered within the 8 s tail | 96 | **192** |
| response gap, p50 | **150 ms** | 460 ms |
| response gap, p90 | — (96 answers; the rule wants 100) | 758 ms |
| gaps between 300 and 800 ms | 10 of 96 | 150 of 192 |
| already talking as the caller stopped | 24 | 12 |
| clips where it spoke while the caller spoke | 65 | 73 |
| backchannels / interruptions | 55 / 35 | 50 / 46 |
| reply words, median | 11 | 16 |
| replies that end in a question | 13 | **115** |
| RTF, median (worst) | 0.69 (1.68: the first call, before anything is warm) | 0.80 (0.82) |
| set-up before each call | 0.01 s | **5.95 s** |
| peak GPU memory | 16,621 MB | 18,138 MB |

- **The ~200 ms claim holds for Moshi's answers, not for Moshi.** When it
  answers, the gap is 150 ms p50, and 24 of its 96 answers begin before the
  caller's last word. But after "Hello, how is it going?" it stays silent
  through the whole 8 s tail on 104 of 200 clips; on the slow cohort it
  answers 18 of 50. Its replies are open chat, since it has no role.
- **PersonaPlex waits, then answers.** 192 of 200 answered, 150 of them
  between 300 and 800 ms: a steady rhythm, close to an endpointer's, but
  chosen by the model. The role took hold: it opens every call as customer
  service and ends 115 of 200 replies with a question.
- **5 per A100: not tested.** No A100 run. On a 24 GB A10 one conversation
  fits (16.6 and 18.1 GB), at RTF 0.69 and 0.80: one conversation per card.
- **PersonaPlex's set-up is a real cost.** It replays its voice prompt and
  its role before every call: 5.95 s before it can hear the caller, outside
  the stream time and the RTF.
- **The text stream is not a transcript.** Both models skip pieces in their
  inner monologue ("What' the problem?", "I' be here"), so `reply_text` shows
  what was said, not word for word. The reply audio was not transcribed.
- **Not bit-reproducible across cards, reproducible on one.** PersonaPlex's
  greeting on the A10 smoke test was "Hello, this is phone support. How can I
  help you today?"; on the A10G eval run, with the same seed and the same
  silence, "Hello, thank you for calling customer service. How can I assist
  you today?". On one card, each clip's conversation repeats exactly
  (`clean-00`, `clean-01` driven twice: identical).
- **Not the same quantity as R0-R4's latency.** R0-R4 are handed a finished
  turn and timed from the clip's end to first audio: the endpointer's wait is
  not in those numbers. R5's gap includes the model's own decision that the
  caller has finished.
- The PersonaPlex file is named `a10` but ran on an A10G: Modal's `A10G`
  request landed on either card. Its report keeps the package's own "Done
  loading" prints, three per call.
