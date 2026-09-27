# Pre-registrations

Rules, from the v1 Research plan (§8):

- **An entry is registered by the commit that adds it.** The commit hash is the
  timestamp. It cannot be backdated.
- **Criteria are written before the data they are tested on is examined.**
- **A registered entry is never edited.** Any change goes in its *Deviations*
  section, dated, with the reason.
- **Results are appended whatever they say.** A registered experiment that goes
  unreported is a missing result, not a non-result.

## Registry

| ID | Question | Registered | Status |
|---|---|---|---|
| PR-000 | T0 vs T1, T1 vs T2, T0 700 vs T2 on the seed-1000 confirmation run | **Not verifiably** — see entry | Reported, Notion page 18 §6 |
| PR-001 | Does a phrase-level completeness check reduce disfluent interruptions beyond T1? | The commit that adds this entry | Open |
| PR-002 | Does streaming the reply (R1) cut the wait for first audio against R0, without changing the words, slowing the turn or stalling playback? | `194e2bb` | **Reported: not supported** (S1, S2, S3 failed; S4, S5 and validity held) |
| PR-003 | On the author's real voice, does a barge-in policy that reads the caller's words (B2) make fewer wrong stop / keep-going decisions than the best duration threshold, and stop within 1.5 s? | `e707190` | **Reported: supported** (S1: wrong on 1 of 20 against 8, p = 0.039; S2: 9 of 10 within 1.5 s; any second error would have failed S1) |
| PR-004 | Does a slow-tier recognizer (R2, large-v3-turbo) cut R1's word error rate by at least 3 points overall and 5 on accented speech, while the reply still starts within a second? | `415ddc7` | **Reported: not supported** (S2 failed: accented −4.3 points, needed −5; S1 −3.4 points and S3 889 ms held) |
| PR-005 | Does telling R1's LLM one word about the caller's tone (a $0 eGeMAPS tagger) let it tell an upset caller from the same words said calmly, for fear and for anger, better than shuffled tags do? | `44a9a5f` | **Reported: not supported** (fear 7 of 13, one under the bar, p = 0.015; anger 11 of 13, p < 0.001, held; sadness 7 of 13, p = 0.025, held though expected to fail) |
| PR-006 | Does a speech-LLM that hears the audio (R3, Qwen2-Audio) rank the upset clip above its calm twin on the 39 locked prosody pairs (Claim A), and answer at least as many pairs right as the $0 side-channel on its own metric (Claim B)? | `95c3d33` | **Reported: Claim A supported** (13 of 13 for every emotion); **Claim B not supported** (11 · 11 · 9 against 7 · 7 · 11; anger short; sign test p = 0.21) |
| PR-007 | On 38 fresh CREMA-D pairs from actors no system here has scored, does giving the $0 tagger the speaker's own neutral clips (a per-caller baseline) let it tell more upset-calm pairs apart than the same tagger without it, without tagging the upset clips neutral? | `02baa74` | **Reported: not supported** (26 of 38 told apart either way; fixed 2, broken 2, one-sided p = 0.69) |
| PR-008 | When the caller's voice is upset rather than calm, with the same words, does R4 (Qwen2.5-Omni, speaking for itself) answer with a wider pitch range -- beyond chance and beyond what R3's words carry into Kokoro? | `f4da0bc` | **Reported: not supported** (R4 14 wider / 16 narrower of 39; no consistent direction, p = 0.86) |
| PR-009 | On two Kokoro voices and eight neutral sentences the render was not developed on, does raising a PHQ-9 or YMRS score move pitch, range, rate and pauses the way page 09's mapping says, at every step and within tolerance? | `6c43d5e` | **Reported: not supported** (S4 failed: pitch offset on target in 169 of 192 checks, 88.0%, needed 173; S1 96 of 96, S2 29 of 32, S3 384 of 384, S5 177 of 192 held) |
| PR-010 | Does a LoRA trained on DailyTalk (kyutai's moshi-finetune, test fold held out) move Moshi's turn gaps closer to DailyTalk's own, on 100 held-out dialogues? | `4df9ef3` | **Reported: not supported, moved away** (distance to the reference 496 ms base, 1,035 ms LoRA; gain −539 ms, 95% interval −685 to −402; overlap 85 → 2 turn changes, but median gap 160 → 1,220 ms against 420) |
| PR-011 | Does a LoRA trained on verbatim transcripts make `whisper-small.en` write down disfluent speech as spoken -- fillers and all -- on the eval set's disfluent cohort, without hurting the other cohorts? | `65c833c` | **Reported: supported** (S1 raw WER on the disfluent cohort 9.9% → 2.3%, −7.6 points, interval −9.6 to −5.7; S2 29 of 29 fillers kept; S3 the other 150 clips −2.3 points normalized where +1.0 was allowed -- most of it EdAcc's verbatim references, not new accent robustness) |
| PR-012 | On eight sentences neither voice has spoken, do renders made by giving page 09's severity targets to FastPitch as its own inputs meet PR-009's five criteria, over the base LJSpeech voice and a 5-minute fine-tune together? | `89edd05` | **Reported: not supported** (all five criteria failed, as predicted: S1 92 of 96, S2 14 of 32, S3 357 of 384, S4 109 of 192, 56.8%, S5 115 of 192, 59.9%; rate on target in 192 of 192; the base voice 67 and 78 of 96 on S4 and S5, the 5-minute fine-tune 42 and 37) |
---

## PR-000 — retroactive record (not a valid pre-registration)

Kept so the record is complete, and to be described accurately wherever it is
cited.

- The three comparisons in `COMPARISONS` (`sweep.py`) were chosen **after**
  looking at seed 0.
- They were written into the working tree **before** the seed-1000 run, but
  committed only afterwards, in `6a0741e`. There is no verifiable timestamp
  showing they preceded the results.
- Correct description: *confirmatory in intent, not verifiably pre-registered.*

The only criterion that was genuinely registered in advance is the v1 P1b
go/no-go on Notion page 16, written before any results existed: *"Prosodic
policy dominates the fixed threshold on the SEP-28k slice at some operating
point with cutoff rate < 5%."* On the synthetic proxy it was **not met**; the
best disfluent result was 28% [22–33%].

---

## PR-001 — Phrase-level completeness check

**Registered:** 2026-09-11, by the commit that adds this entry.
**Author:** Ram Charan

### Background

Error analysis on exploratory data (seed 0, 250 disfluent turns, commit
`67dd495`):

- 89% of T1's remaining interruptions come right after a **content** word, so
  the 46-word list never triggers the hold.
- Only 11% come from a pause that outlasted the hold, so a longer hold would
  rescue at most 17 of 158.
- T1 rescues pauses, not turns. A disfluent turn has 2–4 pauses, and the check
  has to survive all of them.

### Question

Does adding a phrase-level completeness check to T1 — call it **T1b** — reduce
interruptions of disfluent speakers, at an acceptable cost in latency?

### What T1b is allowed to be

The method is left open. These constraints are not:

1. **Same timing as T1.** Base threshold 500 ms, extension 400 ms. Only the
   answer to "is this sentence unfinished?" changes.
2. **A strict extension of T1.** T1b may only *add* holds relative to T1, never
   remove one. Consequence: T1b can never interrupt a turn that T1 spared, so
   its interruption rate cannot go up. The experiment is really about **how
   much it rescues and what that costs in latency.**
3. **Not derived from the evaluation sentences** — see V1.

### Validity requirements

If any of these is not met, the result is reported as **exploratory**, not as a
test of PR-001.

- **V1 · Held-out sentences.** The simulator currently uses three fixed
  sentences. A completeness check developed against them can simply memorise
  them, and success would then be guaranteed and meaningless. Before T1b
  development begins, `simulate.py` gets a sentence split:
  - **DEV sentences**, used to build and tune T1b;
  - **EVAL sentences**, written and committed *before T1b exists*, and used in
    no run until confirmation.
- **V2 · Frozen check.** T1b's definition is committed before the confirmation
  run, and that commit hash is recorded under *Results*.
- **V3 · Fresh seed.** Confirmation uses **seed 2000, 250 turns per cohort**,
  never examined before the run. Development uses seeds 0 and 1000 and DEV
  sentences only.
- **V4 · Strict extension, tested.** A test asserting that T1b never interrupts
  a turn T1 spared passes at the frozen commit.

### Primary outcome

Disfluent interruption rate, **T1_semantic_500ms vs T1b at 500 ms**, compared
turn by turn on the same turns (seed 2000, EVAL sentences).

### Success criterion — all four must hold

| | Criterion |
|---|---|
| **S1 · Effect** | T1b interrupts **at least 10 percentage points fewer** disfluent turns than T1 (paired difference ≥ +10 pp). |
| **S2 · Confidence** | The paired bootstrap 95% CI on that difference lies entirely above 0, **and** the Holm-adjusted McNemar p < 0.05 across the family below. |
| **S3 · Speed guardrail** | On the **clean** cohort, T1b's median wait is at most **50 ms** above T1's, and its p90 at most **100 ms** above T1's. |
| **S4 · Safety check** | Clean and slow interruption rates do not rise above T1's. Guaranteed by V4, so this is asserted rather than tested. |

**Why 10 pp.** At the same threshold T1 improved on T0 by 16 pp. A further gain
under 10 pp — from 67% to above 57% — would not justify the complexity of a
classifier. This is a judgement, recorded before the data.

**Why the speed guardrail carries the weight.** Because T1b can only add holds,
"fewer interruptions" is close to guaranteed in *direction*. A check that holds
after every word would pass S1 and S2 while making every clean turn 400 ms
slower. S3 is what makes this a real test rather than a certainty.

**Family for Holm:** T1 vs T1b on clean, disfluent and slow — 3 comparisons.
Nothing else is tested confirmatorily.

**If any criterion fails,** PR-001 is reported as *not supported*, with a
`diagnose.py` breakdown of T1b's remaining interruptions as the error analysis.

### Analysis plan

1. Build and commit the DEV/EVAL sentence split (V1).
2. Develop T1b on DEV sentences, seeds 0 and 1000 only.
3. Freeze T1b and record the commit (V2). Confirm the V4 test passes.
4. Run the sweep on **seed 2000, n = 250, EVAL sentences**, with the PR-001
   comparison family.
5. Record the results below, verbatim.
6. Only after step 5: run `diagnose.py` on seed 2000, labelled exploratory.

### What this cannot show

Even a pass is a result about **synthetic** speech. The simulator places pauses
at random word boundaries, while real hesitations tend to precede content words.
A pass shows T1b handles random-boundary pauses on unseen sentences. It does not
show T1b helps real disfluent speakers — only the real-audio probe set (v2 §6.3)
can.

### Deviations

*None.*

### Results

*Not yet run.*


---

## PR-002 — Does streaming the reply cut the wait for first audio?

**Registered:** 2026-09-18, by the commit that adds this entry.
**Author:** Ram Charan

### Background

R0 (naive cascade) on the frozen 200-clip eval set, Modal T4, commit `0af142a`:
processing p50 655 ms, p90 1130 ms. LLM + TTS time correlates 0.97 with the
number of words in the reply, so the tail is set by reply length. R0's first
audio is its total: nothing exists until everything does.

R1 (commits `3780e89` → `b83bb84`) keeps R0's three models, prompt and greedy
decoding. It streams the LLM's tokens, cuts them into clauses at punctuation
(`chunking.py`: `. ! ?` always, `, ; :` after `MIN_WORDS = 4`, abbreviations
excepted), and synthesizes and emits each clause while the next is written.

Development data, **the 30 probe clips on a laptop CPU only**:

- first audio p50 3467 ms against R0's 5292 (66%); r(first audio, reply words) = 0.07;
- 11 of 30 replies are a single short sentence with no boundary, and gain nothing;
- the first clause is long, 10.5 words at the median;
- 7 of 30 turns **stall**: the next piece did not exist when the one before
  finished playing (up to 1.8 s of dead air, in no saved WAV);
- the author listened to c12 and n04: an audible pause and pitch reset at each
  seam, noticeable on a call, not annoying, possibly helping intelligibility.
  R1's reply audio runs 0.3–0.6 s longer per seam (Kokoro pads every piece).

The thresholds below were set after this probe-clip run and **before any R1
run on the eval set**. What was already known about the eval set: R0's reply
*lengths* (word counts, for the 0.93 / 0.97 correlations). R0's eval-set reply
texts were not read and no split statistics were computed on them.

### Question

On the same clips, models and GPU, does R1 deliver its first audio reliably
sooner than R0 delivers its reply, without saying different words, slowing the
whole turn, or stalling during playback?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Same run** | `r0-gpu` then `r1-gpu` over all 200 clips inside **one Modal container call**, same T4, same image. R0 is re-run; the saved `0af142a` run is not the comparator. |
| **V2 · Frozen code** | `chunking.py`, `r1.py`, `r0.py` and `rungs.py` are unchanged between this commit and the run. Only `modal_app/` and the analysis script may change. The run's commit is recorded below. |
| **V3 · Same words** | R1's reply has the same words as R0's (`split()` equal) on **at least 190 of 200** turns. Fewer means streaming changed *what* is said, the timing comparison is confounded, and PR-002 is reported as **not a valid test**; the matching subset may then be analysed, labelled exploratory. |
| **V4 · Whole set, one run** | All 200 clips, the first complete run. No clip is dropped and no run is repeated to improve a number; a crashed run is reported as such and repeated once, in full. |

### Primary outcome

`first_audio_ms`: from the rung receiving the finished turn to its first
non-empty piece of reply audio existing. For R0 this equals `total_ms`.
Compared turn by turn on the same 200 clips.

### Success criterion — all five must hold

| | Criterion |
|---|---|
| **S1 · Median** | R1 first-audio p50 ≤ **75%** of R0's p50. |
| **S2 · Tail** | R1 first-audio p90 ≤ **70%** of R0's p90. |
| **S3 · Reply length no longer sets the wait** | Pearson r between R1's first audio and its reply word count is **< 0.3**, and its 95% interval lies entirely below R0's r for total time. |
| **S4 · Speed guardrail** | R1 total p50 ≤ **1.15×** R0's total p50. |
| **S5 · Stall guardrail** | At most **20 of 200** R1 turns stall at all (`stall_ms > 0`), and at most **10 of 200** stall for more than 200 ms. |

**Confidence.** For S1, S2 and S4 the statistic is a ratio of percentiles. A
paired bootstrap resamples the 200 clip ids with replacement (10,000
resamples, seed 0, percentile interval), recomputing both rungs' percentiles
on each resample. The criterion holds only if the **upper end of the 95%
interval** is at or below the threshold. S3 uses the same resampling. S5 is a
count and is taken as observed.

**No multiplicity correction:** success needs all five at once, so testing each
at 95% cannot inflate the chance of a false pass.

**Why 75% and 70%.** R1 adds a thread, a chunker, audible seams and the risk of
stalls. A quarter off the median wait, and more off the tail where R0 hurts
most, is the least that would justify them. ASR and one synthesis are fixed
costs streaming cannot shrink, so 50% is not available with these models.

**Why the guardrails carry weight.** First audio can be made arbitrarily early
by emitting tiny first pieces. S4 and S5 are what make an early start count
only if the rest of the reply keeps up.

**If V3 holds and any of S1–S5 fails,** PR-002 is reported as *not supported*,
with the failing criterion, and a breakdown by number of pieces per reply.

### Predictions, recorded but not criteria

- S1 ratio about 0.70: **the criterion most likely to fail**, since its interval must clear 0.75.
- S2 ratio about 0.60. S3 r about 0.1. S4 about 1.05: one GPU cannot overlap LLM and TTS the way 16 CPU threads did, where R1's total *fell* 10%.
- Stalls on fewer than 5% of turns: the T4 synthesizes about 10× faster than the laptop.
- About a third of replies will be one piece and gain nothing.

### Reported alongside, descriptively

First sound (first audio + leading silence); per-cohort medians; pieces per
reply; the stall distribution; reply audio length R1 − R0 per seam; WER of both
rungs (expected equal; any gap is run-to-run fp16 noise); the run's cost.

### Analysis plan

1. Commit this entry. Code under V2 is frozen from here.
2. Add the two-rung Modal entry point and the comparison script; test both on fake runs.
3. Run once on Modal (V1, V4). Save both JSONs to `runs/`.
4. Run the comparison script; record its output below, verbatim.
5. Only after step 4: any further look at the eval-set replies, labelled exploratory.

### What this cannot show

- **Not what a caller experiences end to end.** Processing time only: no
  endpointer wait, no network, no audio device. Stalls are modelled (pieces
  played back to back from the first), not heard.
- **Not seam quality.** No metric here scores the pause and pitch reset at
  each join. One listener, two dev clips, is an anecdote.
- **Not reply quality.** Both rungs say the same words, good or bad.
- **Order.** R0 runs before R1 in the container; a warm-up or thermal drift
  between the two halves would land in the comparison.
- **One GPU type, small models, one run.** Run-to-run spread on Modal is unmeasured.

### Deviations

- **2026-09-18, registry row.** The PR-002 row in the Registry table was left
  out of the registering commit `194e2bb` and added by `23ecc0f`. The entry,
  including every threshold, is unchanged since `194e2bb`.
- **2026-09-18, smoke test.** Between registration and the run, both rungs ran
  once on the 30 probe clips on a Modal T4 (commit `b107927`), to check R1 on a
  GPU. Development data; no eval-set clip was run before the registered run.

### Results

**Run:** 2026-09-18, commit `b107927`, Modal T4 (driver 580.95.05), one container
call: `r0-gpu` then `r1-gpu`, all 200 clips each, first and only run. 324 s
inside the function, about \$0.055. Files: `runs/r0-gpu/evalset-modal-t4-pr002.json`,
`runs/r1-gpu/evalset-modal-t4-pr002.json`. V1, V2 (zero changed lines in the
four frozen files since `194e2bb`), V3 and V4 held.

**Verdict: NOT SUPPORTED.** Output of `python -m s2slab.compare`, verbatim:

```text
PR-002: r1-gpu against r0-gpu
  r0-gpu: commit b107927 · Modal Tesla T4, 580.95.05, one container: r0-gpu then r1-gpu · run 2026-09-18
  r1-gpu: commit b107927 · Modal Tesla T4, 580.95.05, one container: r0-gpu then r1-gpu · run 2026-09-18
  paired bootstrap: 10000 resamples of 200 clips, seed 0

V3 same words        PASS  same words on 200 of 200 turns; needs >= 190
S1 median            FAIL  first-audio p50 ratio 0.744 [95% CI 0.697, 0.812]; needs upper end <= 0.75
S2 tail              FAIL  first-audio p90 ratio 0.655 [95% CI 0.591, 0.743]; needs upper end <= 0.7
S3 reply length      FAIL  r(first audio, reply words) 0.455 [95% CI 0.308, 0.593]; needs r < 0.3 and upper end below the batch rung's r(total, reply words) = 0.975
S4 speed guardrail   PASS  total p50 ratio 1.030 [95% CI 1.007, 1.084]; needs upper end <= 1.15
S5 stall guardrail   PASS  0 turns stall (max 20), 0 for more than 200 ms (max 10)

NOT SUPPORTED: failed S1 median, S2 tail, S3 reply length

Descriptive, not criteria:
  ms, p50 / p90                  r1-gpu           r0-gpu
  first audio               491 /   732      660 /  1118
  total                     680 /  1181      660 /  1118
  first sound p50                   806              973
  first-audio p50 by cohort: clean 541 vs 647, accented 434 vs 669, disfluent 486 vs 689, slow 500 vs 616
  pieces per reply (turns, median first-audio ratio): 1: 95 turns, 0.98, 2: 75 turns, 0.65, 3: 22 turns, 0.46, 4+: 8 turns, 0.31
  stalls: 0 turns
  extra reply audio per seam: 0.39 s (median, same-words turns)
  identical transcripts: 200 of 200
  WER normalized, r0-gpu: 11.9%
  WER normalized, r1-gpu: 11.9%
```

**The registered breakdown, by pieces per reply.** 95 of 200 replies were a
single piece, and for those R1 is R0 (ratio 0.98). Replies that split reached
0.65, 0.46 and 0.31. R1 is faster (first audio 491 vs 660 ms; first sound 806
vs 973 ms) but not by the margin, or with the reliability, registered here.

**Predictions, scored.** S1 about 0.70: observed 0.744, and it was named the
criterion most likely to fail. S2 about 0.60: observed 0.655. **S3 r about 0.1:
observed 0.455, wrong.** S4 about 1.05: observed 1.030. Stalls under 5%: none.
**"About a third of replies will be one piece": 47.5%, wrong, and the root of
the failure.**

**What I got wrong.**
1. *The development data was not representative.* Replies to the probe clips (a
   caller's dictated requests) split 63% of the time; replies to eval-set
   utterances (audiobook sentences, conversation fragments) split 52%. S1–S3
   were set from the wrong distribution.
2. *Punctuation-only splitting* left every single-sentence reply unsplit. A
   forced split inside long sentences was considered and rejected before
   registration, to protect prosody.
3. *r = 0.07 on 30 clips was over-read.* Its own 95% interval is about ±0.36.

**Exploratory, after the registered analysis** (not evidence for or against
PR-002): among the 105 replies that split, first audio still correlates 0.57
with reply length, because longer replies open with longer first sentences; so
S3 would likely have failed even with fewer one-piece replies. One-piece share
by cohort: clean 52%, accented 30%, disfluent 50%, slow 58%. R0's re-run
reproduces `0af142a`: p50 660 vs 655 ms, median per-turn difference 10 ms, 200
of 200 replies and transcripts identical.

---

## PR-003 — Can a barge-in policy that reads the caller's words beat a duration threshold on a real voice?

**Registered:** 2026-09-20, by the commit that adds this entry.
**Author:** Ram Charan

### Background

The barge-in probe set (`probes/bargein/`, frozen at `e860999`): 20 recordings
of the author speaking over five synthesized agent lines, in a 2 × 2 design --
the agent should stop or keep going, and the caller's sound is short or long --
five per cell. Two of the "keep going" items are a real cough and a real laugh.

B0, a duration threshold, on these 20 (`6dbfb0c`; it reads no words, so it was
never locked): **10, 10, 11 and 8 wrong of 20** at 200, 400, 800 and 1500 ms.
The 8 comes with stops 1.5 to 5 s after the caller began. By design no duration
separates "hang on" (0.5 s) from "okay, yeah, that works for me" (2 s).

B1 (`3f64d14`) streams Whisper base.en partials every 200 ms, charges the
recognizer's compute time to the decision, and lets the caller's opening
decide: an acknowledgement holds the floor, anything else stops the agent. B2
(`7d11d75`) is B1, but ignores a transcript unless the sound behind it is
speech: the recognizer's own no-speech score under 0.25, at most 10 words per
second of sound, at least 25% of it voiced. The recognizer decodes greedily, so
a run repeats exactly (`97b3974`).

Development data, none of it the author's voice:

- `probes/bargein_dev`: 40 synthetic items, four Kokoro voices, labels exact by
  construction. B0 19 to 22 wrong of 40; B1 2 wrong; **B2 0 wrong**. B2's stops
  come 479 / 714 / 1100 ms after onset (fastest / middle / slowest of 20), with
  0 / 3 / 6 agent words said over the caller. Recognizer time charged at 70 ms.
- Ten real coughs and laughs from ESC-50 (CC BY-NC, not committed), all "keep
  going": B0 9 wrong, B1 9 wrong, **B2 2 wrong**, both long voiced laughs.
- Every threshold and word list was set on these two sets.

What was already known about the test set, and by whom:

- The author wrote the script and recorded it. The word lists in
  `bargein_policies.py` were written by an assistant **that had read the
  script**. All 18 spoken items open with a word those lists cover. They were
  drawn from general conversational usage, not copied from the script, but
  they cannot be called blind to it.
- Durations, entry times and loudness of all 20 takes were examined while
  labelling. B0's errors on them are known, item by item.
- **No policy that reads words has been run on the 20 recordings.** The scorer
  refuses without `--confirm PR-003`, and no transcript of them exists.
- One item is known to be hard: `sl05` is an interruption that opens with
  "yes". Nothing was tuned for it.

### Question

On the author's real voice, recorded over a talking agent, does B2 make fewer
wrong stop / keep-going decisions than the best duration threshold, and stop
soon enough to be of use?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen code** | `bargein_policies.py`, `bargein_asr.py`, `bargein.py`, and `context_for` and `score` in `bargein_eval.py`, are unchanged between this commit and the run. Reporting code and tests may change. The run's commit is recorded below. |
| **V2 · Frozen set** | `python -m s2slab.bargein verify` passes on the day of the run: every recording and label matches its SHA-256 in the manifest. |
| **V3 · One run, whole set** | All 20 items, the first complete run, with `--asr-latency-ms 70` and no transcript cache beforehand. The recognizer is deterministic, so a repeat must give the same table; a repeat that differs is reported as a finding. A crashed run is reported and repeated once, in full. |
| **V4 · Evidence** | The transcripts the policies saw (every partial, with its delivery time and no-speech score) and the per-item outcomes are saved to `runs/bargein/` in the commit that reports the result. |

### Primary outcome

Wrong decisions of 20, as `bargein_eval.score` decides them at this commit: a
missed interruption, or a stop on an item where the agent should keep going.
The policy under test is **`B2_speechlike`**. The comparator is
**`B0_duration_1500ms`**, the best of the four B0 thresholds on this set.

### Success criterion — both must hold

| | Criterion |
|---|---|
| **S1 · Fewer wrong decisions** | B2 is wrong on fewer items than B0 at 1500 ms, and an exact two-sided sign test on the items where the two disagree gives **p < 0.05**. |
| **S2 · Soon enough** | At least **8 of the 10** interruptions are stopped within **1500 ms** of the labelled onset. |

**What S1 demands.** B0 at 1500 ms is wrong on 8. With f items fixed and b
broken, p = 2 · P(X ≤ min(f, b)) for X ~ Binomial(f + b, ½): 5 fixed and none
broken gives 0.063, 6 and 0 gives 0.031, 8 and 1 gives 0.039, 7 and 1 gives
0.070. **S1 can hold only if B2 is wrong on at most 2 of 20.** That is what 20
paired items can show; a smaller improvement would be real and undetectable.

**Why 1500 ms B0.** It is the hardest comparator, chosen after its result was
seen, which favours B0. It is no usable system -- it cannot stop in under
1.5 s -- and S2 is what holds B2 to a usable speed. B0 at 1500 ms meets S2 on
0 of 10 by construction.

**Why 8 of 10 within 1500 ms.** On the dev set every stop came within 1100 ms.
A real voice is slower to recognize than a synthetic one, and `sl05` is
expected to wait for the 3 s cap, so two misses are allowed.

**No multiplicity correction:** success needs both criteria at once.

**If either fails,** PR-003 is reported as *not supported*, naming the
criterion, every wrong item, and what the recognizer wrote for it.

### Predictions, recorded but not criteria

- B2 wrong on **1 to 3** of 20. **S1 is the criterion most likely to fail:** a
  third error fails it.
- The likeliest errors: the real laugh `cs04`, if the recognizer writes words
  for it as it did for two ESC-50 laughs; and `cs01` "mm-hm", a quiet sound
  Whisper is known to turn into a stray word.
- The real cough `cs02` will be held correctly.
- `sl05` will be counted right but stopped at about +3000 ms, by the cap: the
  slowest hit, and one of S2's two allowed misses.
- S2: 8 or 9 of 10. Middle stop latency 800 to 1200 ms, slower than the dev
  set's 714 ms.
- B1 will be wrong on 1 or 2 more items than B2, the cough among them.

### Reported alongside, descriptively

All four B0 thresholds, `B1_words` and `B1_words_min1` from the same run and
the same transcripts; errors by cell and on the two noise items; every hit's
stop latency and the agent words said over the caller; what the recognizer
wrote for every item; one further run with the recognizer's time as measured
on the laptop instead of charged, which is not deterministic and is not a
criterion.

### Analysis plan

1. Commit this entry. Code under V1 is frozen from here.
2. Add a per-item listing to the scorer's output and a way to save outcomes; test on the dev set only.
3. `bargein verify`, then the one run (V2, V3). Save the evidence (V4).
4. Count S1's fixed and broken items from the per-item listing; record the output below, verbatim.
5. Only after step 4: any further look at the transcripts, labelled exploratory.

### What this cannot show

- **Not that the word lists generalize.** They were written with knowledge of
  the script, and every spoken opening is on them. This run tests the rest of
  the chain on a real voice: the causal VAD, streamed recognition of short
  real utterances, the timing, the speech check on a real cough and laugh. A
  blind test needs a script written by someone else.
- **One speaker, one microphone, one room, read from a script.** Acted
  interruptions are cleaner than real ones.
- **Two noise items.** B2's speech check is tested on one cough and one laugh.
- **Latency is modelled.** 70 ms per recognizer call is R0's measured ASR time
  on a T4, charged; no GPU runs here, and no audio device or network is in the
  loop. The labelled onset carries a hand-labelling error of tens of ms.
- **Nothing about the agent stopping well.** State truncation (`68b0255`) is
  built and unit-tested; no criterion here scores it.

### Deviations

- **2026-09-20, the run's commit.** The registered run was made from a working
  tree that had not been committed: `e707190` plus the reporting changes that
  were committed afterwards, unchanged, as `19d3ad7`. `git diff e707190 19d3ad7`
  touches only `bargein_eval.py` and its tests; `context_for` and `score` are
  unchanged, and the three other frozen files have zero changed lines. V1 is
  met in substance, but "the run's commit" did not exist when the run was made.
  The repeat run was made at `19d3ad7` and gave a byte-identical result file.
- **2026-09-20, verbatim output.** Step 4 of the analysis plan says the output
  is recorded below verbatim. The per-item table is eleven columns wide, so it
  is kept verbatim in the committed `runs/bargein/pr003-70ms-report.md`
  instead; everything else the run printed is below.
- **2026-09-20, registry row.** Its "Registered" cell was changed from "The
  commit that adds this entry" to that commit's hash, `e707190`, and its status
  updated, as for PR-002.

### Results

**Run:** 2026-09-20, laptop CPU (WSL), recognizer time charged at 70 ms a call,
all 20 items, seven policies, first and only registered run. `bargein verify`:
intact. No transcript cache of the set existed beforehand. Code: `19d3ad7` (see
Deviations). Evidence: `runs/bargein/pr003-70ms.json` (every outcome, every
partial) and `runs/bargein/pr003-70ms-report.md` (the run's complete output). A
repeat at `19d3ad7` with `--refresh` gave a byte-identical result file (`cmp`).

**Verdict: supported.** Both criteria hold, S1 with nothing to spare.

| | Criterion | Result | |
|---|---|---|---|
| **S1** | B2 wrong on fewer items than B0 at 1500 ms, exact two-sided sign test p < 0.05 | B2 wrong on **1** of 20, B0 on 8; fixed 8, broken 1; **p = 0.039** | holds |
| **S2** | at least 8 of 10 interruptions stopped within 1500 ms | **9 of 10**; the tenth, `sl05`, at +3102 ms | holds |

V2, V3 and V4 held; V1 held in substance, with the deviation recorded above.
Given the one broken item, **any second error would have failed S1** (8 fixed
and 2 broken: p = 0.11; 7 and 1: p = 0.07).

The run's output, verbatim, without the per-item table (which is in the report file):

```text
20 items from probes/bargein · confirms PR-003 · recognizer time: charged at 70 ms a call

| policy | stop-long | stop-short | continue-short | continue-long | wrong | missed | false barge-in | of which noise | stop latency of the hits, ms | agent words said over the caller |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0_duration_200ms | 0 | 0 | 5 | 5 | **10 of 20** | 0 | 10 | 2 of 2 | 62 / 202 / 275  (fastest / middle / slowest of 10 hits) | 0 / 0 / 1 |
| B0_duration_400ms | 0 | 0 | 5 | 5 | **10 of 20** | 0 | 10 | 2 of 2 | 262 / 402 / 475  (fastest / middle / slowest of 10 hits) | 0 / 1 / 2 |
| B0_duration_800ms | 0 | 4 | 2 | 5 | **11 of 20** | 4 | 7 | 2 of 2 | 762 / 842 / 875  (fastest / middle / slowest of 6 hits) | 2 / 2 / 5 |
| B0_duration_1500ms | 0 | 5 | 0 | 3 | **8 of 20** | 5 | 3 | 0 of 2 | 1462 / 1575 / 4962  (fastest / middle / slowest of 5 hits) | 4 / 4 / 15 |
| B1_words | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 642 / 1122 / 2662  (fastest / middle / slowest of 10 hits) | 1 / 4 / 10 |
| B1_words_min1 | 0 | 0 | 2 | 1 | **3 of 20** | 0 | 3 | 2 of 2 | 282 / 782 / 2662  (fastest / middle / slowest of 10 hits) | 1 / 3 / 10 |
| B2_speechlike | 0 | 0 | 0 | 1 | **1 of 20** | 0 | 1 | 0 of 2 | 642 / 1122 / 3102  (fastest / middle / slowest of 10 hits) | 1 / 4 / 11 |

B1_words: cs02 (false barge-in at +640 ms), cs04 (false barge-in at +320 ms), cl02 (false barge-in at +1882 ms)

B1_words_min1: cs02 (false barge-in at +640 ms), cs04 (false barge-in at +320 ms), cl02 (false barge-in at +1682 ms)

B2_speechlike: cl02 (false barge-in at +2102 ms)

B0_duration_200ms: 10 of 10 interruptions stopped within 1500 ms
B0_duration_400ms: 10 of 10 interruptions stopped within 1500 ms
B0_duration_800ms: 6 of 10 interruptions stopped within 1500 ms
B0_duration_1500ms: 1 of 10 interruptions stopped within 1500 ms
B1_words: 9 of 10 interruptions stopped within 1500 ms
B1_words_min1: 9 of 10 interruptions stopped within 1500 ms
B2_speechlike: 9 of 10 interruptions stopped within 1500 ms

B2_speechlike against B0_duration_1500ms: wrong on 1 against 8 of 20
  fixed 8: cl01, cl03, cl05, ss01, ss02, ss03, ss04, ss05
  broken 1: cl02
  exact two-sided sign test on the 9 items that differ: p = 0.039
```

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| B2 wrong on 1 to 3 of 20 | 1. Right. |
| S1 is the criterion most likely to fail | It held, at p = 0.039, and was the close one: any second error would have failed it. |
| Likeliest errors: the laugh `cs04` and "mm-hm" `cs01` | **Wrong.** Both were held. The one error, `cl02`, was not foreseen. |
| The cough `cs02` will be held | Right. The recognizer wrote "I'm sorry. I'm sorry. I'm sorry…" for it; B1 stopped, B2 did not. |
| `sl05` counted right, stopped at about +3000 ms by the cap | Right number, **wrong reason**: +3102 ms, but the 3 s cap never fired. See below. |
| S2: 8 or 9 of 10; middle stop latency 800 to 1200 ms | 9 of 10; 1122 ms. Right. |
| B1 wrong on 1 or 2 more items than B2, the cough among them | 2 more: the cough and the laugh. Right. |

**What the registration got wrong**

- *"B0 at 1500 ms meets S2 on 0 of 10 by construction."* It meets it on 1:
  `sl02` at +1462 ms. The causal VAD hears the caller about 40 ms before the
  hand-labelled onset, so a latency counted from the label can fall under the
  threshold. S2 is unaffected.
- The error that happened was not among those imagined. `cl02` is "okay, yep,
  doing that now", said with a pause after "yep" long enough for the VAD to
  hear two sounds. B2 judges each sound by its opening, and the second opens
  with "doing": not an acknowledgement, so it stopped.
- `ss01`, "hang on", was transcribed "Hand-bond." and stopped correctly only
  because anything that is not an acknowledgement stops the agent. Right
  decision, no credit to the recognizer.

**Reported alongside, descriptively**

- B2's ten stops, ms after the labelled onset: 642, 782, 882, 955, 1102, 1122,
  1122, 1142, 1482, 3102. Agent words said over the caller: 1 to 11, middle 4.
- The two noise items: B0 at 200 to 800 ms and B1 stop for both; B2 for
  neither. For the laugh the recognizer first wrote "I'm going to go ahead and
  get" (no-speech 0.52). B2's speech check changed exactly these two decisions
  and no other.
- **Recognizer time as measured on the laptop** (`runs/bargein/pr003-measured.json`;
  Whisper base.en int8 takes about half a second a call on this CPU): the same
  decisions item for item -- B2 wrong on 1, B1 on 3, the same items, p = 0.039
  -- but B2's stops come at 1102 / 1662 / 3402 ms and only **4 of 10 within
  1500 ms**. S1 does not depend on the recognizer's speed; **S2 does**, and
  holds only at a GPU's. Not deterministic, not a criterion.
- The per-item table's last column shows each item's *last* transcript. For
  five items that is an empty transcript of a breath after the caller had
  finished. A fault of the listing; the JSON holds every partial.

**Exploratory, after the result**

`sl05` ("yes it is, um, let me think…") was rescued by the mechanism that broke
`cl02`. "Yes, it is." was held as an acknowledgement; after the "um", "Let me
think" arrived as a new sound, was judged by its own opening, and stopped the
agent. Treating a new sound as a new opening fixed one item and broke one. A
rule that let a sound shortly after an acknowledgement inherit it would fix
`cl02` and lose `sl05` to the 3 s cap. That is untested, and these 20
recordings cannot test it: B2's errors on them are now known, so they are
spent as a test set for any policy of this family. A successor needs new
recordings and its own registration.

---

## PR-004 — Does a slow-tier recognizer hear enough more to be worth the wait?

**Registered:** 2026-09-21, by the commit that adds this entry.
**Author:** Ram Charan

### Background

R1 on the frozen 200-clip eval set, Modal T4 (PR-002 run, `b107927`): first
audio p50 491 ms; normalized WER 11.9% overall -- clean 5.6%, **accented 26.5%**,
disfluent 9.7%, slow 5.6%. A misheard word becomes the answer: on probe clip
f02 R1 hears "half pass too" and replies "half time".

R2 (`4920228`, `a84a3e3`) is the v2 plan's two-tier cascade. The fast tier,
Whisper base.en, serves barge-in and was tested in PR-003. The slow tier, a
larger Whisper, transcribes the finished turn, and its words drive the reply.
Inside `respond()` R2 is R1 with the slow tier's recognizer: the LLM, prompt,
chunker and voice are R1's own, and a test holds that nothing but the
recognizer's size differs. Two slow tiers are built: small.en (`r2-gpu`) and
large-v3-turbo (`r2-gpu-turbo`). Nothing in either is tuned.

Development data, **the 30 probe clips only, one speaker**:

- Modal T4, one container, `4920228-dirty` (dirty only by the edits committed
  as `a84a3e3`): asr p50 65 / 123 / 376 ms for base.en / small.en / turbo;
  first audio 512 / 605 / 900 ms; no playback stalls. First audio rose by more
  than the recognizer's own time (+93 against +58 ms; +388 against +312 ms).
- Laptop CPU: turbo needs 6.9 s a turn; the slow tier is a GPU design.
- One clip, f02: small.en hears "half pass 2"; turbo hears "half past two" and
  the reply is "2:30 PM", right for the first time in this project.

What was already known about the eval set: all of R1's and R0's results on it,
turn by turn. **Neither slow tier has been run on any eval-set clip.** That a
larger Whisper makes fewer errors in general is not in doubt; what is
registered is how many fewer on this set, whether they come where R1 fails,
and what the caller waits for them.

### Question

On the same clips, LLM, voice and GPU, does replacing R1's recognizer with
large-v3-turbo cut the word error rate substantially -- overall and on
accented speech -- while the reply still starts within a second?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Same run** | `r1-gpu`, `r2-gpu` and `r2-gpu-turbo`, in that order, over all 200 clips inside **one Modal container call**, same T4, same image: `modal run modal_app/r0_bench.py --rung r1-gpu,r2-gpu,r2-gpu-turbo --evalset --tag pr004`. R1 is re-run; its saved PR-002 run is not the comparator. |
| **V2 · Frozen code** | `r2.py`, `r1.py`, `r0.py`, `rungs.py`, `chunking.py`, `wer.py` and `evalrun.py` are unchanged between this commit and the run. `modal_app/`, the analysis script and tests may change. |
| **V3 · Clean tree** | The run is made from a committed, clean working tree: the commit stamped into the result files must not end in `-dirty`. A dirty run is not the registered run; it is reported, and the run is repeated once from a clean tree. |
| **V4 · Whole set, one run** | All 200 clips per rung, the first complete run. No clip is dropped and no run is repeated to improve a number; a crashed run is reported as such and repeated once, in full. The container verifies the eval set against its manifest before any model loads. |

### Primary outcomes

- **Normalized WER**, as `wer.score` computes it at this commit (Whisper's
  English normalizer, corpus-level: total errors over total reference words),
  overall and for the accented cohort. The statistic is the difference
  `r2-gpu-turbo` minus `r1-gpu`, in percentage points.
- **`first_audio_ms`** of `r2-gpu-turbo`: from the rung receiving the finished
  turn to its first piece of reply audio existing; p50 over the 200 clips.

### Success criterion — all three must hold

| | Criterion |
|---|---|
| **S1 · Hears more** | Overall normalized WER difference is **−3.0 points or lower**, and the upper end of its 95% interval is below 0. |
| **S2 · Hears more where R1 fails** | Accented-cohort normalized WER difference is **−5.0 points or lower**, and the upper end of its 95% interval is below 0. |
| **S3 · Still starts within a second** | The upper end of the 95% interval of `r2-gpu-turbo`'s first-audio p50 is **at or below 1000 ms**. |

**Confidence.** A paired bootstrap resamples clip ids with replacement (10,000
resamples, seed 0, percentile interval; `compare.resamples`): all 200 for S1
and S3, the 50 accented clips for S2. On each resample the corpus WER of both
rungs is recomputed from the same clips, and their difference taken; for S3
the p50 is recomputed.

**No multiplicity correction:** success needs all three at once.

**Why −3 and −5 points.** The slow tier adds about a third of a second before
the caller hears anything. A quarter off the overall error rate is the least
that would justify that, and accent is where R1 fails worst, so a gain that
missed the accented cohort would not be the gain the tier exists for.

**Why 1000 ms.** First sound follows first audio by Kokoro's 320 ms of leading
silence. Past one second to first audio, the caller waits over 1.3 s to hear a
word, and a faster recognizer would have been the better trade.

**If any of S1–S3 fails,** PR-004 is reported as *not supported*, naming the
criterion. `r2-gpu` (small.en) is reported beside it and is **not** promoted
to a pass: it is a different operating point, described, not tested.

### Predictions, recorded but not criteria

- Turbo's overall WER about 7.5% (−4 to −5 points). Accented about 17% (−8 to
  −10 points), with a wide interval from 50 clips. Both criteria hold.
- Turbo's first-audio p50 860 to 920 ms. **S3 is the criterion most at risk:**
  the eval set's replies open differently from the probe clips', which is what
  sank PR-002's thresholds.
- small.en: −1 to −2 points overall for about +90 ms.
- The two tiers' transcripts agree exactly (normalized) on 35 to 45% of clips
  for base.en against turbo: too few for "start the LLM on the fast tier's
  words" to pay.
- R1's re-run reproduces its PR-002 run: WER 11.9%, first audio within 20 ms.
- No playback stalls.

### Reported alongside, descriptively

- `r2-gpu` (small.en): the same three statistics, against `r1-gpu`.
- Per-cohort WER, raw and normalized, for all three rungs; first-audio p90;
  first sound; total; stalls; R1's re-run against its PR-002 run.
- **Where the added wait comes from, turn by turn:** the change in first audio
  minus the change in asr time, at the median, split by whether the reply's
  words changed.
- **Pairwise error overlap** (the plan's diagnostic before fusion): a clip is
  wrong for a tier if its normalized transcript differs from the normalized
  reference; overlap is clips wrong for both over clips wrong for either.
  Above about 70%, fusion will not pay.
- **Agreement:** the share of clips whose normalized transcripts are identical
  between tiers. **Reply change:** the share of clips whose reply words differ
  from R1's, split by whether the raw transcript differed.
- The run's cost.

### Analysis plan

1. Commit this entry. Code under V2 is frozen from here.
2. Add the analysis script; test it on fake runs only.
3. Run once on Modal from a clean tree (V1, V3, V4). Save the three JSONs to `runs/`.
4. Run the analysis script; record its output below, verbatim.
5. Only after step 4: any further look at the transcripts or replies, labelled exploratory.

### What this cannot show

- **Not that the reply is better.** WER scores what was heard. Nothing here
  scores what was said; "reply change" counts differences, not improvements.
- **Not accent alone.** The accented cohort is spontaneous conversation and
  the clean cohort is read audiobooks; a gain there mixes accent with style.
- **Not a two-tier system running live.** The fast tier is not inside
  `respond()`; its job was tested separately (PR-003). No speculation, no
  fusion, no partials reaching a UI.
- **Order and one run.** R1 runs first, turbo last, in one container; drift
  between them would land in S3. Run-to-run spread on Modal has been measured
  once (10 ms at the median).
- **One GPU type, one LLM, 200 clips.** Fifty accented clips give a wide interval.

### Deviations

- **2026-09-21, registry row.** Its "Registered" cell was changed from "The
  commit that adds this entry" to that commit's hash, `415ddc7`, and its status
  updated, as for PR-002 and PR-003.
- **2026-09-21, who ran the analysis.** Step 4's command was first run by the
  assistant, read-only, on the three saved result files; the author then ran
  the same command. It is deterministic (seed 0), and the two outputs are
  identical.

### Results

**Run:** 2026-09-21, commit `91048b3`, clean tree, Modal T4 (driver 580.95.05), one
container call: `r1-gpu`, then `r2-gpu`, then `r2-gpu-turbo`, all 200 clips each,
first and only run. 666 s inside the function, about \$0.119. Files:
`runs/r1-gpu/`, `runs/r2-gpu/` and `runs/r2-gpu-turbo/evalset-modal-t4-pr004.json`.
V1 to V4 held; zero changed lines in the seven frozen files since `415ddc7`.

**Verdict: not supported.** S1 and S3 hold; **S2 fails**: on accented speech the
slow tier cuts the error rate by 4.3 points, not the 5 registered, with an
interval from −7.4 to −1.5.

Output of `python -m s2slab.tiers`, verbatim:

```text
PR-004: r2-gpu-turbo against r1-gpu
  r1-gpu: commit 91048b3 · Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo · run 2026-09-21
  r2-gpu-turbo: commit 91048b3 · Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo · run 2026-09-21
  paired bootstrap: 10000 resamples of 200 clips, seed 0

V1 same run          PASS  Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo
V3 clean tree        PASS  commit 91048b3
S1 hears more        PASS  normalized WER, slow minus fast: -3.389 [95% CI -4.487, -2.345] points; needs <= -3.0 and upper end < 0
S2 accented          FAIL  accented cohort (50 clips): -4.348 [95% CI -7.379, -1.537] points; needs <= -5.0 and upper end < 0
S3 within a second   PASS  first-audio p50 888.950 [95% CI 870.950, 910.200] ms; needs upper end <= 1000

NOT SUPPORTED: failed S2 accented

Descriptive, not criteria:
  ms, p50 / p90            r2-gpu-turbo           r1-gpu
  first audio               889 /  1191      574 /   864
  total                    1160 /  1814      806 /  1357
  first sound p50                  1204              893
  asr p50                           393               81
  WER normalized                   8.6%            11.9%
  turns that stalled                  0                0
  the added wait, per turn, at the median:
    first audio +310 ms = recognizer +310 ms + everything after it -2 ms
    everything after it: -1 ms on the 66 turns whose reply kept its words, -6 ms on the 134 whose reply changed
  clips with any error: both tiers 100, only r1-gpu 28, only r2-gpu-turbo 5, neither 67
    error overlap 75% (both / either); the plan's rule: above about 70%, fusing the tiers will not pay
  the tiers agree word for word (normalized) on 97 of 200 clips
  the reply's words changed on 134 of 200 turns: 134 of the 154 whose transcript differed, 0 of the 46 whose transcript was identical

Described only, not tested: r2-gpu against r1-gpu
  r1-gpu: commit 91048b3 · Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo · run 2026-09-21
  r2-gpu: commit 91048b3 · Modal Tesla T4, 580.95.05, one container: r1-gpu then r2-gpu then r2-gpu-turbo · run 2026-09-21
  paired bootstrap: 10000 resamples of 200 clips, seed 0

S1 hears more              normalized WER, slow minus fast: -2.296 [95% CI -3.348, -1.294] points; needs <= -3.0 and upper end < 0
S2 accented                accented cohort (50 clips): -2.676 [95% CI -5.653, -0.109] points; needs <= -5.0 and upper end < 0
S3 within a second         first-audio p50 670.600 [95% CI 645.000, 713.500] ms; needs upper end <= 1000

Descriptive, not criteria:
  ms, p50 / p90                  r2-gpu           r1-gpu
  first audio               671 /   969      574 /   864
  total                     898 /  1509      806 /  1357
  first sound p50                   988              893
  asr p50                           167               81
  WER normalized                   9.6%            11.9%
  turns that stalled                  0                0
  the added wait, per turn, at the median:
    first audio +92 ms = recognizer +84 ms + everything after it +3 ms
    everything after it: +2 ms on the 87 turns whose reply kept its words, +14 ms on the 113 whose reply changed
  clips with any error: both tiers 112, only r1-gpu 16, only r2-gpu 8, neither 64
    error overlap 82% (both / either); the plan's rule: above about 70%, fusing the tiers will not pay
  the tiers agree word for word (normalized) on 104 of 200 clips
  the reply's words changed on 113 of 200 turns: 113 of the 128 whose transcript differed, 0 of the 72 whose transcript was identical

WER normalized (raw), by cohort:
  r1-gpu: clean 5.6% (6.3%), accented 26.5% (29.0%), disfluent 9.7% (12.1%), slow 5.6% (6.1%), all 11.9% (13.5%)
  r2-gpu-turbo: clean 2.4% (3.4%), accented 22.2% (25.5%), disfluent 7.4% (9.7%), slow 1.8% (2.5%), all 8.6% (10.4%)
  r2-gpu: clean 3.3% (3.5%), accented 23.9% (27.0%), disfluent 7.6% (9.9%), slow 3.5% (3.7%), all 9.6% (11.1%)

r1-gpu now (commit 91048b3) against its earlier run (commit b107927), 200 clips:
  identical transcripts 200, identical replies 200; first-audio p50 574 vs 491 ms, median per-turn difference +94 ms
```

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Turbo's overall WER about 7.5% (−4 to −5 points) | **Wrong.** 8.6%, −3.4 points. |
| Accented about 17% (−8 to −10 points); both WER criteria hold | **Wrong, and the reason PR-004 failed.** 22.2%, −4.3 points; S2 failed. |
| First-audio p50 860 to 920 ms | Right: 889 ms. |
| S3 is the criterion most at risk | **Wrong.** S3 held with 90 ms to spare; S2 failed. |
| small.en: −1 to −2 points overall for about +90 ms | −2.3 points (a little better than predicted) for +92 ms. |
| The tiers agree exactly on 35 to 45% of clips | 48.5% (97 of 200): a little more than predicted. |
| R1's re-run reproduces its PR-002 run: WER 11.9%, first audio within 20 ms | WER and all 200 transcripts and replies identical; **timing wrong: +94 ms.** |
| No playback stalls | Right: 0 of 600 turns. |

**What the registration got wrong**

1. **It borrowed its optimism from public benchmarks.** Larger Whisper models
   roughly halve the error rate on read speech, and they did here (clean 5.6%
   to 2.4%, slow 5.6% to 1.8%). The accented cohort is spontaneous
   conversation, and there the bigger model cut errors by a sixth. −5 points
   was set from the wrong expectation, not from anything measured.
2. **It treated a Modal T4 as one machine.** The same R1, saying the same
   words, was 17% slower than three days earlier (asr 81 against 68 ms, llm
   432 against 374, tts 300 against 231). The image had been rebuilt and the
   host was another one. The paired comparison inside one container is
   unaffected; S3's absolute bar was met on a slow day. Numbers from
   different runs are not comparable to within 20 ms, or to within 80.

**Exploratory, after the registered analysis — not evidence for or against PR-004**

- On the 50 accented clips turbo makes fewer errors on 21, the same number on
  21 and more on 8. Its 199 remaining errors are spread out: the worst five
  clips hold 49 of them. No handful of clips explains the cohort.
- The smoke test suggested first audio rose by more than the recognizer's own
  time. Turn by turn it does not: +310 ms of recognizer, −2 ms of everything
  else. Medians of 30 clips do not add.
- The reply changed on 134 of 200 turns, though the tiers' words differ on
  only 103: a 0.5B LLM answers differently when only punctuation or casing
  changes. Whether the new replies are better was not measured.
- Starting the LLM on the fast tier's transcript and keeping the reply when
  the slow tier agrees would save the 310 ms on at most the 46 turns whose raw
  transcripts are identical, under a quarter. Not worth building as it stands.
  
---

## PR-005 — Does a $0 affect side-channel let a blind cascade hear that the caller is upset?

**Registered:** 2026-09-21, by the commit that adds this entry.
**Author:** Ram Charan

### Background

The v2 plan (R3, week 3): routing eGeMAPS tags around the text bottleneck is
"a poor man's R3", and the question is "how much of R3's benefit does a $0
side-channel recover?", tested with a shuffled-tag control.

The prosody probe set (`ac8b114`): 60 same-actor, same-sentence pairs from
CREMA-D, one clip neutral and one loaded (sad, fearful or angry), chosen by
what 2,443 raters heard. 21 dev pairs from 14 actors; **39 test pairs from 29
other actors, locked** behind this registration. On the dev pairs R1's
recognizer gives the same transcript for both clips of 18 of 21 pairs: the
words do not carry the difference.

The side-channel (`8548b87` → `ffbc32d`):

- **The tagger** (`affect.py`, `probes/prosody/tagger.json`): 88 eGeMAPS
  functionals (openSMILE) into a class-balanced softmax regression, fitted on
  398 clips from the **16 dev-side actors who have no pair in the set** — no
  training voice is a dev-pair or test-pair voice. One clip in, one word out:
  neutral, sad, fearful or angry. It never sees a clip's twin. On a held-out
  training voice: neutral 69%, sad 53%, fearful 45%, angry 76% tagged right.
- **The probe** (`sidechannel.py`): R1's own Qwen2.5-0.5B, unchanged, is shown
  the transcript and, in one sentence, the tagger's word, and asked *"Judging
  by their tone of voice, is the caller upset?"*. The answer is the next-token
  probability of "Yes" against "No", read off directly: deterministic, and it
  cannot fail to parse.
- **Four conditions**, same clips and LLM: *none* (R1 as it is), *tagger* (the
  side-channel), *oracle* (the acted emotion as the word: what a perfect tagger
  would buy), *shuffled* (the tagger's words dealt to the wrong clips, 1000
  seeded deals: the control).
- **A pair is answered right** when the loaded clip gets p(Yes) > 0.5 and its
  neutral twin gets p(Yes) ≤ 0.5. "Ranked right" (the loaded clip merely
  higher) is reported beside it.

Development data, **the 21 dev pairs only** — answered right · ranked right, of 7:

| | SAD | FEA | ANG |
|---|---|---|---|
| none | 0 · 2 | 0 · 2 | 0 · 2 |
| tagger | 1 · 2 | 6 · 6 | 4 · 5 |
| oracle | 7 · 7 | 7 · 7 | 7 · 7 |
| shuffled (mean of 1000 deals) | 1.7 · 3.1 | 1.8 · 3.1 | 1.7 · 3.0 |

What was tuned, and on what:

- **The prompt wording was chosen on the dev pairs with the oracle tag.** Four
  wordings were tried; three left the 0.5B model answering "No" to nearly
  everything (3 to 5 of 21 answered right, though all ranked 21 of 21). The
  wording that states the tone in a sentence answered 21 of 21. It is fixed in
  the code. The test pairs had no part in it.
- The tagger's regularization (L2 = 1.0) was fixed before any data was seen; a
  sweep on the training actors afterwards found it made no difference (60.7 to
  62.8% held-out mean recall across 1 to 1000) and it was left as it was.
- **No test pair has been tagged, transcribed or scored**: `prosodyset.load`
  refuses the test split without `confirm`, and no cache for it exists.
- Known in advance: sadness is nearly inaudible to every tool so far — to
  CREMA-D's own raters (60% agreement at best for these pairs), to pitch and
  level, and to the tagger (1 of 7 on dev). **The sadness criterion below is
  expected to fail**, and is registered so that the failure is on record.

### Question

On 39 pairs of voices the tagger and the author have never heard, does telling
R1's LLM one word about the caller's tone let it tell an upset caller from the
same caller saying the same words calmly — for fear and for anger — better
than the same words dealt at random would?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | `probes/prosody/tagger.json`, `src/s2slab/affect.py`, and in `src/s2slab/sidechannel.py` the constants `PROBE_SYSTEM`, `QUESTION`, `N_SHUFFLES`, and the functions `user_text`, `YesNoScorer`, `score_all`, `_pair_right` and `evaluate`, are unchanged between this commit and the run. Reporting code (`format_report`, `run`'s output, any per-pair listing or saving added for step 2) and tests may change. |
| **V2 · Untouched test pairs** | No `results/affect/pyes-test.json` or `results/affect/test.npz` exists before the run, and `prosodyset verify` passes on the day of the run. |
| **V3 · One run, whole set** | All 39 test pairs, the first complete run, `python -m s2slab.sidechannel test --confirm PR-005`, from a clean tree. The LLM's answers are one forward pass each and the deals are seeded, so a repeat must reproduce the table exactly; a repeat that differs is reported as a finding. |
| **V4 · Evidence** | Every clip's transcript, tag and p(Yes) under every word, and the per-pair outcomes, are saved to `runs/prosody/` in the commit that reports the result. |

### Primary outcome

Per emotion, over its 13 test pairs: the number of pairs **answered right**
under the *tagger* condition, as `sidechannel.evaluate` counts it at this
commit; and the same count under each of the 1000 shuffled deals.

### Success criterion

For each emotion, the permutation p-value is the share of the 1000 seeded
shuffled deals whose answered-right count is **at least** the tagger's (a
one-sided exact test against the control; a p of 0.000 is reported as
< 0.001).

| | Criterion |
|---|---|
| **S1 · Fear** | tagger answered-right count ≥ 8 of 13, and p < 0.05 |
| **S2 · Anger** | tagger answered-right count ≥ 7 of 13, and p < 0.05 |
| **S3 · Sadness** | tagger answered-right count ≥ 7 of 13, and p < 0.05 — **registered, reported, and expected to fail; it does not decide the verdict** |

**PR-005 is supported if S1 and S2 both hold**, and *not supported* otherwise,
naming the criterion. S3 is reported in the same table whatever it says.

**Why a count as well as a p.** With 13 pairs a lucky control can score 4 or 5;
a p under 0.05 against it could be reached with 6 right. The counts are the
least that would matter: over half the pairs for anger, more than half for
fear, where the tagger is stronger.

**Why the control is the comparator and not "none".** *None* answers no pair
right by construction — both clips get the same words — so beating it says
nothing. Beating the shuffled control says the *voice* carried the answer, not
the prompt's wording or the LLM's lean.

**No multiplicity correction** across S1 and S2: success needs both.

### Predictions, recorded but not criteria

- Fear: 10 to 11 of 13 answered right; shuffled about 3.3; p < 0.01. Holds.
- Anger: 7 to 9 of 13; shuffled about 3.3; **S2 is the criterion most at
  risk** — at 7 of 13 the p may land near 0.05.
- Sadness: 2 to 3 of 13, no better than shuffled. **S3 fails.**
- *None* answers 0 pairs right and says Yes to most clips (dev: 38 of 42).
  *Oracle* answers 12 to 13 of 13 for every emotion: the LLM layer loses
  nothing, and every miss under *tagger* is the tagger's.
- Ranked right runs a little above answered right, by at most 2 pairs per emotion.
- Most tagger misses on anger are neutral clips tagged sad or fearful (as on
  dev), not angry clips tagged neutral.

### Reported alongside, descriptively

Ranked-right counts; the *none* and *oracle* rows; p(Yes) medians by the word
shown; the tagger's own told-apart counts on the test pairs; every pair the
side-channel got wrong, with both tags and both p(Yes); transcripts that differ
between the two clips of a pair.

### Analysis plan

1. Commit this entry. The frozen code is frozen from here.
2. Add the permutation p-value, a per-pair listing and a `--save` to the
   reporting; test them on made-up scores only.
3. `prosodyset verify`, confirm no test cache exists, then the one run (V2, V3). Save the evidence (V4).
4. Record the output below, verbatim.
5. Only after step 4: any further look at the test pairs, labelled exploratory.

### What this cannot show

- **Not what R3 buys.** This is one point on the plan's axis (R1 blind → R1 +
  tags → R3). The comparison with R3 needs R3.
- **Not that the reply gets better.** The probe is an explicit yes/no
  question; nothing scores a spoken reply's tone.
- **Not a live system.** The tag is computed from the whole finished clip; no
  streaming, no per-caller baseline, no latency measured.
- **Acted emotion, one corpus, 13 pairs per emotion.** Large effects only.
  Sadness is a proxy the tools cannot hear, and the negative result there says
  so about the tools, not about sadness.
- **The wording was chosen for this LLM.** A larger LLM might answer the
  bracketed tag that this one ignored; the finding that "a 0.5B model needs
  the tone stated as a sentence" is a finding about Qwen2.5-0.5B.

### Deviations

- **2026-09-21, registry row.** Its "Registered" cell was changed from "The
  commit that adds this entry" to that commit's hash, `44a9a5f`, and its status
  updated, as for the earlier entries.
- **2026-09-21, verbatim output.** The per-pair table (39 rows, 8 columns) is
  kept verbatim in the committed `runs/prosody/pr005-report.md` rather than
  below; everything else the run printed is below.

### Results

**Run:** 2026-09-21, commit `4b50017`, clean tree, laptop CPU; `prosodyset
verify` intact; no cache of the test pairs existed beforehand (V2); the 39
test pairs, first and only run: `python -m s2slab.sidechannel test --confirm
PR-005 --save runs/prosody/pr005.json`. $0. Evidence: `runs/prosody/pr005.json`
(every transcript, tag, p(Yes) under every word, outcome, and the 1000 deal
counts) and `runs/prosody/pr005-report.md` (the complete output). V1 held:
zero changed lines in `affect.py` and `tagger.json` since `44a9a5f`, and the
frozen functions of `sidechannel.py` unchanged (the reporting commit
`4b50017` added below them only).

**Verdict: not supported.** S2 holds; **S1 fails by count**: fear was
answered right on 7 of 13 pairs, one under the registered 8, although it beat
the shuffled control (p = 0.015). S3, expected to fail, held.

Output of the run, verbatim, without the per-pair table (which is in the
report file):

```text
39 test pairs · confirms PR-005 · "Judging by their tone of voice, is the caller upset?" · shuffled = mean of 1000 deals, seed 20260921

| condition | SAD: answered right · ranked right (of 13) | FEA: answered right · ranked right (of 13) | ANG: answered right · ranked right (of 13) |
|---|---|---|---|
| none | 0 · 3 | 0 · 5 | 1 · 8 |
| tagger | 7 · 10 | 7 · 8 | 11 · 13 |
| oracle | 13 · 13 | 13 · 13 | 13 · 13 |
| shuffled | 3.2 · 6.1 | 3.1 · 5.6 | 3.1 · 6.8 |

with no tag the LLM says Yes to 73 of 78 clips (median p_yes 0.78); it puts a median 1.00 of its probability on Yes or No
median p_yes by the word shown: neutral 0.10, sad 0.91, fearful 0.79, angry 0.82

| criterion | pairs answered right | shuffled deals at or above it | permutation p | registered bar | |
|---|---:|---:|---:|---|---|
| S1 fear | 7 of 13 | 15 of 1000 | p = 0.015 | at least 8 and p < 0.05 | FAIL |
| S2 anger | 11 of 13 | 0 of 1000 | p < 0.001 | at least 7 and p < 0.05 | PASS |
| S3 sadness | 7 of 13 | 25 of 1000 | p = 0.025 | at least 7 and p < 0.05 | PASS (reported, not deciding) |

PR-005: NOT SUPPORTED: failed fearful

the tagger alone told apart (neutral clip tagged neutral, loaded clip tagged anything else): SAD 7 of 13, FEA 7 of 13, ANG 11 of 13
```

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Fear 10 to 11 of 13, p < 0.01; holds | **Wrong.** 7 of 13, p = 0.015; failed by count. |
| Anger 7 to 9 of 13; S2 the criterion most at risk | **Wrong in the other direction.** 11 of 13, p < 0.001; the safest of the three. |
| Sadness 2 to 3 of 13, no better than shuffled; S3 fails | **Wrong.** 7 of 13, p = 0.025; S3 held. |
| *None* answers 0 pairs right and says Yes to most clips | Right: 1 of 39 pairs; Yes to 73 of 78 clips. |
| *Oracle* 12 to 13 of 13 per emotion: the LLM layer loses nothing | Right: 13 of 13 for all three. The side-channel equals the tagger's told-apart count exactly. |
| Ranked right at most 2 above answered right | Right for anger (13 against 11) and fear (8 against 7); sadness 10 against 7. |
| Most anger misses are neutral clips tagged sad or fearful | Right, and it holds for every emotion: 12 of the tagger's 15 misses are neutral clips tagged emotional. |

**What the registration got wrong**

1. **It read the 21 dev pairs as if they were the population.** Sadness at 1 of
   7 and anger at 4 of 7 on dev were seven voices each. On the thirteen test
   voices the tagger heard sadness 10 times out of 13 and anger 13 of 13; its
   failures were not the emotions but particular *neutral* voices it takes for
   upset ones. The predictions should have been made per voice, with the
   uncertainty that seven voices carry; they were made per emotion, with none.
2. **The fear bar was set from the dev score, not from the mechanism.** With
   the tagger's misses being neutral clips, an emotion's count is set by how
   many of its 13 *neutral* twins the tagger mishears -- a property of the
   actors drawn, not of the emotion. A bar of 8 of 13 sat on the wrong side of
   that lottery for fear and the right side for the other two.

**Reported alongside, descriptively**

- *None* ranked right on 3 / 5 / 8 pairs: with no tag, p(Yes) differs between
  the clips of a pair only when the transcripts differ, and the recognizer
  heard the same words in **22 of 39** pairs (18 of 21 on dev). It answered 1
  pair right.
- The tagger's misses on the 39 pairs: **12 neutral clips tagged emotional**
  (sad 5, angry 5, fearful 2), **3 loaded clips tagged neutral** (two fearful,
  one sad).
- p(Yes) by the word shown, medians: neutral 0.10, sad 0.91, fearful 0.79,
  angry 0.82: the LLM answers the tag it is given.

**Exploratory, after the registered analysis**

The tagger's remaining problem is speaker independence: it has no idea how a
given person sounds when calm, so five of the 39 calm clips read as sad and
five as angry. Every one of those twelve pairs would be answered right if the
tag were computed *relative to the caller's own earlier speech* -- the
per-caller baseline parked in M15's design. That is the next experiment for
this side-channel, on a fresh set of pairs: these 39 are now spent.


---

## PR-006 — Does a speech-LLM that hears the audio answer the question the cascade cannot?

**Registered:** 2026-09-22, by the commit that adds this entry.
**Author:** Ram Charan

### Background

The v2 plan, R3: "Ask something answerable only from prosody ... R1 physically
cannot answer -- the information was destroyed at the ASR boundary. R3 can.
That is the input bottleneck demonstrated in one prompt." And: "how much of
R3's benefit does a $0 side-channel recover?" -- three points on one axis.

Two of the three are measured, on the same 39 locked prosody pairs and the
same question, under PR-005 (`d1a1b68`), pairs answered right · ranked right
of 13 (SAD · FEA · ANG):

    R1 blind        transcript only                    0 · 3    0 · 5    1 · 8
    R1 + tags       transcript + the tagger's word     7 · 10   7 · 8    11 · 13
    oracle tag      transcript + the acted emotion     13 · 13  13 · 13  13 · 13

R3 is the third point (`fee8c03` → `7f2fa52`): Qwen2-Audio-7B-Instruct on a
Modal A10, given the clip and the question and **no transcript and no tag**.
Its answer is the next-token probability of "Yes" against "No", read off one
forward pass -- the same read as PR-005's `YesNoScorer`, so the scores sit on
one scale. A pair is *answered right* when the loaded clip gets p(Yes) > 0.5
and its neutral twin ≤ 0.5; *ranked right* when the loaded clip's p(Yes) is
strictly higher (`sidechannel._pair_right`, unchanged).

Development data, **the 21 dev pairs only**, one A10 run, four wordings:

| wording | SAD | FEA | ANG | answered right · ranked right, of 7 |
|---|---|---|---|---|
| tone (PR-005's question) | 4 · 7 | 4 · 7 | 4 · 7 | |
| **upset** — "Is the person speaking upset?" | **6 · 7** | **5 · 7** | **5 · 7** | registered |
| not_words | 0 · 7 | 0 · 7 | 0 · 7 | Yes to all 42 clips |
| emotion | 0 · 7 | 0 · 7 | 0 · 7 | mass 0.00: it names the emotion instead |

R3 ranked every dev pair right under every wording (the loaded clip's p(Yes)
was about 1.00 each time) and transcribed every clip correctly. It leans Yes:
with the 0.5 threshold, neutral clips often clear it, which is what the
answered-right column measures. **The wording was chosen on the dev pairs
by this table**, and only there; the threshold stays at 0.5 because that is
PR-005's, and moving it for R3 would compare the two on different scales.

What was known, and by whom: the test pairs have never been sent to any GPU
(`prosodyset.load` refuses the test split without a registration, and there
is no cache for R3 -- every run is a fresh container). The author has seen
the side-channel's per-pair outcomes on the test pairs (PR-005), which decide
nothing here: R3 is not tuned, and its only chosen parameter is the wording,
chosen on dev.

### Question

On 39 pairs of voices it has never been shown by us, does a model that hears
the audio (a) tell the upset clip from the calm one, where a transcript-only
cascade cannot; and (b) do so at least as well as the $0 side-channel, on the
side-channel's own metric?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | `src/s2slab/r3probe.py` (`SYSTEM`, `WORDINGS`, `REGISTERED_WORDING`, `TRANSCRIBE`, `clips_for`, `score_pairs`) and `modal_app/r3_probe.py` (`MODEL`, the image, `SpeechLLM`, `run_probe`, `probe`) are unchanged between this commit and the run. Reporting -- `format_report`, `save`, the local entrypoint's output, and the comparison script added for step 2 -- may change. |
| **V2 · Untouched test pairs** | `prosodyset verify` passes on the day of the run; the test clips reach a GPU for the first time in the registered run. |
| **V3 · One run, whole set** | All 39 pairs, the first complete run, `modal run modal_app/r3_probe.py --split test --confirm PR-006 --wording upset --save runs/prosody/pr006.json`, from a clean tree (the stamp must not end in `-dirty`). One forward pass per clip in bf16 on an A10; a repeat is expected to reproduce the table, and a repeat that differs by more than a pair is reported as a finding. A crashed run is reported and repeated once, in full. |
| **V4 · Evidence** | The run file (every clip's p(Yes), mass and R3's transcript, and the per-emotion scores) and the printed report are saved to `runs/prosody/` in the commit that reports the result. The comparator is `runs/prosody/pr005.json` as committed. |

### Primary outcomes

Per emotion, over its 13 test pairs, under the `upset` wording: pairs
**ranked right** and pairs **answered right**, as `r3probe.score_pairs` counts
them at this commit; and, from `runs/prosody/pr005.json`, the side-channel's
answered-right outcome for each of the same pairs.

### Success criteria — two claims, two verdicts

**Claim A — the input bottleneck.** R3 hears what R1 cannot.

| | Criterion |
|---|---|
| **A1 · Sad** | ranked right on at least **12 of 13** pairs |
| **A2 · Fear** | ranked right on at least **12 of 13** pairs |
| **A3 · Anger** | ranked right on at least **12 of 13** pairs |

**Claim A is supported if A1, A2 and A3 all hold.** Against a coin flip per
pair, 12 or more of 13 has p = 0.002, and R1 blind managed 3 · 5 · 8.

**Claim B — native beats the proxy.** On the side-channel's metric, R3 is at
least as good for every emotion and better overall.

| | Criterion |
|---|---|
| **B1 · Per emotion** | answered right at least the side-channel's count: sad ≥ 7, fear ≥ 7, anger ≥ 11 of 13 |
| **B2 · Overall, paired** | over the 39 pairs, R3 answers right on more pairs than the side-channel did, and an exact two-sided sign test on the pairs where the two disagree gives **p < 0.05** |

**Claim B is supported if B1 holds for all three emotions and B2 holds.**

**Why two verdicts.** A single verdict would be decided by R3's Yes-lean at
the 0.5 threshold, which is calibration, not hearing. Claim A asks whether
the information is there; Claim B asks whether this model, asked plainly and
read at 0.5, beats a $0 tag. Both are worth knowing, and they can come out
differently.

**Why the tag's counts as the bars.** They are the numbers R3 has to match to
be worth 16 GB and an A10 where the tag costs nothing; and they are already
committed, so they cannot move.

**No multiplicity correction** within a claim: each needs all its parts.

### Predictions, recorded but not criteria

- **Claim A holds:** 13 of 13 ranked right for every emotion; the loaded clip's
  p(Yes) above 0.9 on nearly every pair.
- **Claim B fails, on anger.** R3 answered right about 11 · 9 · 9 of 13
  (scaling the dev rate), which beats the tag on sadness and fear but not
  anger's 11; the sign test over 39 pairs will not reach 0.05 (about 9 pairs
  fixed against 5 broken).
- R3's misses under Claim B are neutral clips it calls upset (p(Yes) between
  0.5 and 0.9), not upset clips it misses -- the mirror of the tagger's
  failure, which took calm voices for upset ones for a different reason.
- R3 transcribes at least 35 of 39 neutral clips word for word.
- No training-data contamination is detectable from this run alone: a clean
  ranking says the model hears the difference, not that it has never heard
  these actors. The own-voice mini set, recorded after this run, is the check.

### Reported alongside, descriptively

R3's ranked-right and answered-right counts under the other three wordings
if they are run (they are not part of the registered run and will not be);
p(Yes) medians for neutral and loaded clips; the mass on Yes-or-No; R3's
transcripts and their word error rate against the sentences; the per-pair
table with the side-channel's outcome beside R3's; the cost.

### Analysis plan

1. Commit this entry. The frozen code is frozen from here.
2. Add the comparison against `runs/prosody/pr005.json` (per-pair outcomes,
   B1, B2's sign test) to the reporting; test on made-up runs only.
3. `prosodyset verify`, a clean tree, then the one run (V2, V3). Save the evidence (V4).
4. Record the output below, verbatim.
5. Only after step 4: any further look at the test pairs, labelled
   exploratory; then the own-voice recordings.

### What this cannot show

- **Not that R3 learned nothing from CREMA-D.** Qwen2-Audio's training data
  is not documented and CREMA-D is a standard benchmark. Claim A on these
  actors could be memory. Only audio no model has seen answers that.
- **Not a calibrated R3.** The 0.5 threshold is PR-005's; a threshold chosen
  for R3 would likely lift Claim B, and would be tuning.
- **Not the reply, not latency.** The probe is a yes/no question; nothing here
  scores a spoken reply, and R3 is not built as a rung. The plan's own warning
  stands: this model is not streaming-native, so a rung built on it inherits
  R0's input-side latency.
- **One model, one wording, acted emotion, 13 pairs per emotion.**

### Deviations

- **2026-09-22, registry row.** Its "Registered" cell was changed to the
  registering commit's hash, `95c3d33`, and its status updated, as for the
  earlier entries.
- **2026-09-22, verbatim output.** The two per-pair tables (39 rows each) are
  kept verbatim in the committed `runs/prosody/pr006-report.md`; everything
  else the run printed is below.

### Results

**Run:** 2026-09-22, commit `5c1f3aa`, clean tree, Modal NVIDIA A10 (driver
580.95.05), `prosodyset verify` intact, the test clips' first time on any
GPU; all 39 pairs, first and only run: `modal run modal_app/r3_probe.py
--split test --confirm PR-006 --wording upset --save runs/prosody/pr006.json`.
66 s inside the function, about \$0.061. Evidence: `runs/prosody/pr006.json`
(every clip's p(Yes), mass and R3's transcript, and the scores) and
`runs/prosody/pr006-report.md` (the complete output). V1 held: zero changed
lines in the frozen parts since `95c3d33` (the reporting commit `5c1f3aa`
added below them only).

**Verdicts: Claim A supported; Claim B not supported.** R3 ranked the upset
clip above its calm twin on 13 of 13 pairs for every emotion. On the strict
metric it answered 11 · 11 · 9 right against the side-channel's 7 · 7 · 11:
anger fell short of the tag's count, and the paired sign test over all 39
pairs (11 fixed, 5 broken) gave p = 0.210.

Output of the run, verbatim, without the two per-pair tables (which are in
the report file):

```text
39 test pairs · confirms PR-006 · Qwen/Qwen2-Audio-7B-Instruct on NVIDIA A10, 580.95.05 · commit 5c1f3aa · load 5 s

| wording | SAD: answered right · ranked right (of 13) | FEA: answered right · ranked right (of 13) | ANG: answered right · ranked right (of 13) |
|---|---|---|---|
| upset | 11 · 13 | 11 · 13 | 9 · 13 |

upset: "Is the person speaking upset?" -- Yes to 47 of 78 clips, median p(Yes) 0.96, median mass on Yes-or-No 0.95

Claim A -- the input bottleneck: R3 ranks the upset clip above its calm twin
| criterion | ranked right | bar | |
|---|---:|---|---|
| A1 sad | 13 of 13 | at least 12 | PASS |
| A2 fear | 13 of 13 | at least 12 | PASS |
| A3 anger | 13 of 13 | at least 12 | PASS |

Claim B -- native beats the proxy: answered right, R3 against the side-channel
| criterion | R3 | side-channel | bar | |
|---|---:|---:|---|---|
| B1 sad | 11 of 13 | 7 of 13 | at least 7 | PASS |
| B1 fear | 11 of 13 | 7 of 13 | at least 7 | PASS |
| B1 anger | 9 of 13 | 11 of 13 | at least 11 | FAIL |
| B2 overall | 31 of 39 | 25 of 39 | more, and sign test p < 0.05 | FAIL |
  fixed 11: ang-t02, ang-t04, fea-t06, fea-t07, fea-t08, fea-t10, sad-t02, sad-t04, sad-t06, sad-t07, sad-t10
  broken 5: ang-t01, ang-t05, ang-t06, ang-t12, sad-t08
  exact two-sided sign test on the 16 pairs that differ: p = 0.210

PR-006 Claim A: SUPPORTED: R3 ranks at least 12 of 13 right for every emotion
PR-006 Claim B: NOT SUPPORTED: failed B1 anger, B2

R3's transcript of the neutral clip contains the sentence, word for word after normalizing: 39 of 39
```

**The three points on the axis**, same 39 pairs, same question, pairs
answered right · ranked right of 13 (SAD · FEA · ANG):

| | answered right | ranked right |
|---|---|---|
| R1 blind (PR-005, none) | 0 · 0 · 1 | 3 · 5 · 8 |
| R1 + tags (PR-005, tagger) | 7 · 7 · 11 | 10 · 8 · 13 |
| **R3** (this run) | **11 · 11 · 9** | **13 · 13 · 13** |
| oracle tag (PR-005) | 13 · 13 · 13 | 13 · 13 · 13 |

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Claim A holds, 13 of 13 everywhere, loaded p(Yes) > 0.9 nearly always | Right: 13 · 13 · 13; the lowest loaded p(Yes) was 0.94. |
| Claim B fails on anger; about 11 · 9 · 9; sign test short of 0.05, about 9 fixed to 5 broken | Right in every part: 11 · 11 · 9 (fear better than the dev rate), 11 fixed to 5 broken, p = 0.210. |
| R3's misses are calm clips called upset, never upset clips missed | Right: all 8 misses are neutral clips with p(Yes) 0.53 to 0.98. |
| R3 transcribes at least 35 of 39 neutral clips word for word | 39 of 39. |
| Contamination not detectable from this run | Unchanged: a perfect ranking is consistent with hearing and with memory. The own-voice set follows. |

**What the registration got right, for once, and why.** After PR-004 and
PR-005 the predictions were made from the mechanism (a Yes-lean at 0.5; the
tag's anger count as a high bar) rather than from a dev score read as the
population, and the dev-to-test drift was allowed for by predicting ranges.

**Reported alongside, descriptively**

- p(Yes) for the neutral clips: median 0.26, range 0.04 to 0.98; for the
  loaded clips: median 0.99, range 0.94 to 1.00. Yes to 47 of 78 clips.
- The 8 misses (neutral clips over 0.5): sad-t08 0.79, sad-t12 0.98,
  fea-t01 0.75, fea-t05 0.78, ang-t01 0.53, ang-t05 0.71, ang-t06 0.93,
  ang-t12 0.69. Three of them the side-channel also missed; five it got right.
- Mass on Yes-or-No: median 0.95. The model answered the question it was asked.
- Cost: \$0.061; the whole of M16 on GPUs, \$0.12.

**Exploratory, after the registered analysis**

A threshold of 0.6 on the neutral clips would have turned two anger misses
right and 0.8 four; that is calibration on the test pairs and is reported
only to size the gap. The right experiment is a threshold set on the dev
pairs, then a fresh test set -- as for the tagger's per-caller baseline, these
39 pairs are spent for R3 too.

---

## PR-007 — Does a per-caller baseline let the $0 tagger tell an upset caller from the same voice calm?

**Registered:** 2026-09-23, by the commit that adds this entry.
**Author:** Ram Charan

### Background

The affect tagger of PR-005 (`probes/prosody/tagger.json`, `931de62`) is
speaker-independent: 88 eGeMAPS functionals of one clip, a softmax regression
fitted on 16 CREMA-D actors, one word out. Its failure mode is now documented
twice. Under PR-005, 12 of its 15 misses on the 39 locked pairs were CALM clips
tagged emotional. Under M17, the author's calm voice through a new microphone
was tagged angry at 0.99. A tagger that has never heard a voice has no idea
what that voice sounds like when nothing is wrong.

The per-caller baseline (`src/s2slab/baseline.py`, `e86ec9d`) changes nothing
about the model except its input: every clip's 88 features have the mean of
*the same speaker's other neutral clips* subtracted -- how far this is from
this person's calm voice. Same features, same regression, same 16 training
actors (each training clip relative to its own actor's neutral training clips,
leave-one-out), saved as `probes/prosody/tagger-relative.json`. The deployment
analogue is a caller's earlier calm turns.

Development data, **the 21 dev pairs only**, each dev actor's baseline being
their 9–11 neutral clips that raters heard as neutral (`probes/prosody/baseline/`):

| dev, 21 pairs (7 per emotion) | told apart SAD · FEA · ANG | told apart, all | both right | loaded clip tagged non-neutral |
|---|---|---:|---:|---:|
| absolute tagger (PR-005's) | 1 · 6 · 4 | 11 of 21 | 9 | 14 |
| **relative tagger** | **5 · 6 · 5** | **16 of 21** | **13** | **16** |
| relative, baseline of 1 / 3 / 5 clips | | 11 / 13 / 15 | | |

Pair by pair the relative tagger fixed 7 and broke 2 (`fea-d01`, `ang-d03`).
Held-out training actors: neutral 81% (was 69%), sad 50% (53%), fear 39%
(45%), anger 80% (76%). On the author's five own-voice pairs, each calm
take's baseline being the other four: told apart 2 of 5 either way.

**Why the 39 locked pairs are not the test set.** The hypothesis was read off
PR-005's test misses. Confirming it on the same pairs would confirm a
post-hoc hypothesis on the data that produced it. Those pairs are spent for
the tagger; nothing here reads them.

**The fresh pairs** (`probes/prosody2/`, `b9247cc`). The original rule
(`prosodyset.select`: same actor, same sentence, both clips heard as intended
by enough of enough raters, seeded draw, one pair per actor per emotion, at
most 3 per sentence) applied to the 32 TEST-side actors of the original
seeded split that have no pair in the original set -- voices no system in
this repository has scored. Per emotion, the highest threshold that fills 13
pairs; where none does, the lowest threshold and what it gives:

| raters agreeing | SAD (of 13) | FEA (of 13) | ANG (of 13) |
|---|---:|---:|---:|
| at least 90% | 1 | 0 | 3 |
| at least 80% | 1 | 6 | **13** |
| at least 70% | 4 | 12 | · |
| at least 60% | 8 | **13** | · |
| at least 50% | **12** | · | · |

**38 pairs from 27 actors**: SAD 12 (loaded clips heard as intended by 50–75%
of raters), FEA 13 (60–90%), ANG 13 (80–100%). The thresholds are lower than
the original set's (60 / 80 / 90) because the clearer actors were drawn first:
these pairs are *harder* for both taggers, and the absolute tagger's counts
here are not comparable to PR-005's. Each fresh actor's baseline is their
neutral clips heard as neutral that are in no pair, 268 clips in all, 6 to 11
per actor, median 10 (`probes/prosody2/baseline/baseline.tsv`). **All of them
are used**; the baseline size is not chosen.

What was known, and by whom: the author and the assistant have seen the dev
table above and nothing of the fresh pairs beyond their names, ratings and
counts. No audio of `probes/prosody2/` has been fetched at this commit
(`probes/prosody2/audio/` does not exist), so no feature of any fresh clip
has been extracted by anyone. The assistant ran the fresh draw read-only in a
scratch copy to check it fills; the same draw, run by the author, produced the
committed `pairs.tsv` line for line.

### Question

On 38 pairs of voices no system here has scored, does the same tagger tell
more upset-calm pairs apart when it is given the speaker's own neutral clips
than when it is not, without buying that by tagging the upset clips neutral?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | `probes/prosody/tagger.json` (`931de62`), `probes/prosody/tagger-relative.json` (`e86ec9d`), `src/s2slab/affect.py` (frozen since PR-005), and in `src/s2slab/baseline.py`: `Baseline`, `relative_features`, `told_apart`, `baseline_clips`, `select_fresh`, `freeze_fresh`; `probes/prosody2/pairs.tsv` and `probes/prosody2/baseline/baseline.tsv` as committed in `b9247cc`. Reporting -- a `test` command that runs both taggers, prints the criteria and the per-pair listing and saves the run -- may be added after this commit and is tested on made-up features only. |
| **V2 · Untouched fresh pairs** | The fresh clips and baseline clips are fetched after this commit and frozen (`verify-fresh` intact on the day of the run). No feature cache for `probes/prosody2/` exists before the run (`ls results/affect/` shows no `fresh*.npz`). The run is the first time any of these clips is read by any system here. |
| **V3 · One run, whole set** | All 38 pairs, `python -m s2slab.baseline test --confirm PR-007 --save runs/prosody/pr007.json`, from a clean tree. CPU only, deterministic (openSMILE features, fixed models): a repeat must reproduce the tables exactly; one that does not is reported as a finding. |
| **V4 · Evidence** | The run file (every clip's tag and probability under both taggers, the baseline size used for each speaker, the per-emotion counts) and the printed report are saved to `runs/prosody/` in the commit that reports the result. |

### Primary outcome

Per pair, for each tagger, **told apart**: the neutral clip tagged neutral AND
the loaded clip tagged anything else (`baseline.told_apart`, which counts what
`affect.tag_pairs` prints). The relative tagger's baseline for a pair is the
mean over all of that speaker's baseline clips. Over the 38 pairs: the two
counts, the pairs the relative tagger *fixed* (absolute wrong, relative right)
and *broke* (the reverse).

### Success criteria

| | Criterion |
|---|---|
| **S1 · Primary, paired** | the relative tagger tells apart **more** of the 38 pairs than the absolute tagger, and an exact **one-sided** sign test on the pairs where they disagree (fixed against broken) gives **p < 0.05** |
| **S2 · Guardrail** | loaded clips tagged non-neutral by the relative tagger number at least the absolute tagger's count **minus 2** (over 38) |

**Supported if S1 and S2 both hold.** Per-emotion counts are reported, not
deciding: 12 or 13 pairs per emotion cannot carry a verdict on their own.

**Why one-sided.** The hypothesis is directional -- the baseline fixes calm
clips -- and is registered as such here; a relative tagger that breaks more
than it fixes is "not supported" whichever test is run. On the dev rate
(7 fixed, 2 broken over 21 pairs) scaled to 38, about 12 fixed against 4
broken gives p = 0.038 one-sided; the same split two-sided would not reach
0.05, so the test is close either way, and the choice is written down before
the numbers exist.

**Why the guardrail.** Subtracting a baseline could make upset clips look
neutral (a person whose calm voice is already loud). On dev the loaded-clip
count went up (14 → 16); S2 catches the case where it goes down by more than
noise.

### Predictions, recorded but not criteria

- **Absolute tagger:** told apart about **17 to 20 of 38** -- SAD 2–4 of 12,
  FEA 7–9 of 13, ANG 8–10 of 13. Lower than PR-005's 24 of 39 because these
  pairs were heard as intended by fewer raters.
- **Relative tagger:** about **25 to 28 of 38** -- SAD 6–8, FEA 8–10, ANG
  10–12. Sadness gains most, as on dev; fear least.
- **S1:** about 11 fixed against 3 broken; one-sided p between 0.01 and 0.05.
  S1 is the criterion most at risk, on the sign test rather than the count.
- **S2 holds:** loaded clips tagged non-neutral go up or stay, as on dev.
- The relative tagger's remaining misses are mostly SAD loaded clips tagged
  neutral (the quiet, low-agreement sad clips), not calm clips tagged
  emotional -- the failure mode moves from the calm clip to the upset one.

### Reported alongside, descriptively

Per emotion, both taggers: told apart, both right, neutral clip tagged
neutral, loaded clip tagged its own emotion; the per-pair listing with both
taggers' tags and probabilities; the baseline size of each speaker; the
absolute tagger's counts as a second, harder sample of PR-005's ceiling.
After the registered analysis and labelled exploratory: told apart against
baseline size 1 / 3 / 5 on the fresh pairs.

### Analysis plan

1. Commit this entry. The frozen parts are frozen from here.
2. `baseline fetch-fresh` (76 pair clips and 268 baseline clips from the
   pinned CREMA-D commit, about 17 MB; the download is the author's), which
   writes both manifests; `verify-fresh`.
3. Add the `test` command (both taggers, criteria table, per-pair listing,
   `--save`); test on made-up features only. Commit with the manifests.
4. `verify-fresh`, `ls results/affect/`, a clean tree, then the one run (V2, V3). Save the evidence (V4).
5. Record the output below, verbatim.
6. Only after step 5: the baseline-size curve on the fresh pairs and any other
   look, labelled exploratory.

### What this cannot show

- **Not a deployment number.** The baseline is 6–11 studio-clean neutral
  sentences by the same actor in the same session -- the best case. A
  caller's earlier turns are noisier and may not be calm.
- **Not the original set's difficulty.** Thresholds of 50 / 60 / 80 admit
  clips the raters were less sure about; both taggers should do worse here
  than on PR-005's pairs, and the comparison is between the two taggers, not
  with PR-005.
- **Acted emotion, one corpus, 38 pairs, 27 actors.** Twelve sad pairs at 50%
  agreement carry little on their own; hence the pooled, paired test.
- **Not R3.** The speech-LLM is not run on these pairs; the question is
  whether the $0 proxy improves with a baseline, not where it stands against
  R3.

### Deviations

- **2026-09-23, registry row.** Its "Registered" cell was changed to the
  registering commit's hash, `02baa74`, and its status updated, as for the
  earlier entries.
- **2026-09-23, verbatim output.** The per-pair table (38 rows) is kept
  verbatim in the committed `runs/prosody/pr007-report.md`; everything else
  the run printed is below.
- **2026-09-23, order of steps.** The exploratory baseline-size curve
  (`explore-fresh`) was run after the registered run had printed and saved
  its output, but before this results text was committed -- step 6 says after
  step 5. Nothing in the registered analysis depends on it and the saved run
  file predates it.

### Results

**Run:** 2026-09-23, commit `a591503`, clean tree, `verify-fresh` intact, no
fresh feature cache beforehand, laptop CPU (deterministic; \$0); all 38 pairs,
first and only run: `python -m s2slab.baseline test --confirm PR-007 --save
runs/prosody/pr007.json`. Evidence: `runs/prosody/pr007.json` (every clip's tag
and probability under both taggers, each speaker's baseline size, the counts)
and `runs/prosody/pr007-report.md` (the complete output). V1 held: the frozen
functions are byte-identical between `02baa74` and `a591503` (checked
function by function before the reporting commit).

**Verdict: NOT SUPPORTED — S1 failed.** The two taggers tell apart the same
number of pairs, 26 of 38; the relative tagger fixed 2 and broke 2.

Output of the run, verbatim, without the per-pair table (which is in the
report file):

```text
38 fresh pairs · confirms PR-007 · absolute tagger against the per-caller baseline · 268 baseline clips for 27 speakers · commit a591503

absolute tagger (tagger.json):
| emotion | pairs | both right | told apart | neutral clip tagged neutral | loaded clip tagged its emotion |
|---|---:|---:|---:|---:|---:|
| SAD | 12 | 7 | 8 | 10 | 9 |
| FEA | 13 | 5 | 8 | 11 | 6 |
| ANG | 13 | 8 | 10 | 10 | 11 |

pairs not told apart:
  sad-f03: neutral clip tagged neutral (0.94), loaded clip tagged neutral (0.79)
  sad-f05: neutral clip tagged sad (0.55), loaded clip tagged sad (0.87)
  sad-f08: neutral clip tagged sad (0.89), loaded clip tagged sad (0.83)
  sad-f10: neutral clip tagged neutral (0.98), loaded clip tagged neutral (0.98)
  fea-f01: neutral clip tagged neutral (0.88), loaded clip tagged neutral (0.95)
  fea-f02: neutral clip tagged sad (0.90), loaded clip tagged fearful (0.35)
  fea-f04: neutral clip tagged neutral (0.98), loaded clip tagged neutral (0.89)
  fea-f07: neutral clip tagged sad (1.00), loaded clip tagged sad (0.96)
  fea-f13: neutral clip tagged neutral (0.89), loaded clip tagged neutral (0.91)
  ang-f01: neutral clip tagged sad (0.85), loaded clip tagged angry (0.93)
  ang-f04: neutral clip tagged sad (0.99), loaded clip tagged angry (0.99)
  ang-f13: neutral clip tagged sad (0.55), loaded clip tagged angry (0.95)

relative tagger (tagger-relative.json), baseline = all of each speaker's baseline clips:
| emotion | pairs | both right | told apart | neutral clip tagged neutral | loaded clip tagged its emotion |
|---|---:|---:|---:|---:|---:|
| SAD | 12 | 7 | 8 | 11 | 8 |
| FEA | 13 | 8 | 9 | 11 | 8 |
| ANG | 13 | 8 | 9 | 9 | 11 |

pairs not told apart:
  sad-f03: neutral clip tagged neutral (0.85), loaded clip tagged neutral (0.70)
  sad-f05: neutral clip tagged neutral (0.96), loaded clip tagged neutral (0.41)
  sad-f08: neutral clip tagged sad (0.96), loaded clip tagged sad (0.61)
  sad-f10: neutral clip tagged neutral (0.97), loaded clip tagged neutral (0.93)
  fea-f01: neutral clip tagged neutral (0.91), loaded clip tagged neutral (0.76)
  fea-f04: neutral clip tagged sad (0.73), loaded clip tagged neutral (0.53)
  fea-f07: neutral clip tagged sad (1.00), loaded clip tagged sad (0.98)
  fea-f13: neutral clip tagged neutral (0.99), loaded clip tagged neutral (0.87)
  ang-f01: neutral clip tagged sad (0.74), loaded clip tagged angry (0.99)
  ang-f02: neutral clip tagged sad (0.61), loaded clip tagged fearful (0.96)
  ang-f11: neutral clip tagged angry (0.62), loaded clip tagged angry (1.00)
  ang-f13: neutral clip tagged sad (0.68), loaded clip tagged angry (0.94)

| criterion | absolute | relative | bar | |
|---|---:|---:|---|---|
| S1 told apart, paired | 26 of 38 | 26 of 38 | more, and one-sided sign test p < 0.05 | FAIL |
| S2 loaded clips tagged non-neutral | 33 of 38 | 32 of 38 | at least 31 | PASS |
  fixed 2: ang-f04, fea-f02
  broken 2: ang-f02, ang-f11
  exact one-sided sign test on the 4 pairs that differ: p = 0.688
  told apart per emotion (absolute -> relative): SAD 8 -> 8 of 12, FEA 8 -> 9 of 13, ANG 10 -> 9 of 13

PR-007: NOT SUPPORTED: failed S1
```

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Absolute tagger tells apart 17–20 of 38, fewer than PR-005's 24 of 39 because these pairs were heard as intended by fewer raters | **Wrong, the other way:** 26 of 38 (68%) against 62% on PR-005's pairs. Rater agreement is not tagger difficulty. |
| Relative tagger 25–28; sadness gains most | The count landed in range (26) and the gain it was meant to carry does not exist: SAD 8 → 8, FEA 8 → 9, ANG 10 → 9. **Wrong in what matters.** |
| About 11 fixed against 3 broken, one-sided p between 0.01 and 0.05; S1 most at risk on the test rather than the count | 2 fixed, 2 broken, p = 0.688. Wrong; S1 failed on the count itself. |
| S2 holds | Held: 33 → 32, bar 31. |
| The remaining misses move from the calm clip to the upset one | **Wrong.** Both taggers miss the same way: the calm clip tagged emotional in 7 pairs each (six of them "sad"; `fea-f07` at 1.00 under both), the upset clip tagged neutral in 5. The baseline fixed two calm-clip misses and made two. |

**What the registration got wrong.** Two things. Dev's 7 fixed against 2
broken over 21 pairs from 14 actors was read as a rate to be scaled; it is
p = 0.09 one-sided, and the entry called the test "close" while treating the
direction as settled — the same dev-as-population error as PR-004 and PR-005,
in a new coat. And the mechanism was wrong for this corpus. Subtracting a
speaker's mean over their *other* neutral sentences removes a speaker offset;
a calm clip that is quieter or flatter than that speaker's own average is
still "sad" relative to it. These misses are clip-level, not speaker-level:
on CREMA-D actors recorded in one studio, speaker offset is not what the
tagger stumbles on. The place a baseline plausibly does help — a new voice
through a new microphone, where every calm take read as angry — is the
author's five pairs (calm tagged neutral 2 → 3 of 5, dev-only, one speaker),
and that stays a dev observation.

**Reported alongside, descriptively**

- Per emotion, both right: absolute 7 · 5 · 8, relative 7 · 8 · 8; the
  relative tagger names fear correctly more often (loaded fear clips tagged
  fearful 6 → 8) without telling more fear pairs apart.
- The absolute tagger on a second sample: 26 of 38 told apart, 68%, against
  24 of 39 (62%) on PR-005's pairs — PR-005's ceiling replicates, a little
  higher, on harder-rated pairs.
- Baseline sizes 6 to 11 per speaker; the two fixed pairs had baselines of
  9 and 6 clips, the two broken 9 and 10: nothing there.
- Cost: \$0. The whole of M18 ran on the laptop.

**Exploratory, after the registered analysis** — told apart against baseline
size (`explore-fresh`, from the run's caches; first k baseline clips of each
speaker by name):

| baseline clips per speaker | SAD | FEA | ANG | all |
|---:|---:|---:|---:|---:|
| 1 | 8 of 12 | 8 of 13 | 7 of 13 | 23 of 38 |
| 3 | 7 of 12 | 9 of 13 | 8 of 13 | 24 of 38 |
| 5 | 9 of 12 | 8 of 13 | 8 of 13 | 25 of 38 |
| all | 8 of 12 | 9 of 13 | 9 of 13 | 26 of 38 |

A single baseline clip costs three pairs against the absolute tagger; the
full baseline only gets back to even. Whatever a per-caller baseline buys, it
does not buy it here.

---

## PR-008 — Does R4's own voice react to the caller's voice?

**Registered:** 2026-09-23, by the commit that adds this entry.
**Author:** Ram Charan

### Background

The v2 plan, R4: "No TTS text roundtrip; the model chooses its own delivery.
... Measure whether output F0 variance actually moves with sentiment. If it
doesn't, the rung bought nothing -- and that is the finding."

R4 is built (`src/s2slab/r4.py`, `r4-gpu`): Qwen2.5-Omni-7B hears the turn
and speaks the reply itself, no ASR and no TTS. On the eval set it takes 9.2 s
p50 on an L40S against R3's 0.69 s in the same container (`640cae3`). Its
Talker samples its speech (the model's own defaults); every turn is seeded
(`SEED`), and on the dev run answering a clip twice gave byte-identical audio.

The measure (`src/s2slab/r4probe.py`): the prosody pairs go in as the
caller's turn -- same actor, same sentence, once calm and once upset. Each
rung answers both clips; for each pair, the F0 range (10th-90th percentile,
semitones, `prosodycheck.features` after resampling to 16 kHz) of the reply to
the upset clip minus that of the reply to the calm clip is `d_range`, and the
pair's DIRECTION is "wider" (d_range > +0.5 st), "narrower" (< -0.5) or
"same". The comparator is R3 (Qwen2-Audio hears the clip, Kokoro speaks its
text): Kokoro takes its delivery from the words alone, so any effect in R3 is
the upset clip changing the words, which Kokoro then voices.

Development data, **the 21 dev pairs only** (`a677969`, L40S, \$0.20):

| dev, 21 pairs | wider | same | narrower | median d range | reply words, calm -> upset |
|---|---:|---:|---:|---:|---|
| R3 (Qwen2-Audio -> Kokoro) | 4 | 14 | 3 | +0.0 st | 17 -> 21 |
| **R4 (Qwen2.5-Omni speaks)** | **5** | **5** | **11** | **-0.6 st** | 17 -> 17 |

Fear, R4: 1 wider, 5 narrower, median -1.2 st.

**Two decisions made on the dev pairs, disclosed.** (1) The design note
counted a pair as moving when d_range > 0. On dev, R3's Kokoro proved not
bit-deterministic (a clip answered twice differed by up to 0.18 in sample
value), and where R3 wrote the same words for both clips its d_range stayed
within +-0.05 st, which that rule counted as "wider". The band of +-0.5 st --
ten times that jitter, the same for both directions -- was set from this.
(2) Dev says R4 flattens rather than widens. The registered claim is kept as
the plan's (wider), not turned to fit dev; the two-sided question is
registered beside it as a secondary, with its own verdict.

What was known, and by whom: the 39 test pairs have been heard by
Qwen2-Audio before, as yes/no questions under PR-006; neither rung has ever
answered one as a caller's turn, R4 has never heard one, and nothing here is
tuned -- the rungs answer as built. The author and the assistant have seen
PR-005's and PR-006's per-pair results on those pairs, which say nothing
about either rung's reply.

### Question

When the caller's voice is upset rather than calm, with the same words, does
R4 answer in a voice with a wider pitch range -- more than chance, and more
than R3's words carry into Kokoro?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | `src/s2slab/r4.py` (`OmniS2S`, `SpeechToSpeech`, `SEED`, `SPEAKER`, `INSTRUCTION`), `src/s2slab/r3.py`, and in `src/s2slab/r4probe.py`: `answer_clips`, `to_16k`, `measure`, `pair_deltas`, `direction`, `counts`, `TIE_ST`, `compare`, `format_criteria`; `modal_app/r4_bench.py`'s image, `answer` and `pairs`. All as committed with this entry. Reporting beyond `format_criteria` may be added and is tested on made-up replies only. |
| **V2 · Untouched test pairs** | `prosodyset verify` intact on the day of the run; the test clips reach R3-as-a-rung and R4 for the first time in the registered run (`r3probe.clips_for` refuses the test side without a registration). |
| **V3 · One run, whole set** | All 39 pairs, `modal run modal_app/r4_bench.py::pairs --split test --confirm PR-008 --save runs/prosody/pr008.json`, from a clean tree. R4 is seeded per turn; R3's text is greedy and its Kokoro jitter sits far inside the band. A repeat that changes a pair's direction is reported as a finding. A crashed run is reported and repeated once, in full. |
| **V4 · Evidence** | The run file (every reply's text, timing and measures; every pair's deltas and direction; the seeding check) and the printed report go to `runs/prosody/` in the commit that reports the result. Reply audio stays in `results/` (ignored). |

### Primary outcome

For each rung and each of the 39 test pairs: `d_range` and its direction, as
`r4probe.pair_deltas` computes them at this commit.

### Success criteria

| | Criterion |
|---|---|
| **S1 · R4 widens** | R4's pairs: more "wider" than "narrower", and an exact **one-sided** sign test on wider against narrower gives **p < 0.05** |
| **S2 · R4 beats R3** | pairs where R4 is wider and R3 is not, against pairs where R3 is wider and R4 is not: more of the first, one-sided sign test **p < 0.05** |

**Supported if S1 and S2 both hold.** Pairs that are "same" or unmeasured
count toward neither side.

**Registered secondary, with its own verdict:** the **two-sided** sign test on
R4's wider against narrower -- does R4's delivery move consistently in either
direction? Reported as "moves consistently -- wider / narrower" at p < 0.05,
else "no consistent direction". It cannot turn S1's verdict.

**Why one-sided for S1.** The plan's question is directional: an agent that
hears distress should sound more engaged, not less. Dev points the other way,
and the claim is registered as the plan wrote it rather than as dev suggests;
the other direction is what the secondary is for.

### Predictions, recorded but not criteria

- **S1 fails.** R4 about 9 wider, 20 narrower, 10 same of 39 (scaling dev).
- **S2 fails** with it: R4 wider where R3 is not on about 7 pairs, R3 wider
  where R4 is not on about 5.
- **The secondary is close:** at 9 against 20 the two-sided p is about 0.06;
  it is a coin toss whether it reaches 0.05 ("narrower").
- Fear flattens most (as on dev); anger least.
- R3 mostly "same" (about 26 of 39); its words run longer for the upset clip
  (about 17 -> 21), R4's do not.
- R4's seeding check prints "identical" again.

### Reported alongside, descriptively

Per emotion, both rungs: wider / same / narrower, median d_range, d_pitch
and d_level; reply word counts, calm against upset; every pair's deltas side
by side; the seeding check; a few replies' texts to both clips of a pair. The
reply WAVs are kept in `results/` for listening and are not committed.

### Analysis plan

1. Commit this entry together with the frozen code above.
2. `prosodyset verify`, a clean tree, then the one run (V2, V3). Save the evidence (V4).
3. Record the output below, verbatim.
4. Only after step 3: any further look, labelled exploratory.

### What this cannot show

- **Not whether R4's delivery is appropriate.** A flatter voice could be a
  calmer, deliberately soothing one; F0 range measures animation, not care.
  Nobody listens and rates here.
- **Not the model's ceiling.** One speaker voice (Chelsie), the model's own
  default sampling, one seed; another seed gives other audio.
- **Not Kokoro's.** R3 is the comparator for text leaking into voice, not a
  claim about what a prosody-aware TTS could do.
- **Acted emotion, one corpus, 39 pairs, 13 per emotion** -- per emotion is
  reported, not decided.

### Deviations

- **2026-09-24, registry row.** Its "Registered" cell was changed to the
  registering commit's hash, `f4da0bc`, and its status updated, as for the
  earlier entries.
- **2026-09-24, the printed output.** The registered run's printed output was
  overwritten when the same command was invoked a second time the next
  morning (`| tee results/r4-gpu/pr008-report.md`); the overwrite guard
  refused that invocation before any call to Modal, so no second answer to any
  test pair exists. The report committed as `runs/prosody/pr008-report.md` and
  quoted below is regenerated from the saved run file by the frozen
  `format_report` and `format_criteria`; the run file itself is untouched. The
  run's cost line was lost with it and is estimated from the saved turn times.

### Results

**Run:** 2026-09-23 20:32, commit `f4da0bc` (the registering commit), clean
tree, `prosodyset verify` intact, Modal NVIDIA L40S; all 39 pairs, first and
only run: `modal run modal_app/r4_bench.py::pairs --split test --confirm
PR-008 --save runs/prosody/pr008.json`. Evidence: `runs/prosody/pr008.json`
(every reply's text, timing and measures; every pair's deltas and direction;
the seeding check) and `runs/prosody/pr008-report.md` (regenerated -- see
Deviations). Reply audio in `results/r4-gpu/replies/test/`, not committed.

**Verdict: NOT SUPPORTED -- S1 and S2 failed; the secondary finds no
consistent direction.** R4's reply to the upset clip had the wider pitch range
on 14 pairs and the narrower on 16; R3's on 12 and 6.

The criteria, as the frozen `format_criteria` prints them from the run file:

```text
| criterion | R4 | R3 | bar | |
|---|---|---|---|---|
| S1 R4's reply wider for the upset caller | 14 wider, 16 narrower, 9 same | | more wider, one-sided sign test p = 0.708 < 0.05 | FAIL |
| S2 R4 against R3, pair by pair | wider where R3 is not: 11 | wider where R4 is not: 9 | more, one-sided sign test p = 0.412 < 0.05 | FAIL |
| R3 for reference | | 12 wider, 6 narrower, 21 same | | |
  secondary, two-sided: R4's delivery moves consistently one way -- 14 wider against 16 narrower, p = 0.856

PR-008: NOT SUPPORTED: failed S1, S2
PR-008 secondary: no consistent direction
```

The full per-emotion table and the per-pair listing are in
`runs/prosody/pr008-report.md`. Per emotion, R4 wider / same / narrower: sad
4 / 2 / 7 (median -0.9 st), fear 4 / 4 / 5 (-0.1), anger 6 / 3 / 4 (+0.2).

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| S1 fails | Right. |
| S2 fails; about 7 against 5 | Right on the verdict; 11 against 9. |
| R4 about 9 wider, 20 narrower, 10 same | **Wrong:** 14 / 16 / 9 -- nearly even, not flattening. |
| The secondary close to 0.05, "narrower" | **Wrong:** p = 0.856, no direction at all. |
| Fear flattens most, anger least | **Wrong:** sadness flattened most (7 of 13 narrower); fear even; anger slightly wider. |
| R3 mostly "same", about 26 of 39 | 21 of 39: roughly. |
| R3's words run longer for the upset clip, about 17 -> 21 | **Wrong:** 16 -> 17. |
| R4's seeding check prints "identical" | Right. |

**What the registration got wrong.** Dev's 5 wider against 11 narrower on 21
pairs was read as a direction; on 39 fresh pairs it is 14 against 16. The
fourth time dev was read as the population (PR-004, PR-005, PR-007, now this).
What the registration got right was not to turn the claim toward dev's
direction: registered as "narrower", this would have been a confident false
start; registered as the plan wrote it, with the other direction as a
secondary, the secondary is what showed that dev's direction was noise.

**Reported alongside, descriptively**

- R4's voice moves a lot but either way: only 9 of its 39 pairs fall inside
  +-0.5 st, against R3's 21. Its words do not change (17 -> 17 median).
- On `sad-t06` R3's reply to the upset clip ("Oh, I didn't realize it was
  that late. Is everything okay?") measured a 19.5 st range, +12.3 st over
  its calm twin -- most likely an octave jump in the pitch track on Kokoro's
  question intonation. It changes no count; only the direction is used.
- R4 answered `sad-t06`'s calm and upset clips with the same words and
  different deliveries (-2.8 st): the Talker's sampling, seeded, differs with
  the input even when the text does not.
- Turn times: R3 median 653 ms, R4 8012 ms (the replies here are short).
- Cost: about \$0.45 (estimated from the saved turn times -- the printed cost
  line was lost; see Deviations).


## PR-009 — Does the severity render do what page 09's mapping says, on voices and sentences it was not built on?

**Registered:** 2026-09-24, by the commit that adds this entry.
**Author:** Ram Charan

### Background

v1 P2c (Notion page 16, weeks 5-6) and page 09 §6: a clinical severity score
becomes acoustic targets, and "for fixed text, increasing severity must
monotonically move each target feature in the specified direction, verified by
extracting features from the generated audio." Page 16 §8 makes the same
property part of its duplex go/no-go: "controllability demonstrated within
tolerance monotonically with severity."

`src/s2slab/severity.py` (`88330df`, `0056da9`): page 09's PHQ-9 and YMRS
formulas, taken word for word, give four acoustic targets -- pitch offset
(Hz), pitch range scale, speaking rate, pause length. A neutral Kokoro clip is
edited to them in one Praat pass (pitch tier and duration tier, overlap-add
resynthesis: page 09 §7, strategy 5), and the result is measured again from
the audio: median F0 offset, middle-half F0 spread, speech time from the
clip's durations, silent time where the render's time map puts each pause.
Page 09's non-acoustic outputs (response latency, elaboration, interruption)
are not rendered.

Development data, **dev voices and sentences only**: Kokoro `af_heart` and
`am_michael`, four neutral sentences (`DEV_SENTENCES`), both conditions at six
scores each -- 16 sweeps, 64 feature sweeps, 96 checks per feature. Run twice,
because Kokoro speaks a sentence differently every time:

| dev, 2 voices x 4 sentences | assistant's run (scratch) | author's run (`0056da9`) |
|---|---:|---:|
| feature sweeps in order | 62 of 64 | 63 of 64 |
| -- pitch offset, rate, pauses | 48 of 48 | 48 of 48 |
| -- pitch range | 14 of 16 | 15 of 16 |
| rate within 10% | 96 of 96 | 96 of 96 |
| pauses within 10% | 96 of 96 | 96 of 96 |
| pitch offset within 3 Hz or 10%, the larger | 89 of 96 | 92 of 96 |
| pitch range within 10% | 86 of 96 | 85 of 96 |

The pitch-offset row applies this entry's tolerance (`within`) to the two
saved dev runs; the module's own report used 3 Hz alone (87 and 90 of 96).
Every dev sweep out of order was a pitch-range sweep, and a mania one. At
YMRS 60 the offset landed between 0.7 Hz under and 9.4 Hz over its +45 Hz.

**Decisions made on dev, disclosed.** Each is in the module docstring:
(1) the pitch compression is centred on the median of the measured F0, not
page 09's mean -- octave jumps drag the mean; (2) range is measured as the
middle-half spread, which a linear edit scales exactly; (3) rate is measured
from durations, not voiced frames, which drifted by up to 9% at 1.8x speed;
(4) pauses are measured where the time map puts them, so a pause shortened
below 150 ms still counts; (5) page 09's "f0_variance_scale" is applied as its
own formula applies it, to deviations, so the spread scales by that factor.
The thresholds below were set from the two dev runs above.

**What was known, and by whom.** The held-out voices and sentences below were
chosen and written by the assistant with this entry; no code here has spoken
either voice for the severity work or any of the eight sentences. The author
made listening files on dev (`dev --listen`); no listening notes are part of
this registration.

### Question

On two Kokoro voices and eight neutral sentences the render was not developed
on, does raising the severity move every rendered feature the way page 09's
mapping says, at every step, and land each feature within a stated tolerance
of its target?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | In `src/s2slab/severity.py`, as committed with this entry: `normalize`, `depression`, `mania`, `CONDITIONS`, `SCORES`, `trim`, `f0_hz`, `duration_points`, `warp`, `render`, `silent_s`, `achieved`, `run`, `kokoro_speaker`, and every name in the PR-009 section (`TEST_VOICES`, `TEST_SENTENCES`, the thresholds, `in_order`, `within`, `criteria`, `validity`, `registered_run`). `report` and `format_criteria` may gain lines, tested on made-up runs only. |
| **V2 · Held out** | `af_sarah`, `am_adam` and the eight `TEST_SENTENCES` reach Kokoro for the first time in the registered run. `registered_run` is the only path to them; it refuses without `--confirm PR-009` and before any TTS loads. The tests use a fake TTS. |
| **V3 · One run, whole set** | All 32 sweeps, `PYTHONPATH=src python -m s2slab.severity test --confirm PR-009 \| tee results/severity/pr009-report.txt`, from a clean tree (the run file's commit must not end in `-dirty`). Kokoro speaks each neutral clip once; every render is made from that clip. A crashed run writes no run file; it is reported here and repeated once, in full. |
| **V4 · Evidence** | `runs/severity/pr009.json` -- every target and measure, every neutral clip's SHA-256, the criteria -- and the printed report, committed to `runs/severity/` with the results. The renders and the exact (32-bit float) neutral clips stay in `results/severity/pr009/`, ignored; from those clips any render can be made again. |

### Primary outcome

For each of the 32 sweeps (2 voices x 8 sentences x 2 conditions, six scores
each): whether each feature is in order at every step (`in_order`), and for
each of the 192 renders whether each feature is within tolerance (`within`),
as computed at this commit.

### Success criteria -- all five must hold

| | Criterion |
|---|---|
| **S1 · Page 09's monotonicity test** | Pitch offset, rate and pauses in order in **every** sweep: 96 of 96 |
| **S2 · Pitch range in order** | At least **28 of 32** range sweeps in order |
| **S3 · Rate and pauses on target** | **Every** check within 10% of its target: 384 of 384 |
| **S4 · Pitch offset on target** | At least **90%** of the 192 checks within 3 Hz or 10% of the target, whichever is larger |
| **S5 · Pitch range on target** | At least **80%** of the 192 checks within 10% |

**Supported if S1 to S5 all hold.** A feature that cannot be measured at a
step (no voiced frames, no pause) counts as out of order and as a miss. S1 is
page 09's own test; S2 to S5 hold the render to the targets and give the
range, its weak feature on dev, bars of its own rather than excusing it.

### Predictions, recorded but not criteria

- **S1 and S3 hold.** 96 of 96; 384 of 384.
- **S2 holds narrowly:** 29 to 31 of 32. Any sweep out of order is pitch range, under mania.
- **S4 holds:** about 94%; the misses are mania at YMRS 40 and 60, overshooting by 3 to 9 Hz.
- **S5 holds:** about 85-90%.
- `am_adam` misses more range checks than `af_sarah`.
- **Supported.**

### Reported alongside, descriptively

The full sweep table (`report`): every target and every measure, misses marked;
counts per voice and per condition; the neutral clips' lengths.

### Analysis plan

1. Commit this entry together with the frozen code above.
2. A clean tree, then the one run (V2, V3). Save the evidence (V4).
3. Record the output below, verbatim.
4. Only after step 3: any further look, labelled exploratory -- including listening.

### What this cannot show

- **Not whether it sounds natural, or like anyone.** Nobody listens and rates
  here; page 09 §7 expects audible processing beyond about +-30% rate or
  40 Hz, and mania at YMRS 60 asks for both. Measured features moving on
  command is the whole claim.
- **Not clinical validity of any kind.** Page 09's formulas are one page's
  hand-written anchors, not fitted to clinical speech; a render that meets
  them is a render that meets them. Nothing here is a depressed or manic voice.
- **Not page 09's disentanglement ablation** (§4.4): that needs raters.
- **One TTS, two held-out voices, eight sentences.** Offsets are in Hz, as
  page 09 writes them, so the same score moves a lower voice further in
  semitones.

### Deviations

- **2026-09-24, registry row.** Its "Registered" cell was changed to the
  registering commit's hash, `6c43d5e`, and its status updated, as for the
  earlier entries.

### Results

**Run:** 2026-09-24 18:15, commit `6c43d5e` (the registering commit), clean
tree, laptop CPU (WSL), Kokoro-82M; all 32 sweeps, first and only run:
`PYTHONPATH=src python -m s2slab.severity test --confirm PR-009 | tee
results/severity/pr009-report.txt`. Run file `runs/severity/pr009.json`
(every target and measure, every neutral clip's SHA-256); the printed output
as `runs/severity/pr009-report.md` (standard output only -- the libraries'
warnings went to the terminal). \$0.

The criteria, verbatim:

```
PR-009, the registered criteria (docs/preregistration.md):
  S1  page 09's monotonicity test -- pitch offset, rate, pauses in order, every sweep: 96 of 96   PASS
  S2  pitch range in order, at least 28 of 32: 29 of 32   PASS
  S3  rate and pauses within 10%, every check: 384 of 384   PASS
  S4  pitch offset within 3 Hz or 10%, at least 90%: 169 of 192 (88.0%)   FAIL
  S5  pitch range within 10%, at least 80%: 177 of 192 (92.2%)   PASS
PR-009: NOT SUPPORTED -- failed S4
```

and the report's summary lines above it, verbatim (its pitch-offset line uses
the module's own 3 Hz rule, not S4's):

```
monotonicity: 125 of 128 feature sweeps in order (a sweep = one voice, sentence, condition and feature, every step)
within tolerance, f0_offset_hz: 166 of 192
within tolerance, f0_dev_scale: 177 of 192
within tolerance, rate: 192 of 192
within tolerance, pause_scale: 192 of 192
```

**Verdict: not supported.** S4 failed: 169 of 192 pitch-offset checks within
tolerance, 88.0%, where 90% -- 173 checks -- was needed; four short. The
other four held. Page 09's own test, S1, passed on every sweep: raising the
severity moved the pitch offset, the rate and the pauses the way the mapping
says at every step, 96 of 96. Rate and pauses met every target (384 of 384).
The pitch range, the weak feature on dev, passed both of its bars: 29 of 32
sweeps in order, all three out of order under mania; 92.2% within 10%.

**Where S4 failed** -- pitch offset, measured minus target, over the 16
sweeps at each score:

| condition, score | target | mean error | largest | misses |
|---|---:|---:|---:|---:|
| depression 0 · 5 · 10 · 15 · 20 · 27 | -5.0 to -25.0 Hz | +0.4 to +1.0 Hz | +3.2 | 1 (PHQ-9 10) |
| mania 0 | 0 | +0.8 | +2.2 | 0 |
| mania 12 | +9.0 | +0.9 | +5.4 | 1 |
| mania 20 | +15.0 | +1.4 | +6.3 | 3 |
| mania 26 | +19.5 | +1.5 | +6.4 | 3 |
| mania 40 | +30.0 | +2.7 | +11.9 | 7 |
| mania 60 | +45.0 | +5.3 | +11.8 | 8 |

The error leans one way: 159 of the 192 measured medians sit above their
target, and the mean error is above zero at every score of both conditions --
about a hertz on depression, more the higher mania raises the pitch. 22 of the 23 misses are mania. Two clips
carry 11 of them: `am_adam:t8` (6, including the one depression miss, -9.2 Hz
against -12.4) and `af_sarah:t4` (5). By voice: `af_sarah` 86 of 96,
`am_adam` 83 of 96.

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| S1 and S3 hold: 96 of 96, 384 of 384 | Right. |
| S2 holds narrowly, 29 to 31 of 32; any sweep out of order is range, under mania | Right: 29 of 32, all three mania. |
| S4 holds, about 94%; the misses are mania at YMRS 40 and 60, overshooting by 3 to 9 Hz | **Wrong:** 88.0%, S4 failed. Misses from YMRS 12 up, and one on depression; overshoot to +11.9 Hz. |
| S5 about 85-90% | **Wrong, the good way:** 92.2%. |
| `am_adam` misses more range checks than `af_sarah` | Right by one: 8 against 7. |
| Supported | **Wrong.** |

**What the registration got wrong.** S4's bar of 90% sat three and six points
under dev's 92.7% and 95.8% -- two runs of 96 checks on four sentences. On
eight new sentences the pitch overshot more, and two clips carried half the
misses. A small dev set read as the population again, the fifth time (PR-004,
PR-005, PR-007, PR-008): this time in a tolerance rather than a direction.
The bar is kept as registered; the render fails it. What held is the claim
page 09 itself makes -- every feature moves the right way at every step --
and the duration edit: rate within 0.7% of its target on every check, pauses
within 8.3%.

**Reported alongside, descriptively (after the verdict)**

- The offset error leans upward and grows with the upward shift and the
  speed-up together (mania 60 is +45 Hz at 1.8x), which points at the render
  or the measure rather than at chance. Which one is not tested here: the
  resynthesis at high speed, the median as the measure, or the compression
  centred on the measured median are all candidates. Exploratory, and the
  natural first question for any second version.
- Neutral clips, 3.75 to 4.76 s; every one had its sentence-break pause.

## PR-010 — Does a LoRA move Moshi's turn-taking toward the dialogues it was trained on?

**Registered:** 2026-09-25, by the commit that adds this entry.
**Author:** Ram Charan

### Background

v2 plan, FT-1 (Notion page 17 §7.1): "Instead [of a patient model]: measure overlap ratio and
turn-gap distribution before and after LoRA, against a human-human reference. Pre-registered
go/no-go. Success: the fine-tuned model's overlap-ratio distribution is closer to human than the
base model's, by a bootstrap CI excluding zero. Failure: no movement, or movement away. If it
fails, report it."

**The reference, and what it can claim.** FT-1 trains on kyutai/DailyTalkContiguous (CC-BY-SA
4.0, revision `33e1b50`), the plan's own data: 2,541 dialogues read aloud by two speakers, one per
channel. It is read speech, recorded line by line and joined, so its timing is constructed, not
spontaneous. The reference here is therefore held-out DailyTalk itself, and the claim is worded
accordingly: **toward the timing of the dialogues it was trained on**, not "toward human". A
spontaneous corpus (CANDOR, Fisher) was considered and set aside for cost and access.

**The measure** (`src/s2slab/turngaps.py`, `fae0e21`; after Heldner & Edlund 2010): each
channel's inter-pausal units (voiced frames joined across silences under 200 ms), then every time
the answerer takes the floor from the user, its floor-transfer offset -- a gap if positive, an
overlap if negative. Two offset distributions are compared by their Wasserstein-1 distance, in ms.
For the reference, the answerer is the human on the LEFT channel (the side moshi-finetune trains
Moshi to be) and the user is the RIGHT; for a model, the model hears the right channel through the
full-duplex driver (`duplex.converse`, no lead-in, 4 s of tail, seeded) and answers it.

Development data, **the first 50 dev-fold dialogues** (numbers ending in 1; `97c3a06`, Modal A10,
\$0.335):

| 50 dev dialogues | base Moshi | the reference (the human on the left) |
|---|---:|---:|
| turn changes | 210 | 197 |
| offset p10 / p50 / p90 | -500 / 160 / 240 ms | 200 / 420 / 920 ms |
| overlapping turn changes | 42 | 0 |
| speech inside the user's turn | 42 | 0 |
| overlap ratio, median | 0.107 | 0.000 |

Distance, base to reference: **478 ms**, 95% bootstrap interval over dialogues 406 to 564.
**DailyTalk has no overlap at all** -- every one of the 394 turn changes in both directions on dev
is a gap -- so the plan's overlap ratio cannot be "closer" except by falling to zero; the
registered measure is the turn-gap distribution instead, and overlap is reported beside it. The
measure's own noise: on the left channel, energy-based units and the transcript's word timings
give distributions 54 ms apart (197 against 196 turn changes, medians 420 and 415 ms).

**The training**, `vast/ft1.sh` with this entry: kyutai-labs/moshi-finetune at `2acc879`, moshi at
`9837ca3`, its example configuration unchanged -- LoRA rank 128, scaling 2, learning rate 2e-6,
2,000 steps of batch 16 on 100 s windows, seed 0 -- on every DailyTalk dialogue **except the test
fold** (`vast/ft1_prep.py manifest`; the training list is a file, so the dataset's full manifest
is never read). One GPU on Vast.ai, **on-demand** (never taken away mid-run); checkpoints every 100 steps all
the same, in case the machine fails. moshi-finetune cannot resume, so `ft1_prep.py patch` adds a warm start from the last checkpoint that restores the
adapter, the step and the learning-rate schedule, but **not** AdamW's moment estimates or the data
order (a resumed run reshuffles, seed = the resumed step). If any resume happens, it is listed
under Deviations with its step.

What was known, and by whom: the test dialogues have been neither downloaded nor looked at by
anyone; base Moshi has answered only the 50 dev dialogues; no training has run.

### Question

On 100 held-out DailyTalk dialogues, does Moshi with FT-1's LoRA take the floor with gaps closer
to the dialogues' own than base Moshi does?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | As committed with this entry: `src/s2slab/turngaps.py`; in `src/s2slab/dailytalk.py` `fold`, `load`, `reference`, `pooled`, `TEST_IDS`, `gain`, `pr010`, `pr010_validity`; `duplex.converse` and `duplex.stereo`; `r5.MoshiDuplex` and `r5.MoshiLoraDuplex`; `modal_app/r5_bench.py`'s image, `_dialogues`, `pr010_remote` and `pr010`; `vast/ft1.sh`, `vast/ft1_prep.py`, `vast/ft1.yaml`. Reporting (`report`, `format_pr010`) may gain lines, tested on made-up rows only. |
| **V2 · Held out** | The test fold (numbers ending in 0) is excluded from training: the run's `train.jsonl` and its SHA-256 come back with the adapter, and its count must be 2,541 minus 255. The 100 test dialogues (0, 10, ..., 990) are downloaded only after this entry is committed, and `dailytalk.load` refuses them without `confirm="PR-010"`. |
| **V3 · The adapter** | The one evaluated is the checkpoint at step 2,000. If the run cannot reach it (the GPU lost for good, the budget spent), the last checkpoint is evaluated and that is a deviation. Its SHA-256 is recorded in the run file. |
| **V4 · One run, one container** | `modal run modal_app/r5_bench.py::pr010 --confirm PR-010`, from a clean tree: base Moshi, then Moshi + LoRA, in one A10 container, on the same 100 dialogues with the same seeds. A crashed run writes no run file; it is reported and repeated once, in full. |
| **V5 · Evidence** | `runs/ft1/pr010.json` (every dialogue's offsets for both models and the reference, the adapter's SHA-256 and config, the criterion) and the printed report, committed with the results, together with the training evidence that came back with the adapter (the config actually run, the training log, the list and its SHA-256). The adapter itself and the listening files stay out of git. |

### Primary outcome

For the 100 test dialogues, pooled: the Wasserstein-1 distance between each model's offsets and
the reference's; their difference, base minus LoRA (the **gain**); and its 95% bootstrap interval,
10,000 resamples of the dialogues, seed 0, both models on each resample (`dailytalk.pr010`).

### Success criterion

**Supported if the gain is above zero and the whole 95% interval is above zero** -- the LoRA's
turn-taking is closer to the dialogues it was trained on than the base model's. Otherwise not
supported: no movement, or movement away, reported as such.

### Predictions, recorded but not criteria

- **Supported.** The distance falls from about 480 ms (dev, base) to about 250 ms; the gain is
  around +200 ms with an interval well clear of zero.
- **Overlap falls sharply** -- DailyTalk has none: overlapping turn changes from about 1 in 5 to
  under 1 in 20, and speech inside the user's turn by more than half.
- **The median gap rises** from about 160 ms toward 300 ms, still short of the reference's 420.
- The risk named in advance: the example's learning rate (2e-6) is low; the LoRA may move little.
  A gain inside the interval would say what 2,000 steps at that rate can and cannot do.

### Reported alongside, descriptively

Both models against the reference: turn changes, offset percentiles, overlapping turn changes,
speech inside the user's turn, overlap ratio; the training's loss curve, steps reached, resumes,
wall time and cost; a few conversations to listen to (user left, model right).

### Analysis plan

1. Commit this entry with the frozen code above.
2. Download the 100 test dialogues. Train on Vast.ai (`vast/README.md`); copy the adapter off; put it
   on Modal's `ft1` volume.
3. The one run (V4); save the evidence (V5); record the output below, verbatim.
4. Only after step 3: any further look, labelled exploratory.

### What this cannot show

- **Not "more human".** The reference is read dialogue, stitched line by line; moving toward it
  is moving toward DailyTalk's timing.
- **Not a live conversation.** The user channel was recorded to fit the other speaker, so it keeps
  talking on schedule whatever the model does: an open-loop probe, the same for both models.
- **Not the content.** Only timing is measured; what the model says is not scored.
- **One LoRA, one configuration, one seed.**

### Deviations

- **2026-09-25, the first registered run crashed; the one repeat runs on an L40S.** The run
  (`modal run modal_app/r5_bench.py::pr010 --confirm PR-010`, commit `4df9ef3`, Modal A10) finished
  base Moshi on the 100 test dialogues, then ran out of GPU memory while loading Moshi + LoRA: fusing
  the LoRA into the weights found 21.5 GiB of the A10's 22 GiB already allocated, where one model with
  its LoRA needs about 16.2 GiB (base Moshi's own peak in R5, 16,621 MiB) -- so about 5 GiB from the base pass
  was still held. The function raised before returning, so no run file was written and no measure
  of either model was seen; only the error was printed. V4 allows one repeat, in full. The only
  change for it (`pr010_remote`, V1's frozen code, changed and disclosed here): it runs on an **L40S**
  (48 GB) instead of the A10, and records the memory the base pass leaves allocated. Both models
  still run in one container, on the same dialogues and seeds, so the comparison stays paired; the
  stream-time measures do not depend on the card's speed. The crash cost about \$0.6 of A10 time
  (in the ledger with the results).
- **Training, for the record** (it is not a deviation): the run reached step 2,000 on one H100 in
  11 h 22 min with no resume. The first launch crashed before step 1 (moshi-finetune reads
  `CUDA_VISIBLE_DEVICES`, which the machine did not set) and was relaunched with the same config;
  both launch configs came back with the adapter and are identical.
- **2026-09-25, a correction to the first entry, from the repeat's own numbers.** Its diagnosis
  was wrong. The repeat recorded 16 MiB still allocated after the base pass -- nothing held -- and
  Moshi + LoRA peaking at 29,962 MiB (29.3 GiB) on its own, against base Moshi's 16,677. Loading
  and fusing the LoRA, not memory left over, is what outgrew the A10's 22 GiB: no A10 run could
  have held it. The L40S was the right fix, for a different reason. The first entry is left as
  written; the comment above `pr010_remote` is corrected to match (a comment only, after the run).
- **2026-09-25, registry row.** Its "Registered" cell was changed to the registering commit's
  hash, `4df9ef3`, and its status updated, as for the earlier entries.

### Results

**Run:** 2026-09-25, about 21:35 to 22:37 CDT; commit `3d36f74` (the deviation's commit), clean
tree; Modal NVIDIA L40S, torch 2.8.0+cu126, moshi 0.2.13; the one repeat V4 allows:
`modal run modal_app/r5_bench.py::pr010 --confirm PR-010 2>&1 | tee results/ft1-pr010.txt`. The
adapter is the step-2,000 checkpoint, SHA-256 `7a48bd58...0dc3`, the same as the SHA256SUMS written
on the training machine (V3). Run file `runs/ft1/pr010.json` (every dialogue's offsets and text for
both models, the reference, the adapter's SHA-256 and config, the criterion); the printed output as
`runs/ft1/pr010-report.md`, from the machine line to the cost line (Modal's progress lines and
moshi's warning about the repository's missing `config.json` left out); the training evidence in
`runs/ft1/training/` (V5). 3,658 s inside the function, about \$2.01. Peak GPU memory: base
16,677 MiB, Moshi + LoRA 29,962 MiB; left after the base pass, 16 MiB.

The criterion, verbatim (recomputed from the run file's rows by the frozen `dailytalk.pr010`: the
same numbers):

```
PR-010, the registered criterion (docs/preregistration.md), 100 test dialogues:
  distance to the reference: base 496 ms, LoRA 1035 ms
  gain (base - LoRA): -539 ms, 95% bootstrap interval -685 to -402
PR-010: NOT SUPPORTED -- the LoRA is not closer to the reference beyond the interval
```

**Verdict: not supported -- and the movement is away.** Moshi + LoRA's gaps are 1,035 ms from the
reference's, base Moshi's 496: a gain of -539 ms, with the whole 95% interval, -685 to -402, below
zero. The LoRA cured what base Moshi does wrong -- it all but stopped talking over the user -- and
went past the target: it now waits about three times as long as the people it was trained on.

| 100 test dialogues | base Moshi | Moshi + LoRA | reference (the human on the left) |
|---|---:|---:|---:|
| turn changes | 491 | 355 | 456 |
| offset p10 / p50 / p90, ms | -400 / 160 / 220 | 508 / 1,220 / 2,752 | 250 / 420 / 830 |
| offset p25 / p75, ms | 100 / 200 | 740 / 1,870 | 320 / 580 |
| gaps over 1 s (of those, over 2 s) | 28 (23) | 211 (78) | 20 (0) |
| overlapping turn changes | 85 | 2 | 0 |
| speech inside the user's turn | 98 | 4 | 0 |
| overlap ratio, median over dialogues | 0.122 | 0.030 | 0.000 |
| the answerer's speech, s | 1,182 | 732 | 1,252 |

The LoRA's overlap ratio is not zero although it began inside the user's speech only 6 times: the
recorded user keeps to its own schedule and starts over a late answer.

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Supported: the distance falls from about 480 to about 250 ms; gain about +200, interval well clear of zero | **Wrong:** base 496 (dev 478, as expected), LoRA 1,035; gain -539, interval -685 to -402. |
| Overlap falls sharply: overlapping turn changes from about 1 in 5 to under 1 in 20; speech inside the user's turn by more than half | Right: 85 of 491 (1 in 6) to 2 of 355; 98 to 4. |
| The median gap rises from about 160 toward 300 ms, still short of 420 | **Wrong:** 160 to 1,220 ms, nearly three times the reference's. |
| The risk: at 2e-6 the LoRA may move little | **Wrong:** it moved further than the whole distance, past the target. |

**What the registration got wrong.** The predictions pictured a small push along one line, from
base Moshi toward DailyTalk, with the example's learning rate too weak to push far. The LoRA moved
further than the whole distance, and in a shape nothing predicted: it says less (2,537 words of
text stream against base Moshi's 4,723; 732 s of speech against the reference's 1,252), takes the
floor 101 fewer times than the reference, and in 55 of the 100 dialogues takes fewer turns than the
human did. The registration went from training straight to the test, with no look at the trained
adapter on dev; the 50 dev dialogues would have cost about \$0.35. For FT-2 and FT-3: register the
training, run the adapter on dev, then register the test.

**Reported alongside, descriptively (after the verdict)**

- **The training** (`runs/ft1/training/`): 2,000 steps on one H100 80 GB (Vast.ai, on-demand),
  11 h 22 min, no resume, peak 39.6 GB. Mean loss 1.674 over steps 1-50, 1.214 over 451-500, 1.153
  over 951-1,000, 1.137 over the last 100 (the last step, 1.203). About 20 s a step: four times
  the plan's 3-hour estimate (page 17 §7.1). Training list: 2,286 dialogues, 2,541 minus the 255 of
  the test fold (V2), SHA-256 `427ffb51...cbbe`. Cost \$24.29 on Vast's bill, four times the
  plan's \$5.86 for Vast.
- **What it says changed as well as when.** Base Moshi opens 98 of the 100 conversations with a
  greeting ("Hello, how is it going?" in 68); Moshi + LoRA opens none. In two dialogues its text
  stream is empty. Timing is the only thing measured; the content is not scored.
- **Dialogue by dialogue:** the LoRA's median gap is nearer the human's than base Moshi's in 18 of
  the 100.
- **Why it overshot is not tested here.** Candidates, exploratory: what 2,000 steps of read,
  line-by-line dialogue teach about when to start; the open-loop probe (a user who does not wait
  cuts a late answer short and removes turn changes -- though the probe is the same for both
  models); decoding settings kept from base Moshi. The natural first check is the dev set, at
  checkpoints 500, 1,000 and 1,500 -- which the run did not keep (`num_ckpt_keep: 3`).
- Six conversations to listen to, user left, model right (`results/ft1/listen/`, not in git):
  dialogues 0, 10 and 20, each with both models. In dialogue 10 the LoRA's only words are
  "Goodbye, Mary.".

---

## PR-011 — Does a verbatim LoRA make small.en write down disfluent speech as spoken?

**Registered:** 2026-09-27, by the commit that adds this entry.
**Author:** Ram Charan

### Background

v2 plan, FT-2 (Notion page 17 §7): "ASR LoRA on SEP-28k, via Tier-B pseudo-labels" -- the
highest-ROI item in the WER playbook. Two facts changed the recipe, both checked before any code
was written (ft/README.md). SEP-28k has no transcripts, its annotations are CC BY-NC and its audio
stays the podcasts' copyright; and pseudo-labels from the slow tier would teach the student the
slow tier's habit, which is the very thing the disfluent cohort exists to measure: **Whisper cleans
speech up.** On the eval set's disfluent cohort `small.en` scores 9.9% raw against 7.6%
normalized (PR-004's run), and the gap is the fillers and repetitions it drops. So FT-2's target
is **verbatim transcription**: a LoRA on `whisper-small.en` (R2's slow tier, `r0-gpu-bigger`'s
recognizer, 244M parameters) trained on transcripts that keep the "uh"s, the "the, the"s and the
restarts.

**The data**, and what it can claim. amaai-lab/DisfluencySpeech (Apache-2.0, revision `b7da294`,
the eval set's own pin): ONE speaker re-performing Switchboard utterances in a studio, 4,500
train / 250 validation / 250 test clips, four transcripts each; `transcript_a` is the verbatim
one and the eval set's reference for its disfluent cohort. The training used the TRAIN split
only; the datamodule has no test loader and lists no test file (`ft/src/data/disfluencyspeech_datamodule.py`).
**The eval set's disfluent cohort is 50 clips of the TEST split -- the same speaker, the same
studio.** So S1 and S2 below measure same-speaker adaptation to a verbatim target, and the entry
says so; the other 150 clips of the eval set (LibriSpeech clean, EdAcc accented, the slowed
proxy) are other speakers, and that is where the guardrail is. No open corpus of *other*
disfluent speakers with verbatim transcripts was found (SEP-28k: no transcripts; LibriStutter:
synthetic stutters, CC BY-NC, no verbatim reference; FluencyBank: membership), so there is no
leak-free disfluent test here, and the claim is worded to that.

**The model.** PEFT LoRA r = 32, α = 64, dropout 0.05 on q and v projections (PEFT's own Whisper
recipe): 3,538,944 trainable of 245,273,088 parameters, 1.44%. AdamW 2e-4, 50 warm-up steps then
linear decay, batch 16, 900 steps (about 3.2 epochs), bf16, seed 0; every 150 steps a validation
on 64 clips and the adapter saved. Run `train-f37d0df-20260926-235222` on a Modal A10, commit
`f37d0df`, 545 s, \$0.168 (`ft/`, the Lightning-Hydra project; `modal_app/ft2_train.py::train`).
It converged early: raw WER on the 64 clips 2.4% at step 150, 2.0% at 900; the loss 0.177 → 0.164.
**The adapter under test is `step-000900`, the run's end -- fixed before any look at it against
the base**, merged into the base weights and converted to CTranslate2 float16 (`ft/src/export.py`),
so the bench loads it exactly as it loads Systran's `small.en`. Export
`export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16` on the
Modal volume `ft2`; `ct2/model.bin` SHA-256
`b31833fe4ed0e1cb429d33a52d81e9974fe32838e32e9a85f19e666af2a51cff`.

**Development -- the dev look FT-1 never had** (all 250 validation clips, the training speaker,
`::evaluate`, A10, \$0.014 each):

| 250 validation clips | base `small.en` | + LoRA, step 900 |
|---|---:|---:|
| raw WER | 10.9% | **2.1%** |
| normalized WER | 7.8% | 2.1% |
| fillers kept, of 176 in the references | 8 (4.5%) | 174 (98.9%) |
| validation loss | 0.728 | 0.174 |

Raw and normalized WER meet at 2.1%: the model writes what the normalizer would have deleted.
How much of the 8.8-point drop is the verbatim target and how much is the speaker cannot be told
apart on this data, and this entry does not try.

**The comparator's known numbers** on the eval set (PR-004's run, one T4, commit `91048b3`,
`small.en` as `r2-gpu`'s slow tier, raw / normalized): disfluent 9.9 / 7.6%, clean 3.5 / 3.3%,
accented 27.0 / 23.9%, slow 3.7 / 3.5%. The bench re-runs `small.en` beside the LoRA rather than
reusing them (V1), but they fix the guardrail's bar without a look at the test cohorts.

What was known, and by whom: the LoRA has met none of the eval set's 200 clips; the bench has run
on 8 clean clips with the base model only (`results/ft2/bench-small.en-limit8-20260927-001214.json`,
raw 5.1%); the disfluent references were counted (28 fillers over the 50 clips) to set S2's bar.

### Question

On the frozen eval set, does `small.en` + FT-2's LoRA transcribe the 50 disfluent clips closer
to their verbatim references than `small.en` does -- keeping the fillers -- without costing the
other 150 clips, which are other speakers, more than a point of normalized WER?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | As committed with this entry: `src/s2slab/ft2.py` (the criterion); `src/s2slab/wer.py` and `evalset.py` (unchanged since the ladder's rows); `s2slab.r0.WhisperASR` (the rungs' recognizer class); `modal_app/ft2_train.py`'s image, `bench_remote`, `bench_guard` and `bench`; `ft/src/export.py`. Reporting (`format_pr011`, the tables) may gain lines, tested on made-up transcripts only. |
| **V2 · Held out** | The frozen eval set (`evalset/manifest.json`, verified on the container before any clip is read). The LoRA was trained on DisfluencySpeech's train split only, and the exported model has never been run on any eval-set clip. |
| **V3 · The model** | The step-900 adapter of run `train-f37d0df-20260926-235222`, merged and converted; the run records the export's name and `ct2/model.bin`'s SHA-256, which must equal the one above. |
| **V4 · One run, one container** | `modal run modal_app/ft2_train.py::bench --export export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16 --confirm PR-011`, from a clean tree: `small.en` first, then the exported model, over all 200 clips, same T4 container, the same `WhisperASR` settings the rungs use (float16, greedy, no context between clips). A crashed run writes no run file; it is reported and repeated once, in full. |
| **V5 · Evidence** | `runs/ft2/pr011.json` (both models' transcripts and per-clip recognizer times, the criteria, the export's SHA-256) and the printed report, committed with the results, together with the training evidence brought back from the volume: the run's six `val-step-*.jsonl` (every validation's transcripts), its `metrics.csv` and hydra config, and both dev-look `metrics.json`. The adapter, the merged model and the dataset stay out of git. |

### Primary outcome

`ft2.pr011(base, lora)` on the two transcript sets, as coded at this commit:
- **S1** raw WER on the 50 disfluent clips, LoRA minus base, in points, with the 95% bootstrap
  interval over clips (10,000 resamples, seed 0; corpus WER recomputed on each resample).
- **S2** of the 28 fillers in the disfluent references, how many the LoRA's transcripts keep
  (word for word, up to the reference's count).
- **S3** normalized WER over the other 150 clips (clean, accented, slow), LoRA minus base, in
  points, with the same bootstrap.

### Success criterion

**Supported only if all three hold:** S1's whole interval below zero; S2 at least 90% (26 of 28;
raised from the 80% first coded, because the dev look showed 98.9% and 28 fillers make 80% a bar
of 23 that a broken model could clear -- changed in this commit, before the run); S3's interval
upper end at most +1.0 point. Otherwise not supported, and which criterion failed is reported.

### Predictions, recorded but not criteria

- **Supported.** S1: base about 10% raw, the LoRA about 3% -- a drop of 6 to 8 points with an
  interval wholly below zero (the dev look gave 10.9 → 2.1 on 250 clips of the same speaker; the
  test clips are the same speaker and the same recording set-up).
- **S2: 26 to 28 of 28.** The dev look kept 174 of 176.
- **S3 is the criterion most likely to fail**, and the prediction is that it holds narrowly:
  normalized WER on the other 150 clips rises by 0.2 to 0.6 points -- a LoRA trained on one
  speaker's read-aloud studio speech, applied to LibriSpeech, EdAcc and the slowed clips. If it
  fails, the failure comes from the accented cohort, whose WER is the highest and whose speakers
  are furthest from the training voice.
- **Raw WER on the other cohorts rises more than normalized**: the LoRA writes a filler where a
  fluent speaker made a noise the base ignored. Descriptive.
- The per-clip recognizer time of the merged model is the base's, within noise: LoRA weights
  merged add no parameters.

### Reported alongside, descriptively

Both models against the references, per cohort, raw and normalized (the eval set's own table);
the fillers each model kept; recognizer time p50 per model; the training's validation curve (six
points, 64 clips) and the two dev looks; what the LoRA wrote where the base wrote nothing, on a
few disfluent clips.

### Analysis plan

1. Commit this entry with the frozen code above (S2's bar included).
2. The one run (V4); save the evidence (V5); record the output below, verbatim.
3. Only after step 2: any further look, labelled exploratory.

### What this cannot show

- **Not generalisation to disfluent speakers.** One speaker's performed disfluency trained on,
  the same speaker's held-out clips tested. S3 says whether other speakers' fluent speech was
  harmed; nothing here says whether another disfluent speaker would be transcribed verbatim.
- **Not stuttering.** Fillers, repetitions and restarts, performed; no blocks, no prolongations.
- **One LoRA, one configuration, one seed, 900 steps.**
- **Not the ladder's latency.** The bench times the recognizer alone on a T4; the rungs' first-audio
  numbers are untouched.

### Deviations

- **2026-09-26, the filler count.** The entry says the disfluent references hold 28 fillers and
  sets S2's bar as "26 of 28"; the frozen code counts **29** (the entry's count came from a crude
  split on spaces and commas, made while writing; `ft2.fillers` uses the raw normalizer), so the
  bar as coded was 27 of 29. The LoRA kept 29 of 29; the verdict does not depend on it.
- **2026-09-26, the premise of S3.** The entry calls the other 150 clips "other speakers, fluent
  speech". EdAcc's 50 accented clips are other speakers, but their references are verbatim
  conversational speech -- 28 fillers and 16 immediate word repetitions across the 50 -- so S3
  was not only a guardrail against harm on fluent speech: on the accented cohort it also
  measured the verbatim target directly. The criterion stands as registered (it asked whether
  normalized WER rose by more than a point, and it fell); the reading of the gain is in Results,
  and the clean + slow cohorts alone (read speech, no fillers) are reported as the fluent guardrail
  the entry meant, labelled exploratory.
- **2026-09-26, registry row and dates.** Its "Registered" cell was changed to the registering commit's
  hash, `65c833c`, and its status updated, as for the earlier entries. The entry's "Registered:
  2026-09-27" is the UTC date; the commit and the run were the evening of 2026-09-26, CDT, the
  clock the ledger and the other entries use.

### Results

**Run:** 2026-09-26, 22:03 CDT, commit `65c833c` (the registering commit), clean tree, Modal Tesla T4 (driver
610.57.04), one container: `small.en` then the exported model over all 200 clips, the first and
only run: `modal run modal_app/ft2_train.py::bench --export export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16 --confirm PR-011 2>&1 | tee results/ft2-pr011.txt`.
The exported model's `ct2/model.bin` SHA-256 on the container, `b31833fe4ed0...a51cff`, equals the
registered one (V3). Run file `runs/ft2/pr011.json` (both models' transcripts and per-clip times,
the criteria, the SHA-256); the printed output as `runs/ft2/pr011-report.md`; the training
evidence under `runs/ft2/training/` (the six validations' transcripts, `metrics.csv`, the
hyperparameters and the hydra config), the two dev looks under `runs/ft2/dev-look/`, and the
training, export and smoke-test summaries under `runs/ft2/summaries/` (V5). 67 s inside the
function, \$0.012.

The criteria, verbatim:

```
PR-011, the registered criteria (docs/preregistration.md), 200 eval-set clips:
  S1  raw WER on the 50 disfluent clips: base 9.9%, LoRA 2.3%; -7.6 points, 95% interval -9.6 to -5.7   PASS
  S2  fillers kept on those clips: LoRA 29 of 29 (100.0%; base 1), bar 90%   PASS
  S3  normalized WER on the 150 other clips: base 10.4%, LoRA 8.0%; -2.3 points, interval -3.8 to -0.9, allowed up to +1.0   PASS
PR-011: SUPPORTED
```

**Verdict: supported.** All three held, S3 with room to spare in the direction nobody predicted.

| 50 clips a cohort, raw / normalized WER | `small.en` | + FT-2's LoRA | errors, base → LoRA (normalized S / D / I) |
|---|---:|---:|---|
| disfluent (the training speaker) | 9.9 / 7.6% | **2.3 / 2.4%** | 21 / 57 / 3 → 15 / 5 / 6 |
| accented (EdAcc) | 26.7 / 23.5% | **18.3 / 17.6%** | 68 / 125 / 18 → 76 / 60 / 22 |
| clean (LibriSpeech) | 3.5 / 3.3% | 3.0 / 2.8% | 22 / 2 / 4 → 20 / 2 / 2 |
| slow (the 0.75× proxy) | 3.7 / 3.5% | 3.2 / 3.1% | 25 / 1 / 4 → 22 / 2 / 2 |
| **all 200** | 11.1 / 9.6% | **6.6 / 6.4%** | |

Base `small.en` reproduced PR-004's numbers (its run through the rung, five days earlier, another
T4): disfluent, clean and slow to the decimal, accented within 0.4 points. Recognizer time p50
143.6 against 144.7 ms, p90 188 against 190: the merged adapter costs nothing at inference.

**What the gains are.** On the training speaker the base model's errors were **deletions**: 57
of its 81 normalized errors, the fillers and repetitions it left out ("And they have a policy for
testing" for "And, uh, they have a policy for, uh, testing"). The LoRA writes them down: 5
deletions left, and raw and normalized WER meet at 2.3 / 2.4%. On EdAcc the same thing
happened: 125 of the base's 211 normalized errors were deletions, because EdAcc's references keep
the "uh"s, the "mhm"s and the "I, I think"s of conversation, and `small.en` does not; the LoRA
halved them (125 → 60) at the price of eight more substitutions. So most of the 5.9-point
accented gain is the verbatim target reaching a second corpus with verbatim references, **not**
new robustness to accents -- 29 accented clips got better, 9 worse, 12 did not change. On the two
read-speech cohorts, which have no fillers to keep, the LoRA changed little: clean + slow
together, exploratory, 3.4 → 2.9% normalized, −0.47 points, interval −1.34 to +0.29 -- no harm,
and no gain that the interval will vouch for. That interval is the fluent guardrail the entry
meant to register.

**Predictions, scored**

| Prediction | Outcome |
|---|---|
| Supported | Right. |
| S1: base about 10% raw, the LoRA about 3%, a drop of 6 to 8 points | Right: 9.9 → 2.3%, −7.6. |
| S2: 26 to 28 of 28 | Right in substance, wrong count: 29 of 29. |
| S3 most likely to fail, holding narrowly at +0.2 to +0.6 | **Wrong**, the good way: −2.3 points, interval −3.8 to −0.9. The reason is in the Deviations: EdAcc's references are verbatim, which the entry did not know. |
| A failure of S3 would come from the accented cohort | Not tested (S3 held); the accented cohort was where the change was, downward. |
| Raw WER on the other cohorts rises more than normalized (fillers written where none were spoken) | **Wrong.** Raw fell more than normalized on accented (−8.4 against −5.9), because the references had the fillers; on clean and slow the LoRA wrote no filler at all (0 in 100 clips, 0 in the references). |
| Recognizer time unchanged | Right: 144 against 145 ms p50. |

**What the registration got wrong.** Not the direction this time, and not a dev set read as the
population: the registration did not check what the guardrail cohorts' references contain. One
line of the eval set's README ("EdAcc ... conversational") and a count of fillers in
`evalset/manifest.json` -- the same count made for the disfluent cohort -- would have shown that
S3 mixed two things. The criterion still did its job (it would have caught harm), but the entry's
"other speakers, fluent speech" was half true, and the S3 prediction was built on the wrong half.

**Reported alongside, descriptively (after the verdict)**

- **The training** (`runs/ft2/training/`): 900 steps, 545 s on an A10, \$0.168, bf16, batch 16;
  1.44% of the parameters trained. Validation on 64 clips: raw WER 2.4% at step 150, 2.1 / 2.2 /
  2.1 / 2.1 / 2.0% at 300 to 900; the model had learned the target by the first validation.
- **The dev looks** (`runs/ft2/dev-look/`, 250 validation clips of the training speaker): base
  10.9 / 7.8%, 8 of 176 fillers kept; the LoRA 2.1 / 2.1%, 174 of 176.
- **Where the LoRA is worse on the training speaker:** substitutions of near-homophones it now
  hears verbatim -- "lunch time but not to access" for "lunch time, but not to excess"
  (`disfluent-11`); 17 raw substitutions against the base's 22, so fewer, but of a different kind.
- **The plan's FT-2 line** was an RTX 4090 on Vast, 5 h, \$0.70; this was an A10 on Modal,
  9 minutes of training, \$0.17, and about \$0.25 for the whole milestone's GPU time including
  the smoke runs, the dev looks and the test (`docs/cost-ledger.md`). The reasons are in
  `ft/README.md`.

---

## PR-012 — Do PR-009's criteria hold when FastPitch is given the severity targets as its own inputs?

**Registered:** 2026-09-27, by the commit that adds this entry.
**Author:** Ram Charan

### Background

v2 plan, FT-3 (Notion page 17 §7): a TTS fine-tune whose prosody can be driven by page 09's
severity mapping. PR-009 tested that mapping on audio edited AFTER synthesis (Praat, on Kokoro's
output) and was not supported on one criterion: S1 96 of 96, S2 29 of 32, S3 384 of 384, S4 169
of 192 (88.0%, bar 90%), S5 177 of 192. FT-3 asks the same five questions of a model that takes
duration and pitch as INPUTS: NVIDIA NeMo FastPitch + HiFi-GAN (`nemo-toolkit[tts]==3.0.0`).

**The control** (`ft/src/ft3/control.py`, `synth.py`, `sweep.py`). The model predicts a duration
and a pitch per token; the targets change them and the changed values are forced through
`forward()`: speech tokens' durations divided by the rate; runs of pause tokens of 150 ms or more
multiplied by the pause scale; pitch scaled about a centre and shifted by offset / pitch std, in
the model's normalized units. The measures are PR-009's own (`severity.achieved`: Praat F0, median
and interquartile range; rate and pauses from the time map), against the model's neutral render.

**Two decisions made on dev audio** (the base voice, PR-009's four DEV sentences, 2026-09-26),
both frozen here: the pitch is scaled about the neutral render's MEASURED median, and the range is
asked for with a gain of 1.2, because the model answers a per-token scale with about 0.83 of it.

**The voices.** (1) The base: NGC `tts_en_fastpitch` 1.8.1 and `tts_hifigan` 1.0.0rc1, LJSpeech,
as downloaded. (2) The fine-tune: NeMo's own recipe (FastPitch_Finetuning.ipynb) on its 5-minute
cut of Hi-Fi TTS speaker 9017 -- 76 training clips (290.6 s), 2 validation. FastPitch 1,000 steps,
Adam 2e-4, batch 24, from the base (run `fastpitch-12bc015-20260927-140956`, Modal A10); HiFi-GAN
1,000 steps -- 500 generator updates, its step counter advances twice a batch -- lr 1e-5, batch
32, on the fine-tuned FastPitch's own mels with ground-truth alignment (run
`hifigan-84cf350-20260927-150730`). Pitch statistics of the speaker, extracted at the training's
30-512 Hz: mean 162.3 Hz, std 75.1 Hz (the tutorial states 152.3 and 64.0; the minimum sits on the
30 Hz floor, so tracking errors remain). Both trainings read the manifest's `text`, which is
lowercase and unpunctuated. The runs that failed on the way are in `docs/ft3-log.md`; one
extract was made at the wrong pitch range (65-2093 Hz) and never reached a training.

**Development** (both voices, the four DEV sentences, the frozen settings; PR-009's counts):

| 4 dev sentences | PR-009's bar | base voice | fine-tuned 9017 | pooled |
|---|---|---:|---:|---:|
| S1 offset, rate, pauses in order | all | 23 of 24 | 22 of 24 | 45 of 48 |
| S2 range in order | 28 of 32 | 5 of 8 | 0 of 8 | 5 of 16 |
| S3 rate and pauses within 10% | all | 93 of 96 | 87 of 96 | 180 of 192 |
| S4 pitch offset within 3 Hz or 10% | 90% | 33 of 48 | 23 of 48 | 56 of 96 (58.3%) |
| S5 pitch range within 10% | 80% | 31 of 48 | 11 of 48 | 42 of 96 (43.8%) |

Rate alone: 48 of 48 for each voice. The fine-tuned voice's range is unstable before any range
is asked for: at depression 0 (a 5 Hz shift, range target 1.00) it measured 1.27, 1.03, 0.80 and
1.02 on the four sentences, the base voice 1.06, 0.94, 1.05, 1.04; its neutral renders' F0
tracks hold 4 to 8 jumps of more than 40% between neighbouring frames. Its pause misses are all
in mania (15 of 24 within 10%; depression 24 of 24). Why the pitch is worse was not measured.
No setting was tuned on the fine-tuned voice.

What was known, and by whom: neither voice has spoken any of the eight sentences below; they were
written with this entry. PR-009's TEST sentences were not reused, so that set stays single-use.

### Question

On eight sentences neither voice has spoken, do renders made by giving page 09's targets to
FastPitch as its own inputs meet PR-009's five criteria, over the base voice and the 5-minute
fine-tuned voice together?

### Validity requirements

| | Requirement |
|---|---|
| **V1 · Frozen** | As committed with this entry: `ft/src/ft3/control.py`, `synth.py`, `sweep.run`, and in `ft/src/ft3/pr012.py`: `VOICES`, `SENTENCES`, `SETTINGS`, `model_files`, `run`, `by_voice`, `validity`, `registered_run`. In `src/s2slab/severity.py`, unchanged since PR-009: the targets, the measures, `in_order`, `within`, `criteria` and its thresholds. `format_criteria` and `severity.report` may gain lines, tested on made-up runs only. |
| **V2 · Held out** | The eight `SENTENCES` reach a model for the first time in the registered run. `registered_run` is the only path to them; `sweep`'s command line takes the dev sentences only. The tests use a fake voice. |
| **V3 · The models** | Four files, by SHA-256, checked before anything loads: base FastPitch `69bfdf02…9240dd`, base HiFi-GAN `e2f05884…804684`, fine-tuned FastPitch `afa0731b…77af60`, fine-tuned HiFi-GAN `1ebd9d77…5a0d4b` (in full in `pr012.VOICES`). |
| **V4 · One run, whole set** | All 32 sweeps (2 voices x 8 sentences x 2 conditions, six scores each), `PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.pr012 test --confirm PR-012`, laptop CPU, from a clean tree; the run refuses a dirty one. A crashed run writes no run file; it is reported here and repeated once, in full. |
| **V5 · Evidence** | `runs/ft3/pr012.json` (every target and measure, every neutral clip's SHA-256, the criteria pooled and per voice, the model files' SHA-256) and the printed report as `runs/ft3/pr012-report.md`, committed with the results, with the training evidence: the stages' run files and the dev sweeps. The renders, the models and the training audio stay out of git. |

### Primary outcome

`severity.criteria` on the 32 pooled sweeps, as coded at PR-009's commit and unchanged: S1 to S5.

### Success criterion

**Supported only if all five hold**, with PR-009's bars: S1 every one of the 96 offset, rate and
pause sweeps in order; S2 at least 28 of the 32 range sweeps in order; S3 every one of the 384 rate
and pause checks within 10%; S4 at least 90% of the 192 pitch-offset checks within 3 Hz or 10%;
S5 at least 80% of the 192 range checks within 10%. Otherwise not supported, and which failed is
reported.

### Predictions, recorded but not criteria

- **Not supported; all five criteria fail.** From the pooled dev counts: S1 86 to 94 of 96; S2
  6 to 14 of 32; S3 350 to 376 of 384; S4 50 to 68%; S5 35 to 55%.
- **Rate is within 10% on all 192 checks**, both voices: durations are integers of 11.6 ms
  frames and the model renders the frames it is given.
- **The base voice meets more S4 and S5 checks than the fine-tuned voice**, each of the two.
- **The fine-tuned voice's range is in order on at most 3 of its 16 sweeps.**
- **Every pause miss of more than 10% is in mania.**

### Reported alongside, descriptively

Each voice's five counts alone; PR-009's Praat-render counts beside them (other voices, other
sentences, the same measures and bars); the range measured at depression 0 for each sentence and
voice; each neutral render's median F0 and the number of pauses found.

### Analysis plan

1. Commit this entry with the frozen code above.
2. The one run (V4); save the evidence (V5); record the output below, verbatim.
3. Only after step 2: any further look, labelled exploratory.

### What this cannot show

- **Not that FastPitch cannot be controlled.** One control scheme (per-token scale and shift,
  one gain), two checkpoints, settings fixed on four sentences of one voice.
- **Not what fine-tuning does in general.** One speaker, five minutes, one recipe, one seed; the
  cause of the fine-tuned voice's pitch behaviour is not identified here.
- **Not a comparison of renderers on equal terms.** PR-009's numbers are Kokoro voices edited by
  Praat on other sentences; they are beside these, not against them.
- **Not naturalness, and not clinical validity.** No listening test; the measures say whether the
  targets were reached, not whether the speech sounds depressed or manic to anyone.

### Deviations

- **2026-09-27, the report file.** V5 names the printed report as evidence. NeMo writes its
  "restored from" messages to standard output, so the captured report begins with six lines of
  NeMo's, four of them with paths of the author's machine. `runs/ft3/pr012-report.md` leaves those
  six lines out and is otherwise the output, verbatim (170 lines). Nothing in them is a measure.
- **2026-09-27, the training evidence.** V5 promised "the stages' run files". The FastPitch
  training has none: it finished its 1,000 steps and failed in the packaging that followed, before
  its record was written, and the packaging run that saved the model lost its record the same way
  (`docs/ft3-log.md`, entries 7 and 8). What is committed instead: both trainings' validation
  losses as their logs printed them, the data stage's two run files and the vocoder stage's. The
  four model files are held by SHA-256 in the run file, which is what V3 asked.
- **2026-09-27, the Background's count of clips.** HiFi-GAN trained on 74 of the 76 training
  clips: its config prunes clips under 0.75 s, and two are. The entry said 76. FastPitch used 76.
- **2026-09-27, registry row.** Its "Registered" cell was changed to the registering commit's
  hash, `89edd05`, and its status updated, as for the earlier entries. Times here are CDT.

### Results

**Run:** 2026-09-27, finished 13:47 CDT, commit `89edd05` (the registering commit), clean tree,
laptop CPU (WSL), all 32 sweeps, first and only run; the four model files matched their SHA-256
before anything loaded. 490 s. \$0. Run file `runs/ft3/pr012.json`; report
`runs/ft3/pr012-report.md`; the renders and the exact neutral clips in `results/ft3/pr012/`
(208 files, ignored).

The criteria, verbatim:

```
PR-012, PR-009's criteria on FastPitch's renders (docs/preregistration.md):
  S1  pitch offset, rate, pauses in order, every sweep: 92 of 96   FAIL
  S2  pitch range in order, at least 28 of 32: 14 of 32   FAIL
  S3  rate and pauses within 10%, every check: 357 of 384   FAIL
  S4  pitch offset within 3 Hz or 10%, at least 90%: 109 of 192 (56.8%)   FAIL
  S5  pitch range within 10%, at least 80%: 115 of 192 (59.9%)   FAIL
PR-012: NOT SUPPORTED -- failed S1, S2, S3, S4, S5

each voice alone (descriptive; the bars are the pooled run's):
  tts_en_fastpitch_align   S1 47 of 48   S2 11 of 16   S3 175 of 192   S4 67 of 96   S5 78 of 96
  fastpitch_9017           S1 45 of 48   S2 3 of 16   S3 182 of 192   S4 42 of 96   S5 37 of 96
rate within 10%: 192 of 192
```

and the report's summary lines above it, verbatim (its pitch-offset line uses the module's own
3 Hz rule, not S4's):

```
monotonicity: 106 of 128 feature sweeps in order (a sweep = one voice, sentence, condition and feature, every step)
within tolerance, f0_offset_hz: 106 of 192
within tolerance, f0_dev_scale: 115 of 192
within tolerance, rate: 192 of 192
within tolerance, pause_scale: 165 of 192
```

**Verdict: not supported.** All five criteria failed. Giving page 09's targets to FastPitch as
its own inputs reaches the rate on every check and the pitch on little more than half of them.

**The predictions, scored.**

| Prediction | Outcome |
|---|---|
| Not supported, all five fail | **right** |
| S1 86 to 94 of 96 | right: 92 |
| S2 6 to 14 of 32 | right, at the upper edge: 14 |
| S3 350 to 376 of 384 | right: 357 |
| S4 50 to 68% | right: 56.8% |
| S5 35 to 55% | **wrong**: 59.9% |
| Rate within 10% on all 192 checks | right: 192 |
| The base voice meets more S4 and S5 checks than the fine-tuned voice | right: 67 against 42, 78 against 37 |
| The fine-tuned voice's range in order on at most 3 of its 16 sweeps | right, at the edge: 3 |
| Every pause miss of more than 10% is in mania | **wrong**: 8 of the 27 are in depression |

**What I got wrong.** Both misses come from reading four dev sentences as if they were the voice.
S5: the base voice met 31 of 48 range checks on dev (65%) and 78 of 96 here (81%); the range I
predicted was the dev estimate with a margin too small for four sentences. The pauses: on dev
every depression pause was on target for both voices; here the base voice missed 8, all on two
sentences (u7: 2.95, 3.57, 4.49 for 2.67, 3.22, 4.00; u8: every step from 5 up, 5.45 for 4.00).

**Reported alongside, descriptively.**

| | base voice | fine-tuned 9017 | PR-009 (Praat on Kokoro, other voices and sentences) |
|---|---:|---:|---:|
| S1 offset, rate, pauses in order | 47 of 48 | 45 of 48 | 96 of 96 |
| S2 range in order | 11 of 16 | 3 of 16 | 29 of 32 |
| S3 rate and pauses within 10% | 175 of 192 | 182 of 192 | 384 of 384 |
| S4 pitch offset within 3 Hz or 10% | 67 of 96 (69.8%) | 42 of 96 (43.8%) | 169 of 192 (88.0%) |
| S5 pitch range within 10% | 78 of 96 (81.3%) | 37 of 96 (38.5%) | 177 of 192 (92.2%) |

By condition, checks within tolerance of 48:

| | base, depression | base, mania | fine-tuned, depression | fine-tuned, mania |
|---|---:|---:|---:|---:|
| pitch offset | 44 | 23 | 18 | 24 |
| pitch range | 36 | 42 | 18 | 19 |
| rate | 48 | 48 | 48 | 48 |
| pauses | 40 | 39 | 48 | 38 |

| per sentence, u1 to u8 | base voice | fine-tuned 9017 |
|---|---|---|
| range measured at depression 0 (target 1.00) | 1.01, 0.95, 0.84, 1.00, 1.10, 1.03, 1.03, 0.99 | 1.12, 1.13, 0.70, 1.04, 0.74, 1.09, 0.77, 1.13 |
| the neutral render's median F0, Hz | 211.7, 203.1, 202.9, 220.4, 215.0, 201.5, 212.6, 205.1 | 144.2, 139.6, 155.6, 148.2, 150.3, 143.3, 149.3, 163.3 |
| pauses found | 2, 2, 3, 2, 2, 2, 2, 1 | 2, 2, 3, 3, 2, 3, 3, 3 |

**What the run shows, and no more.**

- **Rate is exact; it is an input the model obeys.** 192 of 192, both voices, as on dev.
- **The base voice's range is on target more often than its offset**: 78 of 96 against 67, and
  alone it is over S5's share (81.3%; no per-voice criterion was registered, and the pooled run
  failed). Its offset misses are mostly in mania: 23 of 48 on target against 44 in depression.
- **The fine-tuned voice is worse on every pitch count**, and its range is in order on 3 of 16
  sweeps. At depression 0, where only a 5 Hz shift is asked, its range measured 0.70 to 1.13; the
  base voice's 0.84 to 1.10. The two voices differ in speaker as well as in training, and the
  cause was not measured: this is a difference between two checkpoints, not an effect of
  fine-tuning.
- **The fine-tuned voice's pauses are better than the base voice's in depression** (48 of 48
  against 40), and about the same in mania. It trained on text without punctuation and finds as
  many pauses or more.
- **Shortened pauses miss low**: all 19 pause misses in mania are under the target (at YMRS 60,
  0.00 to 0.25 for 0.30). Durations are whole frames of 11.6 ms, and a short pause made shorter
  has few.
- **Not explained:** the base voice on u8. One pause was found; in mania at 40 and 60 the measured
  pause is 0.00, in depression it grows by more than the target at every step. Not looked into.
- **Beside PR-009:** the Praat edit of Kokoro's audio met S1, S2, S3 and S5 and missed S4 by four
  checks; FastPitch's own inputs met none. Other voices, other sentences: beside, not against.
