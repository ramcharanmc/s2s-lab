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

*None.*

### Results

*Not yet run.*