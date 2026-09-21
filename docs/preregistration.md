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
| PR-003 | On the author's real voice, does a barge-in policy that reads the caller's words (B2) make fewer wrong stop / keep-going decisions than the best duration threshold, and stop within 1.5 s? | The commit that adds this entry | Open |
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

*(none yet)*

### Results

*(to be appended)*