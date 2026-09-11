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
