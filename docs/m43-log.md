# M43 run log: the watermark on the synthetic voices

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m42-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-20 | — | the design (section 1); the author's go, scope B: the severity renders and the cascades' replies | — | — |
| 2 | 10-20 | `pip install audioseal==0.2.0` and its two models, the author, laptop | installed `audioseal 0.2.0`, `omegaconf 2.4.0`, `einops 0.8.2`; torch untouched (2.14.0+cpu); the models 56.1 MB and 33.1 MB, into `~/.cache/audioseal` | — | — |
| 3 | 10-20 | a smoke test outside the repo: a synthetic voice at 24 kHz, marked and read back | found as written (every frame, 16 of 16 bits) and after a 16-bit WAV; **not** found when read at 16 kHz, after a trip through 16 kHz or through Opus. Read at 24 kHz instead, it was found after both. The first mark 7.4 s (AudioSeal compiles its layers), later ones 0.23 s for 3 s of audio | `watermark.detect` reads at 24 kHz | — |
| 4 | 10-20 | the tests, laptop; the slow test with the real models | the wrapper, the development check and its rule, with a stand-in mark (a quiet 100 Hz tone); the real mark found, through 16 kHz too, and absent from the unmarked voice | `watermark.py`, `watermarkdev.py`, tests | b45bd26 |
| 5 | 10-20 | `python -m s2slab.watermarkdev`, the author, laptop, free, at b45bd26 (step 1a: the full mark) | found on 104 of 104 marked clips after a 16-bit WAV and a trip through 16 kHz (and after Opus on 96 of 96, all 16 bits); no false alarm on 304 unmarked clips (the share at most 0.170); the leading silence unmoved. **But the pitch moved over its bars**: the offset up to 0.884 Hz (bar 0.3), the range up to 8.19% (bar 1%); the rate 0.24%, the pauses 0.90%. **STOP by the rule**: nothing wired in | — | — |
| 6 | 10-20 | the run file read again, exploratory | the typical render barely moved (the offset 0.10 Hz, the range 0.53% at the median), the tails did: the offset over its bar on 20 of 96 renders, the range on 29. No direction: the range's signed change centred near zero, from -4% to +8%, worst on the lower voice (am_michael), and present at score 0, where the render barely edits the speech. The mark disturbs the pitch track; it does not make another patient | — | — |
| 7 | 10-20 | — | the author's choice of the three options: a weaker mark (section 3), its strength chosen on the dev set by a rule written first, then confirmed once on PR-009's held-out set; the gate unchanged | `watermark.embed` takes AudioSeal's `alpha`; `watermarkdev` `alpha` and `confirm`; tests | this commit |

## 1. The design

**What v1 §9 asked:** "Synthesized 'patient' audio watermarked and labelled synthetic in filenames and metadata."
The labels were there (`severity.write_synthetic`; the demo's WAV tags). A label goes when a file is copied or
converted; a watermark is in the sound itself, so whoever holds the clip can check that it was made here.

**The tool: AudioSeal** (Meta; code and weights MIT since 2024-04-02). A generator adds a mark shaped to the
speech's own envelope; a detector says, frame by frame, whether it is there, and reads back a 16-bit message. The
project's message is `0x5332` ("S2"). Since AudioSeal 0.2 nothing is resampled inside it: marks are made at
24 kHz, Kokoro's rate, and read at 24 kHz, a clip at any other rate resampled first (row 3).

**What is marked, scope B:** the severity renders (the laptop's `severity` runs and the demo's live render, the
neutral clip and the render each), and the replies R0 to R3 speak in the demo (Kokoro, whole clips). Not marked: R4's
streamed pieces and Moshi's calls (another model's voice, streamed); FT-3's FastPitch renders (a closed
registration, its own environment); and never a recording of a person.

**The development check** (`s2slab.watermarkdev`, the laptop, free): severity's dev set, its two voices, four
sentences and twelve scores, 8 neutral clips and 96 renders at 24 kHz, each marked as the demo will mark it; as
controls never marked, the same clips unmarked and the eval set's 200 real clips.

**The rule, written before the check** (`watermarkdev.gate`). **GO** to wiring the mark in if, on every clip:
- every marked clip reads as marked after a 16-bit WAV and after a trip through 16 kHz;
- no unmarked clip reads as marked (the renders, the neutral clips, the eval set's 200);
- severity's four measures move less than a tenth of PR-009's tolerances: the pitch offset under 0.3 Hz, the
  pitch range, the rate and the pauses under 1%;
- a reply's leading silence (the rungs' measure) moves under 20 ms, one of its frames.

Otherwise **STOP**: nothing wired in, and why recorded. The WebM/Opus round trip is reported, not a bar.

**Refined after the smoke test (row 3), before the check:** the design said "detected after resampling to
16 kHz". The detector reads at the rate the mark was made at, so a clip that went through 16 kHz is read back at
24 kHz. The bars did not change.

## 2. Step 1a, the full mark (rows 5 and 6)

`runs/watermark/dev.json`: 8 neutral clips and 96 renders of severity's dev set, the eval set's 200 as controls.

| check | result |
|---|---|
| found after a 16-bit WAV and a trip through 16 kHz | **104 of 104**; after WebM/Opus 96 of 96, all 16 bits |
| no false alarm | **0 of 304** (the share at most 0.170) |
| the render unchanged | **failed**: pitch offset 0.884 Hz at most (bar 0.3), range 8.19% (bar 1%); rate 0.24%, pauses 0.90% |
| a reply's leading silence | unmoved, 0 ms |

The mark sat 26.6 dB below the speech at the median (23.6 to 29.1). **STOP by the rule.** Read again (row 6),
the pitch moves are random, both ways, worst on the lower voice: the mark disturbs Praat's pitch track. The worst
moves are inside PR-009's own tolerances (3 Hz, 10%) but not inside the tenth of them the rule set, and the rule
stands.

## 3. Steps 1b and 1c: a weaker mark (row 7)

AudioSeal scales its mark by `alpha` (its own tune-down ratio). The full mark was found on every frame, every bit
read back, even after Opus, so there is room to weaken it. **The gate is step 1a's, unchanged.**

**Step 1b** (`python -m s2slab.watermarkdev alpha`): the gate at `alpha` 0.5, 0.3, 0.2 and 0.1 on the same dev
renders. **The rule, written before the run:** the strength CHOSEN is the strongest that passes in full (a stronger
mark is the harder to strip). If none passes: **STOP**, nothing wired in, and the plan's watermark recorded as tried.

**Step 1c** (`python -m s2slab.watermarkdev confirm`), once, refused until step 1b's file is committed: the chosen
strength on material the choice never saw, **PR-009's held-out voices (af_sarah, am_adam) and sentences (t1 to
t8)**, every score: 16 neutral clips and 192 renders, the eval set's 200 as controls again. **The rule, written
before the run:** the same gate. **GO**: the mark is wired in at that strength (step 2). Otherwise **STOP**, nothing
wired in. PR-009's result stands as reported; its renders are made again here for another question, the watermark's.
