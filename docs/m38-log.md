# M38 run log: leading silence across TTS engines

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m37-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-18 | — | the design (below); the author's go, Piper in. Asked for by both plans: v1's P3 and v2's week 8, the last of the roadmap not started | — | — |
| 2 | 10-18 | the tests, laptop: 1056 passed, 2 skipped (17 new); a smoke render of three clauses through each engine, outside the repo | Kokoro and its ONNX copy 300 to 320 ms of leading silence, the same in both; Kokoro's own padding 325 to 400 ms, longer than the silence measured, so its first sound starts inside the padding; FastPitch (9017) 40 to 80 ms; Piper not yet installed | `leadsilence.py`, `ft/src/ft3/leadsilence_fp.py`, tests | ec3e4d7 |
| 3 | 10-18 | the author's runs, laptop, \$0: `leadsilence render`, `src.ft3.leadsilence_fp`, `leadsilence analyze` | 200 first clauses through five engines; R4 from its run. **Kokoro 320 ms (p50), 360 (p90), the same through ONNX; its own padding 350 ms.** Piper and FastPitch's base voice 0 ms; FT-3's 9017 30 ms, its padding 46 against the base's 23; R4 200 ms. A trim cuts 300 ms from Kokoro at the median and loses no first word (194 of 200 heard after it, 193 before). **GO by the rule** | — | 9a86f5b |
| 4 | 10-18 | the tests, laptop | step 2's design (section 3): `r1-gpu-trim`, R1 with its first piece's leading silence cut; a development run on the probe clips, the rule written first | `r1.py` (`_shape`, `FirstTrimmed`, `r1-trim`, `r1-gpu-trim`), `rungs`, `evalrun` (a rung's `turn_extra`), `r1trim.py`, `::trimdev`, tests | ee15442 |
| 5 | 10-18 | `modal run modal_app/r0_bench.py::trimdev`, the author, at ee15442 | one T4, 53 s in the function, about \$0.010. On the 30 probe clips: first sound 517 ms (p50) and 807 (p90) trimmed, against 807 and 1,108 as is on the same turns (0.641, 0.728); no turn stalled either way; the cut 298 ms at the median, every cut at least 56 dB below its piece's loudest; the second piece still at least 0.9 s ahead of need. **GO by the rule** | — | — |
| 6 | 10-18 | — | **PR-018 registered** before any eval-set turn of `r1-gpu-trim` | `pr018.py`, `::pr018`, `docs/preregistration.md`, tests | 39c5e1b |
| 7 | 10-18 | `modal run modal_app/r0_bench.py::pr018 --confirm PR-018`, the author, at 39c5e1b | one T4, 165 s in the function, \$0.028. **PR-018 supported**: on the 200 eval clips first sound 498 ms against R1's 797 on the same turns (p50, 0.625 [0.617, 0.636]), 751 against 1,029 (p90, 0.729 [0.697, 0.736]); no turn stalled; every cut at least 50 dB below its piece's loudest (the loudest −50.6) | — | — |
| 8 | 10-18 | — | the results recorded verbatim (`docs/preregistration.md`); M38 closed | records | 2030ba2 |
| 9 | 10-18 | — | the backlog checked against Notion pages 18 and 20: nine open items listed that only Notion had; the candidate "the silence between later clauses", which no plan or result asked for, withdrawn; the trim in the demo chosen next, R1 only (PR-018 tested R1 alone) | records | this commit |

## 1. The design

**Why.** Page 07 §1.1 defines leading silence as "dead air between the first audio byte and the first audio
*energy*", says engine spreads here are large, and pitches a study: measure it across engines, then find where it
comes from (padding conventions, vocoder priming, the duration predictor at the onset). Page 20 found Kokoro
starting every reply with about 320 ms of it, about 40% of what a caller waits for R1's first word.

**The text.** R1's first clause of each of R0's 200 eval-set replies (`runs/r0-gpu/evalset-modal-t4-pr002.json`,
split by R1's own chunker): the clause whose silence a caller waits through.

**The engines**, all on the laptop's CPU:

| engine | what it is |
|---|---|
| `kokoro` | Kokoro-82M through the kokoro package, voice `af_heart`: the rungs' TTS |
| `kokoro-onnx` | the same model, M26's fp32 ONNX file through ONNX Runtime, given the package's phoneme ids: another runtime |
| `piper` | Piper, VITS with espeak-ng's phonemes, the LJSpeech voice (`en_US-ljspeech-medium`) |
| `fastpitch-ljspeech` | NeMo FastPitch + HiFi-GAN, the LJSpeech base voice (PR-012's, its files pinned) |
| `fastpitch-9017` | FT-3's fine-tune of it to Hi-Fi TTS speaker 9017 (PR-012's) |
| `r4-omni` | R4's own speech, from its eval-set run: its own replies, not these clauses, the rungs' measure only |

**The measures**, per render:

- leading silence by the rungs' measure (20 ms frames, the first within 40 dB of the render's loudest), and by
  finer ones (5 ms frames, within 30 and within 50 dB), so that no result rests on one threshold;
- the engine's own duration for the padding before the first phoneme, where it gives one: Kokoro's duration
  predictor (`pred_dur`, its first entry), FastPitch's (the space it puts first). The ONNX copy, Piper and R4
  give none;
- the first sound's class, from Kokoro's phonemes for the clause: vowel, sonorant, voiced obstruent, voiceless
  stop, voiceless fricative. A stop's closure is speech that is silent, not padding;
- what a trim would cost: the render cut to 20 ms before its first frame within 50 dB, and Whisper small.en on
  it and on the uncut render: is the first word still heard, and the clause's word errors.

**Descriptive.** Nothing is set against a bar, so nothing is registered. Renders stay out of git
(`results/m38/`); every number goes to `runs/tts/leading-silence.json`.

**The rule for what follows, written before the run:** if, for `kokoro` (R1's engine), the trim cuts **100 ms or
more at the median** and the first word is heard after the trim on **no fewer than 2 below** the untrimmed renders
(of 200), M38's step 2 puts the trim into R1's TTS stage and registers **PR-018** before any eval run: does R1's
first sound come sooner, by a bar set from step 1, without more word errors (one T4 run, about \$0.30). Otherwise
M38 ends as a characterization.

**Piper, the author's download** (`data/` is ignored by git):

    ~/.venvs/s2slab/bin/pip install piper-tts
    mkdir -p data/piper && cd data/piper && ~/.venvs/s2slab/bin/python -m piper.download_voices en_US-ljspeech-medium

**The runs, the author's, \$0:**

    PYTHONPATH=src ~/.venvs/s2slab/bin/python -m s2slab.leadsilence render
    cd ft && PYTHONPATH=.:../src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.leadsilence_fp && cd ..
    PYTHONPATH=src ~/.venvs/s2slab/bin/python -m s2slab.leadsilence analyze

## 2. Leading silence, by engine (row 3)

`runs/tts/leading-silence.json`, R1's 200 first clauses (R4: its own replies, from its eval-set run).

| engine | leading silence, the rungs' measure, p50 / p90 | 5 ms frames, 30 dB | 50 dB | padding by its own durations | first word heard, as is / trimmed |
|---|---:|---:|---:|---:|---:|
| `kokoro` | **320 / 360 ms** | 325 / 375 | 320 / 366 | **350 / 425** | 193 / 194 |
| `kokoro-onnx` | 320 / 360 | 325 / 375 | 320 / 360 | — | 193 / 193 |
| `piper` | **0 / 0** | 15 / 25 | 0 / 0 | — | 189 / 189 |
| `fastpitch-ljspeech` | **0 / 0** | 10 / 20 | 0 / 5 | 23 / 71 | 199 / 199 |
| `fastpitch-9017` | 30 / 80 | 45 / 90 | 30 / 80 | 46 / 81 | 196 / 196 |
| `r4-omni` | 200 / 220 | — | — | — | — |

**Where Kokoro's silence comes from: its duration predictor.** The padding token it puts before the first phoneme
gets about 14 of the model's 25 ms frames (350 ms at the median), and the first sound rises about 30 ms before that
padding ends. The same through ONNX Runtime, so it is the model, not the runtime; the same whatever the clause's first
sound (300 to 340 ms for vowels, sonorants, obstruents, stops and fricatives alike), so it is not speech that happens
to be silent. Page 07 §1.1's "duration-predictor behaviour at utterance onset" is all of it here.

**The spread is as page 07 said, large:** Piper and FastPitch's base voice start sounding at once; FT-3's
fine-tune doubled its padding; R4 waits 200 ms.

**The trim:** cut to 20 ms before the first frame within 50 dB, Kokoro loses 300 ms at the median and no first
word. Whisper made one error more on 11 of its 200 trimmed clauses, none on the first word (28 in 1,828 words
against 18); the ONNX copy, the same audio, showed no change. Read as Whisper's sensitivity to the leading
silence, not as a clipped onset, but kept as a guardrail in what follows.

By the rule written before the run (section 1), **step 2 follows: the trim in R1, and PR-018.** \$0.

## 3. Step 2: the trim in R1 (row 4)

**The rung.** `r1-gpu-trim` (`s2slab.r1`): R1 as it is, but its reply's **first** piece cut to 20 ms before its
first frame within 50 dB (`leadsilence.trim`, step 1's trim). Later pieces are left alone: their silence is the pause
between clauses, and trimming it is another question. Each turn records what was cut (`trim_ms`) and the loudest
5 ms frame in the cut against the piece's loudest (`cut_peak_db`), so a cut that took sound shows. `r1-trim` is the
same on the laptop's CPU.

**The risk step 1 could not see: stalls.** R1 plays its pieces back to back while the next clause renders; a first
piece 300 ms shorter must be followed 300 ms sooner.

**Paired, as PR-017 was.** R1 as it is, on the very same turn, is computed from the turn's pieces: the first one
longer by the cut, the leading silence longer by the same (the cut is below 50 dB of the piece's loudest, so silent by
the rungs' 40 dB measure too). The same words: a trim changes when the first sound is heard, not what is said.

**The development run.** `modal run modal_app/r0_bench.py::trimdev`: one T4, the 30 probe clips, about \$0.05.

**The rule, written before the run** (`s2slab.r1trim`): **GO** to PR-018 if the trimmed first sound's p50 is at
most **0.85** of R1's on the same turns **and** the trimmed turns that stall are no more than **2** above R1's;
otherwise M38 stops here, nothing registered. PR-018's bars are set from this run.

## 4. The development run (row 5)

`runs/r1-gpu/trim-dev-t4.json`: `r1-gpu-trim` on the 30 probe clips, R1 as it is computed from the same turns.

| on the same turns | first sound p50 | p90 | stalled turns |
|---|---:|---:|---:|
| R1 as it is | 807 ms | 1,108 | 0 |
| **first piece trimmed** | **517 ms** | **807** | **0** |
| ratio | 0.641 | 0.728 | |

The stall risk turned out small: R1 renders its second clause while the first plays, and the first clause plays
long enough that, even 300 ms shorter, the second piece came at least 0.9 s before it was needed (2.7 s at the
median); 12 of the 30 replies were a single piece. Every cut was at least 56 dB below its piece's loudest: silence.

**Then:** PR-018 registered (row 6), its run the author's: `modal run modal_app/r0_bench.py::pr018 --confirm PR-018`,
one T4, about \$0.25.

## 5. PR-018's run (row 7) and what M38 found

`runs/r1-gpu-trim/pr018.json`: `r1-gpu-trim` on the 200 eval clips, R1 as it is computed from the same turns.

| on the same turns | first sound p50 | p90 | stalled turns |
|---|---:|---:|---:|
| R1 as it is | 797 ms | 1,029 | 0 |
| **first piece trimmed** | **498 ms** | **751** | **0** |
| ratio [95% interval] | 0.625 [0.617, 0.636] | 0.729 [0.697, 0.736] | |

**Supported**, as predicted: about 0.3 s sooner on every cohort. The verbatim output and the exploratory notes are
under PR-018 in `docs/preregistration.md`.

**What M38 found, in one paragraph.** Kokoro, the TTS in R0 to R3, puts about 320 ms of silence before every
clause, the padding its duration predictor gives the token before the first phoneme; Piper and FastPitch put
none, R4 200 ms. Cutting it from R1's first clause, and only its first, loses no first word on the laptop's check
and lets the caller hear R1 0.3 s sooner, 0.50 s instead of 0.80 at the median, with no stall. The silence before
R1's later clauses is the pause between them, left alone.

**Next:** the trim in the demo's R1, M39 (`docs/backlog.md`). R0 and R2 keep their silence: PR-018 tested R1 alone.

M38 cost \$0.038 on Modal (rows 5 and 7); step 1 ran on the laptop.
