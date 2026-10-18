# M38 run log: leading silence across TTS engines

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m37-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-18 | — | the design (below); the author's go, Piper in. Asked for by both plans: v1's P3 and v2's week 8, the last of the roadmap not started | — | — |
| 2 | 10-18 | the tests, laptop: 1056 passed, 2 skipped (17 new); a smoke render of three clauses through each engine, outside the repo | Kokoro and its ONNX copy 300 to 320 ms of leading silence, the same in both; Kokoro's own padding 325 to 400 ms, longer than the silence measured, so its first sound starts inside the padding; FastPitch (9017) 40 to 80 ms; Piper not yet installed | `leadsilence.py`, `ft/src/ft3/leadsilence_fp.py`, tests | this commit |

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
