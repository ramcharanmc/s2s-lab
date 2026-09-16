# Eval set v1 — 200 utterances in four cohorts

The fixed test set every rung is measured on (v2 plan §6.2). Built by
`python -m s2slab.evalset build`; frozen by `manifest.json`, which records where
every clip came from and the SHA-256 of every WAV. The audio is not in git:
rebuild it with the same command (pinned dataset revisions, seed 20260915) or
copy the WAVs, and check either with `python -m s2slab.evalset verify`.
`python -m s2slab.evalset summary` prints the counts, durations and download.

| Cohort | Clips | Source | License | What it is — and is not |
|---|---:|---|---|---|
| clean | 50 | [LibriSpeech](https://www.openslr.org/12) test-clean, via `openslr/librispeech_asr` | CC BY 4.0 | Read audiobook speech. The easy case. |
| accented | 50 | [EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc) test split | CC BY-SA 4.0 (dataset card); CC BY 4.0 (Edinburgh DataShare) | Spontaneous two-person conversation, 22 accent labels other than mainstream US English. Accent as EdAcc labels it. |
| disfluent | 50 | [DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech) test split | Apache-2.0 | **One speaker re-enacting** disfluent phone calls from Switchboard. Performed disfluency: not stuttering, not clinical, not many speakers. |
| slow | 50 | The clean 50, slowed to 0.75× with Praat PSOLA | as clean | **An elderly-speech proxy, and only a proxy.** It changes rate alone: no tremor, breathiness or age-related articulation. |

## How clips were chosen

- 16 kHz mono, 16-bit. 3–15 s long and at least 6 words; the slowed clips are
  therefore 4–20 s. At most 2 clips per speaker.
- **clean**: a seeded shuffle of test-clean.
- **accented**: no "Mainstream US English", no "Don't know", and no segment
  containing a `<tag>` such as `<OVERLAP>`. Drawn only from the 13 Parquet row
  groups that together cover all 22 accents — a download budget of about
  380 MB instead of 3.6 GB — and interleaved across accents.
- **disfluent**: only clips whose verbatim transcript (`transcript_a`) differs
  from the fluent one (`transcript_c`), meaning at least one filled pause,
  editing term, discourse marker or false start. The reference text for WER is
  `transcript_a`, every word spoken; `transcript_c` is kept as `text_fluent`.
- **slow**: the clean cohort, clip for clip, so clean vs slow is a paired
  comparison of speaking rate.

## Attribution

- LibriSpeech: *LibriSpeech: an ASR corpus based on public domain audio books*
  (ICASSP 2015). CC BY 4.0.
- EdAcc: *The Edinburgh International Accents of English Corpus: Towards the
  Democratization of English ASR* (arXiv:2303.18110). The EdAcc transcripts
  in `manifest.json` are shared under CC BY-SA 4.0.
- DisfluencySpeech: amaai-lab, `huggingface.co/datasets/amaai-lab/DisfluencySpeech`.
  Apache-2.0.