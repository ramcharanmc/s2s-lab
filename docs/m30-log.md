# M30 run log: voice activity projection on real speech

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m29-log.md` and those it names. The registration:
`docs/preregistration.md`, PR-014.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-03 | — | the design: VAP on real speech, the eval set as the test bed, a registration; the author's go | — | — |
| 2 | 10-03 | the VAP repository, read | MIT code; a state dict in the repo; a CPC checkpoint fetched on first use; stereo 16 kHz in; `p_now` and `p_future` every 20 ms | — | — |
| 3 | 10-03 | the eval clips' structure, and the VAD's ends on DEV, laptop | accented and disfluent clips end within 20 ms of the last word; on DEV the VAD's end meets the labels within 100 ms on 26 of 30, one 6.5 s off | — | — |
| 4 | 10-03 | T0, T1 and T1b on the 30 DEV recordings, padded, laptop | T1 cuts off 21 of 30, T1b 20, T0 at 900 ms 13; the added noise reads as silence on all 30 | `realturns.py`, tests; PR-014 registered | ac3a932 |
| 5 | 10-04 | the author installs VAP in `~/.venvs/vap` (torch 2.4.1), at `f39a78b`; one forward pass on the repo's example | the published state dict loads; the CPC weights are in the clone | — | — |
| 6 | 10-04 | VAP over the 30 DEV turns, a silent second channel; the causality check | VAP reads no audio ahead (0 frames in 90 cuts); its `probs()` fails on audio under 2 s, so `p_now` is computed as `probs()` does, without its loss: identical | `vaprun.py`, `vapturns.py`, `policies.ProjectionThreshold` | — |
| 7 | 10-04 | the registered rule on DEV | no θ: the median wait 1,435 to 1,500 ms at every θ; VAP rates the speaker's pauses (0.37) above the end of the turn (0.23) | — | — |
| 8 | 10-04 | Deviation 1, the author's choice: the system's prompt on VAP's second channel; DEV again | still no θ: median wait about 1,480 ms; after the end 0.29, pauses 0.37 | the prompt (pinned) | — |
| 9 | 10-04 | — | **PR-014 reported: not run**, no TV could be frozen; the eval set never touched | PR-014's Deviations and Results | this commit |

## 1. The design

PR-001's error analysis left timing, not words: T1b's remaining disfluent cut-offs are pauses where
the words so far could already be a sentence, or pauses longer than any hold. VAP (Ekstedt &
Skantze, Interspeech 2022) is trained on real two-party conversation to project, from the audio
alone, who speaks next. It cannot run on the simulator, whose turns are pulse trains, so M30 builds a
real-audio test bed. That also lifts the "synthetic only" caveat that every endpointing result has
carried. The author's choices: the frozen eval set as the test bed, its end of turn by the policies'
own VAD; a registration before any VAP number; VAP's one setting chosen on the 30 own-voice
recordings.

## 2. What VAP is, as published

`ErikEkstedt/VoiceActivityProjection`, MIT licence.
- **Model:** a state dict in the repository (`example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt`),
  a small transformer over CPC speech features. The CPC encoder's weights (Libri-light,
  `60k_epoch4-d0f474de.pt`) are fetched from Facebook's public file server on first use.
- **Input:** a stereo 16 kHz waveform; a mono file gets a silent second channel.
- **Output, every 20 ms:** a voice activity estimate per channel, and `p_now` and `p_future`: the
  probability that each channel speaks next, within about 0.6 s and from 0.6 to 2 s.
- **Training data:** Switchboard, Fisher and Candor, English telephone and video conversation.

## 3. The harness: real clips as turns

`realturns.py` builds the simulator's own `Utterance` from a real clip, so every policy and
`sweep.evaluate` run unchanged.
- **Ends.** The 30 DEV recordings keep their labelled ends. The eval clips have none, so their end
  is the end of the last voiced stretch of speech by the policies' VAD (as the DEV labels were
  drafted). On DEV that rule meets the labels within 100 ms on 26 of 30, and one recording is 6.5 s
  off, a noise after the turn; hence the labels on DEV. The eval clips are corpus segments cut at the
  last word (accented and disfluent clips end within 20 ms of it), so nothing after the speech can be
  taken for more of it.
- **Silence after the end.** Every turn is padded 2.5 s past its end, with noise at the clip's own
  floor. Not digital silence: the VAD's noise floor is the clip's quietest tenth of frames, and
  zeros would drag it down until the room noise inside the clip read as speech. Checked: on all 30
  DEV turns, nothing after the end reads as speech.
- **Words.** The eval clips' words come from faster-whisper base.en with word times, R0's recognizer
  on the CPU. It hears the whole clip, so its words are a little better than a streaming recognizer's.
- **The lock.** The eval turns are refused without PR-014's confirmation.

## 4. PR-014, registered before any VAP number

In `docs/preregistration.md`.
- **The question:** on real speech, does an endpointer that ends the turn when VAP projects the
  system to speak next (TV) cut off fewer disfluent turns than T1b, without making clean speakers
  wait longer?
- **TV:** ends the turn in a silence run of at least 100 ms when VAP's `p_now` for the silent channel
  reaches θ, else at 1,500 ms.
- **θ:** chosen on DEV by a written rule.
- **The criteria:**
  - S1: fewer disfluent cut-offs, the paired CI above 0;
  - S2: Holm p < 0.05 over the four cohorts;
  - S3: clean wait at most +100 ms median and +300 ms p90;
  - S4: at most 2 more clean cut-offs.
- **Known limits:** 50 turns a cohort; the truth is the VAD's; VAP's training domain is telephone
  conversation.

## 5. The DEV baselines, before any VAP number

`runs/turn-taking/pr014-dev-baselines.json`, on the 30 own-voice recordings, labelled ends, padded:

| policy | cut off | median wait ms | p90 ms |
|---|---:|---:|---:|
| T0_fixed_500ms | 26 of 30 | 472 | 494 |
| T0_fixed_700ms | 17 of 30 | 690 | 700 |
| T0_fixed_900ms | 13 of 30 | 900 | 900 |
| T1_semantic_500ms | 21 of 30 | 490 | 500 |
| T1b_complete_500ms | 20 of 30 | 490 | 501 |

These recordings were made with long, deliberate pauses (function-word, content-word and natural
dictation), so even T0 at 900 ms cuts off 13. T1b barely improves on T1 here, 20 against 21, with
the words exact (the labels'). It is the set on which VAP's cut-off will be chosen.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:
the full suite, 908 passed, 2 skipped (with the DEV recordings and eval audio copied in), and the DEV baselines above.

## 6. Next

Commit 2: VAP in an environment of its own, at a pinned commit; the causality test; θ on DEV by
the rule; TV frozen, with the confirmation's exact command. The author's steps, before it:

```bash
git clone https://github.com/ErikEkstedt/VoiceActivityProjection ~/vap && git -C ~/vap log --oneline -1
python3 -m venv ~/.venvs/vap && ~/.venvs/vap/bin/pip install -e ~/vap
```

The CPC weights are fetched by the first run. Exact versions are pinned once the install is seen.

## 7. VAP installed, and run (at ac3a932)

The author's install: VAP cloned at `f39a78b`, its own venv, torch 2.4.1 CPU (VAP's loader calls
`torch.load` on a checkpoint holding objects, which torch 2.6 and later refuse by default), einops,
numpy 1.26.4, soundfile. The published state dict loaded as it is; the CPC weights were already in the
clone, so nothing more was fetched.

The two environments meet through files. `vapturns export` writes each turn exactly as the policies
see it (float32 WAVs, their SHA-256, words and ends); `vaprun.py`, standalone in VAP's environment,
writes VAP's `p_now` for the second channel per 20 ms frame; `vapturns` reads it back into each turn's
context, where TV (`policies.ProjectionThreshold`) reads it.

**A fault in VAP's code, found on DEV:** at `f39a78b`, `VapGPT.probs()` overwrites its `vad` argument
with the model's own estimate and then always computes a training loss, which fails on audio shorter
than its 2 s projection window. `vaprun.p_system` computes `p_now` as `probs()` does, without the loss;
on 8 full-length DEV turns the two agree exactly.

**V3, causality:** each DEV turn run whole and cut short at a quarter, a half and three quarters: 90
cuts, 0 frames differing before any cut. VAP reads no audio after the decision frame.

## 8. The rule on DEV, as registered: no θ

| θ | cut off | median wait ms | ended by the fallback |
|---:|---:|---:|---:|
| 0.2 | 27 of 30 | 1500 | 2 |
| 0.3 | 22 of 30 | 1435 | 5 |
| 0.4 | 16 of 30 | 1490 | 13 |
| 0.5 | 10 of 30 | 1490 | 21 |
| 0.6 | 9 of 30 | 1490 | 23 |
| 0.7 | 8 of 30 | 1490 | 25 |
| 0.8 | 6 of 30 | 1490 | 28 |

At every θ the median wait is 1,435 to 1,500 ms: most correctly ended turns are ended by the 1,500 ms
fallback, not by VAP. Inside the speaker's own pauses, VAP's highest probability that the system speaks
next has a median of 0.37; within 1.5 s after the true end, 0.23, reaching 0.5 on 5 of 30 turns. With
a silent second channel, VAP rates the speaker's pauses above the end of the turn.

## 9. Deviation 1, the system speaks first: still no θ

The author's choice (B of three). VAP is trained on two parties, and a voice agent always speaks
before the caller's turn. Each DEV turn was given the system's prompt on VAP's second channel ("How can
I help you today?", Kokoro `af_heart`, 1.82 s, then 0.3 s of silence; the file pinned by SHA-256
`e78e6b589f44f9ee...`), with the caller's own floor noise on the first channel until the turn begins. VAP's
107 frames before the turn are dropped, so TV reads the same frames as before. Causality with the
prompt: 0 frames in 90 cuts.

| θ | cut off | median wait ms | ended by the fallback |
|---:|---:|---:|---:|
| 0.2 | 16 of 30 | 1480 | 8 |
| 0.3 | 15 of 30 | 1480 | 11 |
| 0.4 | 11 of 30 | 1480 | 17 |
| 0.5 | 11 of 30 | 1480 | 18 |
| 0.6 | 9 of 30 | 1490 | 21 |
| 0.7 | 7 of 30 | 1490 | 24 |
| 0.8 | 6 of 30 | 1494 | 26 |

The prompt cut the false cut-offs at low θ (at 0.2: 27 to 16 of 30; from 0.5 up, within one), and VAP's probability after the end rose
(median 0.29, reaching 0.5 on 9 of 30), but still below the speaker's own pauses (0.37). The median
wait is 1,480 to 1,494 ms at every θ. No θ.

## 10. PR-014: not run

As agreed before Deviation 1: if no θ qualified with the prompt either, PR-014 is reported as not run,
with both findings. No TV could be frozen; the confirmation refuses without one, and the 200 eval
turns were never built. PR-014's Results say what it shows (VAP as published, deciding from one
speaker's audio, does not mark the end of these turns in time, with or without a spoken prompt) and
what it does not (dialogue; anything on the eval set). Every registration is reported: fourteen.

The code stays: the real-audio harness, TV, the two-environment bridge and the causality check are
what a next attempt would use (VAP on real two-party agent dialogue, or VAP fine-tuned for it).

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 921 passed, 2 skipped; the export refuses the eval turns without a frozen θ.
