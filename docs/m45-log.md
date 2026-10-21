# M45 run log: the closure

The author's closure list, agreed 2026-10-20 (`docs/backlog.md`): first a check of the repository before it is made
public (M44, `docs/m44-log.md`); then a final check of the deployed demo; then the author's owed items; then the
plans' open items, each closed or written down as not done, with why. This log holds the last three. The earlier
logs: `docs/m44-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-20 | the deployed page read from a browser: `/api/status`, the Measured tab, its six replay clips; no code entered, no machine woken | open, \$3.00 of \$3.00 left, the access code set; every panel renders, every clip loads, no error in the console (section 1) | — | — |
| 2 | 10-21 | the deployed demo, the author, a laptop's microphone in a quiet room: Send A (every box ticked, every machine cold), Send B (the same sentence said another way, every machine warm), two severity renders | every machine answered both Sends and no card showed an error; R4 streamed never ran dry; Moshi answered neither call; the renders heard at two warning scores; \$0.543 by the app's count (sections 1 and 2) | — | — |
| 3 | 10-21 | — | the owed items closed or written down (section 3) | records | 99f6597 |
| 4 | 10-21 | — | the author's choice for the plans' open items: three done (P1c's latency, the flicker rate, the cascade at the plans' sizes), five written down as not done (section 4); P1c's latency rule written before its run (section 5) | `p1clatency.py`, tests | 6091653 |
| 5 | 10-21 | `python -m s2slab.p1clatency`, the author, laptop, free, at 6091653 | **NOT MET, as predicted**: the serial sum's p90 195.3 ms against the bar 30 ms; eGeMAPS alone 170.5 ms at p90 (section 6) | — | — |
| 6 | 10-21 | the run file read again, exploratory | eGeMAPS never took longer than this laptop's recognizer, so beside it the side-channel would add only the longer prompt: 23.2 ms at p50, 42.4 at p90, over the bar too; the tagger said "angry" on 116 of the 200 clips (section 6) | — | — |
| 7 | 10-21 | — | P1c's latency recorded | records | 028a3ed (its message names the flicker rate's definition by mistake; the definition is the next commit) |
| 8 | 10-21 | — | the flicker rate's definition, written before the count (section 7) | `flicker.py`, tests | this commit |

## 1. The deployed demo (rows 1 and 2)

**Row 1, nothing woken.** `/api/status`: open, the access code set, \$3.00 of the day's \$3.00 left; the T4 holds R0,
R1, R2 and R2 verbatim, and R3, R4, R4 streamed and R5 are offered, with R1 described as the trimmed rung. The
Measured tab: the eight ladder cards (R0 to R5, both R5 models), the verbatim, prosody and findings panels through
PR-018's; the six replay clips each 200, `audio/wav`; no error in the console. The Live tab: the four boxes with their
wakes, the recorder, the severity render.

**Row 2, two Sends.** Send A with every machine asleep, Send B a few minutes later with every machine up. The same
sentence both times, said plainly and then another way; the four T4 cards heard *"I've had a long week but I think
things are getting better"* in A, and *"I think I've had a long week, but I think things are getting better"* in B.

| Rung | A: first audio | first sound | whole reply | B: first audio | first sound | whole reply |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 590 ms | 910 ms | 590 ms | 632 ms | 932 ms | 632 ms |
| R1 (trimmed, 285 ms cut both times) | 383 ms | 403 ms | 669 ms | 339 ms | 359 ms | 589 ms |
| R2 | 427 ms | 727 ms | 1.16 s | 440 ms | 740 ms | 685 ms |
| R2 verbatim | 433 ms | 733 ms | 748 ms | 426 ms | 726 ms | 676 ms |
| R3 | 931 ms | 1.25 s | 931 ms | 891 ms | 1.21 s | 891 ms |
| R4 | 5.67 s | 5.89 s | 5.67 s | 6.28 s | 6.50 s | 6.28 s |
| R4 streamed | 2.16 s | 2.38 s | 5.10 s | 2.19 s | 2.41 s | 5.86 s |
| R5, Moshi | not answered: a call of 18.1 s at 0.70 of real time | | | not answered: 20.5 s at 0.73 | | |

- **Every machine answered both Sends, and no card showed an error.** R1's first sound came 20 ms after its first
  audio, the others' 300 to 320 ms after: PR-018's trim, as deployed in M39.
- **R4 streamed never ran dry** in the browser, either time. Its first audio reached the browser **108.0 s** after
  the cold Send (the L40S's wake in it) and **6.33 s** after the warm one, where the model's own first audio was at
  2.19 s: 4.1 s outside the card, a third sighting of the backlog's "browser's first audio" item (section 3).
- **Moshi answered neither call**; it opened both. In the eval run it answered 96 of 200.
- **Seen once, not looked into:** R3's warm reply repeated the caller's sentence back; R5's card named its GPU
  "NVIDIA A10" in Send A and "NVIDIA A10G" in Send B (the name is the host's).

The app's count for the day: \$0.543 (\$2.457 of \$3 left after the session, read from `/api/status`), the two
renders \$0.020 and \$0.028 of it.

## 2. One sentence said two ways, and the renders at the warning scores (row 2)

**The voice card**, Send B against Send A: median pitch 106 to 144 Hz (**+5.3 semitones**), range 10.6 to 8.7
semitones (**-1.9**), 2.3 to 1.8 words a second (**-0.5**). The first time the card has compared one sentence with
itself, near enough: the second take put "I think" in front.

**The renders**, `af_sarah`, sentence t4. Both scores are past the page's warning (about 30% of rate or 40 Hz of
pitch); mania was heard at 40, not the 26 asked for, which is past it too.

| | asked for | measured | in the registered run | |
|---|---:|---:|---:|---|
| depression, PHQ-9 20: pitch offset | -19.82 Hz | -19.10 | -18.30 | met |
| range | 0.56 | 0.57 | 0.54 | met |
| rate | 0.63 | 0.63 | 0.63 | met |
| pauses | 3.22 | 3.20 | 3.20 | met |
| mania, YMRS 40: pitch offset | +30.00 Hz | +38.25 | +36.44 | **missed** |
| range | 1.53 | 1.35 | 1.38 | **missed** |
| rate | 1.53 | 1.54 | 1.54 | met |
| pauses | 0.53 | 0.55 | 0.55 | met |

**Heard, the author:** depression at 20, *"slower and lower"*; mania at 40, *"high speaking rate... it was faster"*.
Mania's pitch missed as it did in PR-009's registered run, upward: the parked "upward pitch bias in the Praat
severity render" (PR-009), not a new fault. Not listened to: depression at 27, mania at 26 and 60.

## 3. The owed items (row 3)

| Owed | Now |
|---|---|
| One sentence said two ways, through the voice card | **done** (section 2) |
| The severity render at the scores that carry a warning | **done** at depression 20 and mania 40 (section 2); the other warning scores not heard |
| A warm turn's time outside the cards (R3's, R4's and R5's requests 4 to 7.3 s longer than their cards, `docs/m27-log.md`, entry 31) | **written down, folded into the backlog's candidate "the browser's first audio, later than a laptop client's"**. It was seen when each machine had a request of its own; since M37 one request carries every machine, so that measure no longer exists. What can still be seen is the browser's side, and row 2 adds one: 6.33 s after a warm Send against 2.19 at the model. Both want the same thing, a page that times each event as it arrives |
| A live turn with a reference, for the verbatim tier | **not done, the author's choice** (2026-10-21). What is known stays as it was: on read speech FT-2's recognizer adds no filler, on other speakers' spontaneous speech it writes fillers `small.en` drops (`docs/m27-log.md`, entry 11) |
| `probes/README.md`, "Recording notes" | **done**: the own-voice set (M17, 2026-09-22) was recorded on the laptop's built-in microphone in a quiet room, the author's answer, now in `probes/README.md` |
| Making the repository public | the author's, in GitHub's settings (`docs/m44-log.md`, section 3) |

## 4. The plans' open items (row 4)

The eight open rows of `docs/backlog.md`'s "From the plans, not done", each with what doing it would take. The
author's choice, 2026-10-21: **three done**, the rest **written down as not done**, with why.

| Item | Plan | Now | Why |
|---|---|---|---|
| P2's controllability and a blinded preference test, duplex against cascade | v1 §8 | **not done** | it needs listeners other than the author, since a test rated by its builder is not blind. **P2 stays undecided**: of its three conditions only the latency is measured (page 20 §20) |
| P2c's disentanglement ablation | v1 §4 | **not done** | it needs listeners, as above |
| R5's "five conversations per A100" | v1 §4, v2 §8 | **not done** | several Moshi calls at once on one card needs a batched duplex step that nothing here has: engineering, not a measurement of what exists |
| Full-Duplex-Bench's protocol for the duplex numbers | v1 §7 | **not done** | R5 was measured with this project's own harness (response gap, overlap, backchannels, interruptions in stream time, page 20 §20), said so wherever its numbers appear; adopting the benchmark's data and scoring is its own piece of work |
| "VAP is what Moshi learned implicitly" | v2 §3.1 | **not done** | its engineered side has no result: PR-014 was not run, no setting met its rule on development (M30); VAP on real two-party dialogue stays parked |
| P1c's latency criterion: the side-channel adds under 30 ms | v1 §8 | **to do**, first (section 5) | \$0, the laptop |
| The harness's partial flicker rate | v1 §2 | **to do**, second | \$0, the laptop: from the streamed partials of the fast tier (R0's and R1's recognizer, re-run on the growing audio for barge-in), saved in PR-003's committed runs; no rung streams partials otherwise. Entity-F1 and B-WER against U-WER **not done**: they need entity labels and a list of biased words the eval set does not have |
| The cascade at the plans' sizes: a large-v3-turbo recognizer (int8) and a 7 to 8B LLM (AWQ) | v1 §4, v2 §3 | **to do**, third | a few dollars of Modal: every cascade number rests on Qwen2.5-0.5B, which refused four of eight harmless questions in the demo (`docs/m27-log.md`, entry 31) |

## 5. P1c's latency criterion: the rule, written before the run

v1 §8, P1c: the affect condition beats the baseline **and** the shuffled-tag control, **and** "added latency
< 30 ms". PR-005 decided the first half (not supported) and recorded "no latency measured".

**What "added" covers**: what the side-channel puts in front of R1's reply that R1 does not do already, per turn.
(a) eGeMAPS, openSMILE's 88 functionals over the turn's audio (`affect.egemaps`); (b) the tagger (`Tagger.tag`);
(c) the longer prompt: R1's LLM (`r0.QwenLLM`, Qwen2.5-0.5B) given the transcript and " Their tone of voice was
<word>." (the tagger's word for that clip), one forward pass, less the same pass without the sentence; floored at 0.

**On**: the eval set's 200 clips, the laptop's CPU, its transcripts R0's from the same laptop
(`runs/r0/evalset-laptop-cpu.json`), so no recognizer runs. Warmed up once; each part timed three times a clip and
the median kept; (c)'s two passes alternate which goes first.

**The rule:** the serial sum (a) + (b) + (c), as if the side-channel ran after the recognizer. **MET** if its p90
over the 200 clips is under 30 ms; **NOT MET** otherwise. Its p50 and maximum beside it.
**Exploratory, outside the rule:** the parallel bound, max(0, (a) - the recognizer's time) + (b) + (c), since eGeMAPS
needs only the audio and could run beside the recognizer; the recognizer's time R0's on this laptop.

**The prediction, written first: NOT MET** on the serial sum. One eGeMAPS pass took about 0.1 s on this CPU
(`docs/backlog.md`), so (a) alone should be 50 to 150 ms.

**What it cannot show:** the laptop's CPU is not the T4 container's; (c) on a CPU overstates a GPU's prefill; a
streaming eGeMAPS, its features taken while the caller speaks, is not built and would change the answer.

## 6. P1c's latency: the result (rows 5 and 6)

`runs/prosody/p1c-latency.json`: the eval set's 200 clips, the laptop's CPU (16 threads), the median of three
timings a clip, at `6091653`.

| Part | p50 | p90 | max |
|---|---:|---:|---:|
| (a) eGeMAPS | 103.8 ms | 170.5 ms | 252.8 ms |
| (b) the tagger | 0.02 ms | 0.03 ms | 0.06 ms |
| (c) the longer prompt | 23.2 ms | 42.4 ms | 175.6 ms |
| **(a) + (b) + (c), the rule** | **125.3 ms** | **195.3 ms** | 319.7 ms |
| parallel bound, exploratory | 23.2 ms | 42.4 ms | 175.6 ms |

**NOT MET by the rule**: 195.3 ms at p90 against 30, as predicted. By cohort, the sum's p50: clean 128.6 ms,
accented 119.1, disfluent 122.0, slow 145.6 (the slow cohort's clips are the longest).

**Read again (row 6), exploratory:**
- **eGeMAPS is the cost**, about 16 ms a second of audio (the clips' median 6.3 s). It never took longer than this
  laptop's recognizer on the same clip (0 of 200), so run beside the recognizer it would add nothing, and the
  parallel bound is the longer prompt alone. That is over the bar too at p90 (42.4 ms), and at p50 under it
  (23.2 ms).
- **The longer prompt** adds about 4% to a 550 ms prefill on this CPU (572 against 550 at the median); the
  difference is noisy, 23 of 200 below zero and floored. A GPU's prefill is far shorter; not measured here.
- **Seen, outside the question:** the tagger's word was "angry" on 116 of the 200 clips, "neutral" on 77, "sad" on
  4, "fearful" on 3, in every cohort but the disfluent one mostly "angry". Fitted on CREMA-D's acted clips, it reads
  the eval set's ordinary read and spontaneous speech as angry more often than not: in a live cascade it would tell
  the LLM most callers sound angry. PR-005 tested it on CREMA-D pairs only, and its own-voice look (M17) on five
  pairs; this is the first look at its words on speech of another kind.

**P1c, both halves now decided:** the side-channel did not beat the shuffled-tag control on all three emotions
(PR-005, not supported) and it adds more than 30 ms (here, not met). A streaming eGeMAPS, taking its features while
the caller speaks, is the one change named that could bring (a) off the critical path; not built.

## 7. The partial flicker rate: the definition, written before the count

v1 §2 lists "flicker rate" among the harness's metrics, with no bar: a measurement, not a test.

**The data**, committed, nothing run: PR-003's two runs, `runs/bargein/pr003-measured.json` (the recognizer charged
what it took on the laptop, about 500 ms a call) and `runs/bargein/pr003-70ms.json` (charged a T4's 70 ms). Each saved
every streamed partial for the 20 barge-in items: when it was delivered, its words, whether settled, which sound
event it transcribes. The fast tier, R0's and R1's recognizer run again and again on the growing audio, is the only
thing here that streams partials.

**The definition** (`s2slab.flicker`): words as the barge-in policies read them (`bargein_policies.words_of`: lower
case, punctuation dropped). Within one sound event, each partial after the first is an **update**; an update is a
**revision** when the previous partial's words are not a prefix of its own. A **tail** revision changes only the
previous partial's last word; a **deep** one an earlier word.
- the flicker rate: revisions / updates, tail and deep apart;
- words withdrawn per word shown: the words taken back / the previous partials' words, over every update;
- a settled partial against the unsettled one before it: how often settling changed the words;
- per run, and per kind of item: should stop (stop-long, stop-short), should keep going (continue-short,
  continue-long).

**What it cannot show:** twenty short items in one voice, not the eval set; no rung's reply reads partials, so this
is the flicker the barge-in policy reads, nothing else.
