# M45 run log: the closure

The author's closure list, agreed 2026-10-20 (`docs/backlog.md`): first a check of the repository before it is made
public (M44, `docs/m44-log.md`); then a final check of the deployed demo; then the author's owed items; then the
plans' open items, each closed or written down as not done, with why. This log holds the last three. The earlier
logs: `docs/m44-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-20 | the deployed page read from a browser: `/api/status`, the Measured tab, its six replay clips; no code entered, no machine woken | open, \$3.00 of \$3.00 left, the access code set; every panel renders, every clip loads, no error in the console (section 1) | — | — |
| 2 | 10-21 | the deployed demo, the author, a laptop's microphone in a quiet room: Send A (every box ticked, every machine cold), Send B (the same sentence said another way, every machine warm), two severity renders | every machine answered both Sends and no card showed an error; R4 streamed never ran dry; Moshi answered neither call; the renders heard at two warning scores; \$0.543 by the app's count (sections 1 and 2) | — | — |
| 3 | 10-21 | — | the owed items closed or written down (section 3) | records | this commit |

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

## 4. The plans' open items

To come: each of `docs/backlog.md`'s "From the plans, not done" closed or written down as not done, with why.
