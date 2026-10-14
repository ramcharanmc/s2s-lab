# Backlog

What is not done, what was chosen next, and what is owed. Updated at every milestone. The results
are in `docs/preregistration.md` and `runs/`; this file is only the list of what comes after them.

Last updated 2026-10-13, at commit `a12cec6`.

## Chosen next: none

Candidates: R4 streamed from the Thinker's first words (the Talker still waits for the whole text); a start rule
timed from the turn's own code rate (the 0.5 s hold waits for the second block); a cheaper L40S load (capture
the 6144-place step only when a turn needs it: the wake grew from about 160 s to 251 s); the repository made
public, the author's call.

## Done: M34, R4 streamed on a right-sized cache

M33's follow-up, the author's go: a new rung, `r4-gpu-stream-sized`, our captured loop on a 2048-place cache
when the turn leaves room, else 6144; its start rule chosen on the probe clips by a rule written first; then
PR-016, registered before any eval run; then the demo (`docs/m34-log.md`). Development (`deb2294`, \$0.505): at
0.5 s the sized stream's first sound 2,684 ms against the stream's 3,433, no stall against 4 of 30, the same
codes. **PR-016 supported** (`d65274d`, \$2.468): on the eval set the sized stream's first sound 2,622 ms against
3,333 (p50), 3,005 against 3,716 (p90), no stall, the same codes and words. Next, step 4: the new rung under
"R4 streamed" in the demo, a `modal serve` check and `modal deploy`, the author's. **Deployed 2026-10-13**
(`a12cec6`): warm, the visitor's browser played R4 streamed's first audio 4.77 s after Send (2.86 s on the server);
the L40S's wake is now 251 s. M34 cost about \$3.47 on Modal.

## Done: M33, where a Talker step's time goes

R4's Talker has 1,351 M parameters (a code reads 2.64 GB: 3.06 ms on an L40S at best). Captured as one CUDA
graph, its step took 10.26 ms a code on today's 6144-place cache and **6.62 on a 2048-place one**, the same
codes on 10 of 10 probe clips (`a4502d2`, \$0.309): attention over the cache's empty places was about half
the step. Under the 8 ms line written before the run, so a design note follows (`docs/m33-log.md`).

## Done: M32b, a Talker decode loop of our own, stopped by its rule

R4 streamed's first block is made 3.1 to 3.4 s into a turn on the L40S, and the stream at 0.86 of real time;
the design note took most of the Talker's 12.7 ms a code to be overhead around its layers. Our own decode
loop (`src/s2slab/talkerloop.py`) writes `generate`'s codes exactly, on a tiny Talker and on R4's own (10 of
10 clips). But on R4's Talker it wrote a code in 11.24 ms, and with its step captured as one CUDA graph in
10.15, against `generate`'s 11.77 (`250eeca`, \$0.165): over the 10 ms the stop rule set before the run, so
M32b stopped there and PR-016 was not registered. The premise was wrong: about 10 ms a code is work on the
GPU. A lead, not measured: the step attends over the static cache's 6144 places while a turn uses about
600 to 1,000 (`docs/m32-log.md`, section 7).

## Done: M32a, R4 streamed in the demo

The demo's L40S offers R4 and R4 streamed side by side, on one set of weights (two would not fit the
card): the same reply, heard sooner. Commit 1: the pair, PR-014 and PR-015 in the measured part, the
door at thirty wrong codes in five minutes, the day's cap at \$3. Commit 2: the stream's blocks played
in the browser as they arrive: on Modal a browser played R4 streamed's first audio 5.83 s after Send, its first
block made at 3.41 s. Commit 3: the page's container a full core, so measuring the visitor's voice no longer
holds the stream back (its first lines at 1.53 s, 4.17 before). Deployed 2026-10-12: a visitor hears R4 streamed
about 5.5 s after Send on a warm L40S, R4 decoded whole 5 s or more after it. About \$1.14 of Modal across four
sessions (`docs/m32-log.md`, `docs/cost-ledger.md`).

## Done: M31, R4 streams its reply

R4 (Qwen2.5-Omni speaks its own reply) waits 6,859 ms for its first audio on the L40S (M28), every
other rung under 700 ms: its Thinker, Talker and token2wav each run to the end before the next begins,
about 0.93 s of wait per second of reply. M31 adds `r4-gpu-stream`, the same model and settings with the
audio handed out as it is made, as R1 is to R0. Step 1 (`911fea3`, \$0.278): the Talker is 75% of R4's
time and runs at real time, 19.8 ms a code; token2wav costs about 890 ms a call whatever the length. So
streaming alone would stall. Step 1b, the author's choice: can they run faster with a static cache and
compiled steps? First run (`f7e8d99`, \$0.164): the DiT compiled runs 2 to 4 times faster; the Talker's
static cache failed on a mask `transformers` hands it, fixed. Second run (`567e7f2`, \$0.287): the
Talker compiled at 0.63 of real time, its codes another sample than eager's. Step 2 (`ceae735`, \$0.685):
`r4-gpu-fast` and `r4-gpu-stream` on the probe clips; the stream's first sound 3,284 ms against R4's
5,865 with playback held to 0.5 s, no stall. **PR-015 supported** (`e2a4b4e`, \$2.592): on the eval set the caller
hears R4's reply after 3.3 s instead of 6.8 (p50) and 3.7 instead of 11.8 (p90), no stall, the same words.
M31 cost \$4.006 on Modal (`docs/m31-log.md`).

## Done: M30, voice activity projection on real speech

PR-001's error analysis left timing, not words. VAP (Ekstedt & Skantze 2022) projects who speaks next
from the audio alone; it needs real speech, so M30 builds a real-audio endpointing test bed from the
frozen eval set. PR-014 is registered: does TV (end the turn when VAP projects the system to speak)
cut off fewer disfluent turns than T1b, without making clean speakers wait? Commit 1: the harness, the
registration, the DEV baselines. Commit 2: VAP in its own environment (causal: 0 frames read ahead); the
registered rule found no cut-off on DEV, with a silent partner or with the system's prompt (Deviation
1): VAP rated the speaker's own pauses above the end of the turn. **PR-014 reported: not run**; the eval
set never touched. Every registration is reported. Next for VAP, if wanted: real two-party agent
dialogue, or a fine-tune for it (`docs/m30-log.md`). \$0, the laptop.

## Done: M29, PR-001

The last registration open (registered 2026-09-11): does a check on whether the sentence is finished cut off
fewer disfluent speakers than T1's word list, without making clean speakers wait? The simulator's turns
stopped mid-sentence, so the confirmation uses whole-sentence turns, recorded as a deviation before any
check existed (`docs/m29-log.md`). Commit 1 (`8eaac58`): the turns, the sentences, the deviation. Commit 2:
T1b, T1's list or a small language model's judgement that the sentence is unfinished, its cut-off
(0.01) chosen on DEV by a written rule and frozen. The author's one run on seed 2000: **supported**,
disfluent cut-offs 68.8% to 51.2% with the clean wait unchanged. Every registration is now reported.
What is left, by the exploratory analysis: pauses where the words so far could already be a sentence,
and pauses longer than any hold. Timing, not words: the case for VAP (Parked, below). \$0, the laptop.

## M27, a live demo on Modal

Asked for by the author on 2026-09-27: a version of the ladder that can be shown on demand, for
the portfolio. Neither plan has one. v2's week 8 says "publish"; v1 argues against a live bot as
the centrepiece and scopes the client transport out. So the demo shows measured turns first and
live turns second. The design and what has run are in `docs/m27-log.md`.

| Phase | What | State |
|---|---|---|
| 1, commit 1 | the measured part and a live turn, on the laptop's CPU | done |
| 1, commit 2 | FT-2's verbatim recognizer as a rung and a tier; its measured panel | done, on a stand-in model: PR-011's model has not run in a rung yet |
| 1, commit 3 | the Modal app: a CPU container for the page and the door, a T4 for the rungs; the access code as a secret, the spend counter, the volume `ft2`; a rung that fails answered with a sentence | done |
| 1, first run | the author creates the secret and runs `modal serve`; the wake is measured; PR-011's model answers in a rung for the first time | done: one live turn through four rungs on a T4, the wake 72 s (`docs/m27-log.md`, entry 9) |
| 1, commit 4 | the live handler not `async`, after Modal's warning in the first run | done; on Modal two turns ran and the warning was gone (`docs/m27-log.md`, entry 15) |
| 1, commit 5 | wrong codes counted, five in ten minutes close the live part for ten; a code under ten characters refused | done; on Modal it refused a code that was too short (`docs/m27-log.md`, entry 13) |
| 1, commit 6 | the recorder shows that it is recording, the time and the limit, and plays the recording back; Send off while the live part is closed | done; used by the author with his microphone on Modal. The author: "recorder is clear now" |
| 1, commit 7 | prosody: a measured panel from PR-006's run; a live card of the visitor's pitch, range and words a second, beside the words the rungs received | done; on Modal the card filled from the author's voice, twice (`docs/m27-log.md`, entry 17) |
| 1, commit 8 | the T4's window 300 s, so a visitor's next turn finds it up | done; on Modal a second turn found the machines up (`docs/m27-log.md`, entry 21) |
| 2 and 3, commit 9 | R3 on an A10 and R4 on an L40S, live, each woken only when a visitor asks; one cap over three machines | done; on Modal both answered, the wakes 37 s and 51 s (`docs/m27-log.md`, entry 21) |
| 3, commit 10 | R5 on the visitor's recording: Moshi, on an A10 of its own, hears it as one side of a call | done; on Modal Moshi took a recording as a call, at 1.67 of real time, and did not answer it (`docs/m27-log.md`, entry 23) |
| 4, commit 11 | measured panels for the rest: barge-in (PR-003), FT-1, the two severity renders, quantization; the note on R3's card; a throwaway call when Moshi loads | done; on Modal the panels served, and Moshi ran at 0.68 of real time after its throwaway call (`docs/m27-log.md`, entry 31) |
| commit 12 | answers arrive as they are ready: one request a machine, the door asked once | done; on Modal four requests a turn, each back when its machine was done (`docs/m27-log.md`, entry 31) |
| commit 13 | a severity render that can be heard: a fixed sentence, a condition, one of the registered scores; neutral and rendered speech, targets against what was measured | done; on Modal three renders, the two seen with every target within its tolerance (`docs/m27-log.md`, entries 28 and 30). The author: "Render sounded well" |
| commit 14 | the MIT licence; the author's local path out of nine files | done |
| deployed | `modal deploy`: https://ramcharanmc--s2s-lab-demo-web.modal.run | done, by the author (`docs/m27-log.md`, entry 32) |
| pushed | a private repository on GitHub, `ramcharanmc/s2s-lab` | done, by the author (`docs/m27-log.md`, entry 33) |
| after it | a link in the portfolio's `index.html` (drafted in entry 32); the repository made public when the author chooses | the author's |
| left out | a conversation with R5, and barge-in while a reply plays: both need the microphone open while the machine speaks (`docs/m27-log.md`, entry 19) | can follow; not planned |

## Done since: M28, the ladder on one card

Asked for on 2026-09-29, for the author's presentation: every GPU rung on one card. The L40S, since
R4 does not fit the A10; R0 to R4 in one container, R5's two models beside. Three runs, \$3.88, no
failure (`docs/m28-log.md`). Found on the way: two containers of one card differ by up to a quarter;
the earlier cost column mixed mean and median.

## Owed

| What | Why |
|---|---|
| Making the repository public | pushed private on 2026-09-29: the author wants to choose first which of the experiments below to run. Making it public is one setting and changes no commit. The history holds his local path, and every commit his name and e-mail address (`docs/m27-log.md`, entries 29 and 33) |
| `probes/README.md`, "Recording notes" | microphone and room are still to fill in |
| A warm turn's time outside the cards | with every machine up, R3's, R4's and R5's requests took 4 to 7.3 s more than their cards' times, the T4's 0.3 s (`docs/m27-log.md`, entry 31). Seen once; not looked into |
| The severity render at the scores that carry a warning | the author heard depression at 10: "sounded well". Nobody has listened at depression from 20 or mania from 26 |
| One sentence said two ways, through the voice card | the card's comparison has been made between two different sentences only |
| A live turn with a reference, for the verbatim tier | in the demo's first live turn it wrote "uh" twice and "wanna" where three other recognizers wrote none; what was said is not known. The committed data says it adds no filler to read speech (`docs/m27-log.md`, entry 11). Still open: a recording whose words are known, a script read once plainly and once with fillers put in |

## Raised by PR-013, none started

| What | It would answer | Cost |
|---|---|---|
| fp16 on a GPU, and why it returns no audio on a CPU | whether Claim B failed on the file or on the CPU runtime | under \$0.50 on Modal, or \$0 if found on the laptop |
| A constant pitch correction for the 8-bit copy | whether it comes back inside the 3 Hz guardrail | \$0 |
| A listening check | whether a 4 Hz shift is heard | \$0 |
| `af_sarah` v10 | why the torch package and fp16 both sit more than 3 Hz from fp32 on it | \$0 |

## The rest of week 8

| What | Cost |
|---|---|
| Leading silence across TTS engines | \$0 to small |
| Cold-start economics on Modal | a few dollars |

## Parked

| What | From |
|---|---|
| T4, the voice-activity-projection turn-taking model | v2 §3.1 |
| A second FT-1, keeping dev checkpoints to see what the LoRA learns when | PR-010 |
| R3 with streamed output (R4's is M31) | R3 |
| A Talker decode loop of our own, its whole step compiled, so R4's stream starts from its first block without stalling | PR-015 |
| R4 streamed from the Thinker's first words, not after its whole text | PR-015 |
| The compiled Talker timed stage by stage on a slow and a fast host (1.6 times faster on one, no faster on the other) | M31 |
| The upward pitch bias in the Praat severity render | PR-009 |
| The base voice's sentence u8, whose pause vanishes in mania | PR-012 |
| Another disfluent speaker with verbatim transcripts | PR-011 |
| Wiring the barge-in policy and truncation into R1's live playback | PR-003 |
| A blind barge-in set, its script written by someone else | PR-003 |
| emotion2vec features for the tone tagger | PR-005 |
| MiniCPM-o as a third full-duplex model | set aside 2026-09-24 |
| A splitter that also cuts long sentences without commas | PR-002 |
| PR-001, registered and open | the turn-taking track |
| The Vast kit's fixes, untested on a machine | FT-1 |
