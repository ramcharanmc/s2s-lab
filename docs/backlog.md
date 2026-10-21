# Backlog

What is not done, what was chosen next, and what is owed. Updated at every milestone. The results
are in `docs/preregistration.md` and `runs/`; this file is only the list of what comes after them.

Last updated 2026-10-21, at commit `f90d5e2`.

## Chosen next: the author's closure list

Agreed 2026-10-20, in this order: ~~a check of the repository before it is made public~~ (M44, done); then ~~a final
check of the deployed demo, one Send to each machine and the Measured tab~~ (M45, done); then ~~the author's own owed
items~~ (below, "Owed"; M45, each done or written down); then the plans' open items each closed or written down as
not done, with why (M45, next).

Candidates:

- **Hand the L40S the browser's recording as it came** (55 kB of WebM for 14 s, not 0.9 MB of 32-bit floats):
  the handoff took 1.9 s on a 14 s clip (`docs/m37-log.md`, entry 14). Measured before it is changed.
- **The browser's first audio, later than a laptop client's** by 1.5 to 4.5 s on two turns, unexplained: a page
  that times when the response began and when each event came would say where. A third sighting in M45: a warm
  Send's first audio 6.33 s after Send in the browser, the model's at 2.19 s (`docs/m45-log.md`). The warm turn's
  4 to 7 s outside the cards, once owed, is this item since M37's one request per Send.
- **R4's own compile from the compile cache** (about 30 s of the L40S's wake), left alone in M36: loading the
  cache first once made R4's load 288 s.
- The repository made public, the author's call.

## In progress: M45, the closure

Items 2 to 4 of the closure list (`docs/m45-log.md`). **Done:** the deployed demo checked on 2026-10-21, every machine
answering two Sends, cold and warm, no card in error, R4 streamed never dry, \$0.543; one sentence said two ways
through the voice card; the severity render heard at depression 20 and mania 40, mania's pitch overshooting as in
PR-009's run. The owed items each done or written down (below). **The plans' open items:** the author chose
three to do (P1c's latency, the flicker rate, the cascade at the plans' sizes) and five written down as not done
(below, "From the plans, not done"). **P1c's latency: not met** (195 ms at p90 against 30; eGeMAPS the
cost). **The flicker rate:** about 3 in 10 of the fast tier's updates take back words already shown.
**Next:** the cascade at the plans' sizes, its rule written first (fp16, not
AWQ).

## Done: M44, the check before the repository is made public

Read-only scans of every tracked file and the whole history: no secret, no audio or weights, no large file, no local
path or e-mail in any file now. The author's name and e-mail are on all 201 commits, and 22 lines of old commits hold
the local path: the author chose to publish with the history as it is, since a rewrite would break every cited hash.
Every model's and dataset's licence checked and put in one table in the README (`docs/m44-log.md`).

## Done: M43, the watermark, tried and not adopted

v1 §9's watermark on the synthetic voices, with AudioSeal (MIT), against a rule written before each check
(`b45bd26`, `dbeb681`, the laptop, \$0). The full mark was found on every clip and on none unmarked, and survived a
16-bit WAV, 16 kHz and Opus; but it moved the severity render's measured pitch past the bar (the range up to 8.19%,
the bar 1%): STOP. Weaker marks were found only at half strength and above, where the pitch still moved: STOP. The
renders keep their synthetic labels (`docs/m43-log.md`).

## Done: M42, CI and the severity badge

v1's "severity monotonicity passing" badge and the suite in CI, on GitHub Actions (`20aa981`): the suite in 2 min 46 s
on a fresh machine, the severity test alone in 30 s with `requirements/base.txt`, both green on their first push.
`requirements/test.txt` writes down the five packages the laptop's environment had and no requirements file named;
two tests skip without their audio, as eight others did (`docs/m42-log.md`).

## Done: M41, the cold-start write-up

v1's and v2's "cold-start economics", from the runs already made (`docs/cold-start.md`, \$0): a cold turn costs
about \$0.06 on the T4 and \$0.10 to 0.11 on the L40S, mostly its idle window; the L40S's wake is mostly compiling;
a warm T4 would cost what about 250 cold turns do. What is not measured is listed there (`docs/m41-log.md`).

## Done: M40, P1b on real disfluent speech

v1 §8's P1b on DisfluencySpeech, one speaker performing disfluent phone calls, not on SEP-28k, whose 3-second clips
have no turn ends (`28657de`). DEV, the validation split's 215 turns: GO (`39e4d7c` records it). **PR-019 not
supported** (`39e4d7c`, the laptop): on the test split's 207 turns T2 at 900 ms cut off one, T0 at 900 ms none, the
same median wait; its one cut-off a falling pitch at a sentence's end inside the turn. \$0 (`docs/m40-log.md`).

## Done: M39, the trim in the demo's R1

The demo's R1 is `r1-gpu-trim`, PR-018's rung (`3c54204`); R0, R2 and R3 keep Kokoro's silence, since PR-018 tested R1
alone. Every turn card now shows first sound beside first audio, R1's what was cut; R1's measured row and a PR-018 panel
come from the registered run. One live turn on Modal: R1's first sound 20 ms after its first audio, R0's 320. Deployed
2026-10-18. About \$0.06 (`docs/m39-log.md`).

## Done: M38, leading silence across TTS engines

Kokoro, the TTS in R0 to R3, puts 320 ms (p50) of silence before every clause: its duration predictor's
padding before the first phoneme (350 ms). Piper and FastPitch put none, R4 200 ms; a trim to 20 ms before the
first sound lost no first word on the laptop (step 1, `ec3e4d7`, \$0). In R1, the first clause trimmed
(`ee15442`; development \$0.010, GO): **PR-018 supported** (`39c5e1b`, \$0.028): on the eval set first sound
498 ms against R1's 797 on the same turns (p50), 751 against 1,029 (p90), no stall. About \$0.04
(`docs/m38-log.md`).

## Done: M37, a start rule timed from the turn

R4 streamed's playback starts when the turn's own code rate says the second block is due, handed out by a timer
(`fe65b94`). The margin, 50 ms, by a rule written before the development run (\$0.224, GO). **PR-017 supported**
(`eca9da1`, \$0.800): on the eval set first sound 2,269 ms against the 0.5 s hold's 2,681 on the same turns (p50),
2,695 against 3,108 (p90), no stall. In the demo (`0d9b0d9`), the page's new count of gaps found the stream
reaching the browser in lumps; fixed by one request per Send (`d2af2d1`) and by declaring the streamed responses
unencoded, against zstd at Modal's front door (`d70c21a`). Deployed 2026-10-18: playback never ran dry. About
\$2.27 (`docs/m37-log.md`).

**Set aside, 2026-10-15: R4 streamed from the Thinker's first words.** On one GPU the time to the second block,
which a stall-free start must wait for, is the Thinker's whole reply plus 120 Talker steps, both bound by
reading weights: interleaving them changes their order, not their sum (`docs/m36-log.md`, section 1). It would
pay only with the two models on two GPUs.

## Done: M36, a compile cache for the L40S

What the L40S compiles kept on a Modal volume, `s2s-lab-compile-cache` (`f7e5bef`), loaded after R4 and before
R4 streamed (`6e66acc`; loaded first, R4's load once took 288 s, not explained). Two cold wakes: the L40S loaded
in 93.1 s and 71.8 s, against 185.7 in M35; deployed 2026-10-15. R4's own compile, about 30 s, is not cached.
About \$0.75 (`docs/m36-log.md`).

## Done: M35, a cheaper L40S wake

M34 had lengthened the L40S's wake to 251 s. Only the 2048-place cache is now prepared at load (a demo turn,
15 s at most, never takes the 6144-place one, whose step is captured on the first turn that needs it), and
the load is timed by phase (`d61c79d`). On Modal the L40S loaded in 185.7 s; deployed 2026-10-14, its wake
194.1 s. About \$0.24 (`docs/m35-log.md`).

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
| Making the repository public | **checked in M44** (`docs/m44-log.md`): nothing in the files to remove; the author's e-mail is on every commit and stays, by the author's choice. Left: the private e-mail setting and the switch to public, both in GitHub's settings, the author's |
| ~~`probes/README.md`, "Recording notes"~~ | **done in M45**: the own-voice set's, the laptop's built-in microphone in a quiet room |
| ~~A warm turn's time outside the cards~~ | **written down in M45**: measured when each machine had its own request, a path gone since M37; folded into the candidate "the browser's first audio" (above), which M45 saw once more |
| ~~The severity render at the scores that carry a warning~~ | **done in M45**: heard at depression 20 ("slower and lower", every target met) and mania 40 ("high speaking rate... faster"; pitch over its target, as in PR-009's run). Not heard: depression 27, mania 26 and 60 |
| ~~One sentence said two ways, through the voice card~~ | **done in M45**: +5.3 semitones, the range -1.9, 0.5 fewer words a second; the second take had "I think" in front |
| ~~A live turn with a reference, for the verbatim tier~~ | **not done, the author's choice** (2026-10-21, M45). Known: it adds no filler to read speech, and writes fillers `small.en` drops on other speakers' spontaneous speech (`docs/m27-log.md`, entry 11) |

## From the plans, not done

What pages 16 (v1 Research) and 17 (v2 Engineering) asked for and has not been done, checked against both on
2026-10-18. Week 8's leading silence and cold-start economics are in their own section below; publishing is under
Owed.

| What | Plan | State | Cost |
|---|---|---|---|
| ~~P1b on real disfluent speech~~ | v1 §8 | **done in M40: PR-019 not supported**, on DisfluencySpeech (one speaker, performed; not SEP-28k, whose 3-second clips have no turn ends): T2 cut off one turn the fixed threshold kept, at the same median wait. Real stuttering is still untested | \$0 |
| ~~P2's controllability and a blinded preference test, duplex against cascade~~ | v1 §8 | **written down in M45 as not done**: it needs listeners, so P2 stays undecided (only its latency part is measured) | listeners' time |
| ~~P2c in CI: the "severity monotonicity passing" badge~~ | v1 §4, §10 | **done in M42**: the severity test in CI in 30 s, its badge in the README; the whole suite in CI too | free |
| ~~P2c's disentanglement ablation~~ | v1 §4 | **written down in M45 as not done**: it needs listeners | listeners' time |
| ~~R5's "five conversations per A100"~~ | v1 §4, v2 §8 | **written down in M45 as not done**: it needs a batched duplex step nothing here has | a few dollars |
| ~~Watermarking the synthetic voices~~ | v1 §9 | **tried in M43, not adopted**: AudioSeal's mark, at any strength it is found at, moved the severity render's measured pitch past the bar written first; the renders keep their synthetic labels | \$0 |
| ~~Full-Duplex-Bench's protocol for the duplex numbers~~ | v1 §7 | **written down in M45 as not done**: R5 was measured with this harness, not theirs, and says so | small |
| ~~"VAP is what Moshi learned implicitly"~~ | v2 §3.1 | **written down in M45 as not done**: VAP built (M30), PR-014 not run: the engineered side of the comparison has no result | — |
| The cascade at the plans' sizes: a large-v3-turbo recognizer (int8, L4) and a 7 to 8B LLM (AWQ, A10) | v1 §4 (P1 W1), v2 §3 | **M45, to do third**, its rule written (`docs/m45-log.md`, section 9); built with base.en and Qwen2.5-0.5B on one T4; 1.5B and small.en tried on Colab only. Every cascade number is for the small model, which refused four of eight harmless questions in the demo (`docs/m27-log.md`, entry 31) | a few dollars |
| ~~The harness's entity-F1, B-WER against U-WER, and partial flicker rate~~ | v1 §2 | **M45: the flicker rate counted**, about 3 in 10 of the fast tier's updates take back words already shown (`docs/m45-log.md`, section 8); **entity-F1 and B-WER against U-WER written down as not done** (no entity labels, no biased-word list); before M45 none built; WER raw and normalized, per-stage latency, first audio and first sound, leading silence, RTF and the overlap ratio are | \$0 |
| ~~P1c's latency criterion: the side-channel adds under 30 ms~~ | v1 §8 | **done in M45: NOT MET**, 195.3 ms at p90 on the laptop's CPU, eGeMAPS 170.5 of it; beside the recognizer, the longer prompt alone, 42.4 ms (`docs/m45-log.md`, section 6) | \$0 |

Recorded, nothing to do: v2's week 1 expected R1 to be "a 4× win" over R0; it was 26% sooner (PR-002, not
supported). v1's P3 named TensorRT and F0 RMSE; PR-013 ran ONNX on a CPU and measured the median's shift. v1 §7's
data: CANDOR gave way to DailyTalk (FT-1's reference, page 20 §22) and Common Voice to EdAcc (the accented cohort,
page 20 §5); MentalChat16K, listed there, was never needed and not used. v2 §6.3's prosody probe came from CREMA-D
pairs and an own-voice mini set rather than 30 recorded takes (page 20 §13, §16). Checked against both plans in full
on 2026-10-18, after M39.

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
| ~~Leading silence across TTS engines~~: M38 step 1 (Kokoro 320 ms, Piper and FastPitch 0, R4 200; Kokoro's is its duration predictor's padding); step 2: the trim in R1, **PR-018 supported** (0.50 s heard instead of 0.80) | \$0.04 |
| ~~Cold-start economics on Modal~~: **written up in M41**, `docs/cold-start.md`: each machine's wake and what changed it, a cold turn's cost (mostly the idle window), cold against warm | \$0 |

## Parked

| What | From |
|---|---|
| ~~T4, the voice-activity-projection turn-taking model~~: built in M30 (VAP on real speech); PR-014 not run, no setting met its rule on development. Left: VAP on real two-party dialogue, or fine-tuned for it | v2 §3.1 |
| A second FT-1, keeping dev checkpoints to see what the LoRA learns when | PR-010 |
| R3 with streamed output (R4's is M31) | R3 |
| ~~A Talker decode loop of our own, its whole step compiled~~: built in M32b (exact, stopped by its rule); the gain came from the cache's size (M33, M34, PR-016) | PR-015 |
| ~~R4 streamed from the Thinker's first words~~: analysed and set aside, 2026-10-15 (above) | PR-015 |
| The compiled Talker timed stage by stage on a slow and a fast host (1.6 times faster on one, no faster on the other); M33 profiled its step on the slow host only | M31 |
| The upward pitch bias in the Praat severity render | PR-009 |
| The base voice's sentence u8, whose pause vanishes in mania | PR-012 |
| Another disfluent speaker with verbatim transcripts | PR-011 |
| Wiring the barge-in policy and truncation into R1's live playback | PR-003 |
| A blind barge-in set, its script written by someone else | PR-003 |
| emotion2vec features for the tone tagger | PR-005 |
| MiniCPM-o as a third full-duplex model | set aside 2026-09-24 |
| A splitter that also cuts long sentences without commas | PR-002 |
| ~~PR-001, registered and open~~: reported in M29, supported on whole-sentence turns | the turn-taking track |
| The Vast kit's fixes, untested on a machine | FT-1 |

**Listed here 2026-10-18, after a check against Notion.** These were open on Notion pages 18 and 20 and missing from
this file. Nothing is new: each comes from the result named.

| What | From |
|---|---|
| Cap the reply length: R0's p90 is set by its reply's length (r = 0.97), and R1's first audio still is (r = 0.46) | PR-002 (page 20 §9) |
| Accent apart from speaking style: the accented cohort is spontaneous speech (EdAcc), the clean one read (LibriSpeech); a third comparison, accented read speech or conversational US speech, would separate them | R0 on the eval set (page 20 §5) |
| More speakers, or a corpus newer than the model, for the contamination question PR-006 left | PR-006, the own-voice set (page 20 §16) |
| A sound shortly after an acknowledgement taking the acknowledgement's decision (fixes `cl02`, costs `sl05`) | PR-003 (page 20 §11) |
| Silero in place of the energy VAD, and the sweep diffed: how much of the failure is the detector | the turn-taking track (page 18 §14, C) |
| The clean cohort's tail cost: are T1's slow clean turns the ones that end on a function word? | the turn-taking track (page 18 §5, §14 A3) |
| The frontier's stability across more unseen seeds | the turn-taking track (page 18 §14 A4) |
| A development and evaluation split for T2's `slope_threshold`, tuned on the data it was evaluated on | the turn-taking track (page 18 §9, §14 A5) |
| A completeness check that knows dictation (digit groups, spelled letters, lists), registered first and tested on fresh recordings | the endpointing probe set (page 18 §8, §14 H) |
