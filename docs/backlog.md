# Backlog

What is not done, what was chosen next, and what is owed. Updated at every milestone. The results
are in `docs/preregistration.md` and `runs/`; this file is only the list of what comes after them.

Last updated 2026-09-28, at commit `ab45907`.

## Chosen next: M27, a live demo on Modal

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
| 3, commit 10 | R5 on the visitor's recording: Moshi, on an A10 of its own, hears it as one side of a call | done on the laptop, with a made-up model; Moshi itself has not run in the demo |
| 4, commit 11 | measured panels for the rest: barge-in (PR-003), FT-1, severity, quantization | next |
| commit 12 | `modal deploy`, the URL that stays, and what is owed before a public link | the author's call |
| left out | a conversation with R5, and barge-in while a reply plays: both need the microphone open while the machine speaks (`docs/m27-log.md`, entry 19) | can follow; not planned |

## Owed

| What | Why |
|---|---|
| A licence file for the code | the root README says there is none |
| A remote | the repo has never been pushed |
| Local paths in committed files | six report files from M16 to M20 under `runs/` hold Modal's "Created mount" lines with the author's local path, and three comments in `src/s2slab/` (`policies.py`, `prosody.py`, `simulate.py`) hold a local command; every Modal report holds the workspace's URL. Checked for only from M25 on. Removing them from files is a commit; removing them from history is a rewrite, and evidence files are never overwritten: a decision for the author before any push |
| `probes/README.md`, "Recording notes" | microphone and room are still to fill in |
| R3's replies may describe the speaker, and name a feeling | in the fifth session R3 said of the author "The speaker is expressing frustration" (`docs/m27-log.md`, entry 21). The author's decision: whether R3's card should say so |
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
| A streaming Talker, or R3 with streamed output | R3, R4 |
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
