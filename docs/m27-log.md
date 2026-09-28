# M27 run log: a live demo on Modal

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. Times are CDT. The earlier logs: `docs/ft1-log.md`, `docs/ft2-log.md`,
`docs/ft3-log.md`, `docs/m26-log.md`.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 09-27, evening | — | the design, five decisions, four phases | — | — |
| 2 | 09-27, evening | — | the root README and the backlog, owed since the first milestone | `README.md`, `docs/backlog.md` | 6037535 |
| 3 | 09-27, evening | the pilot, laptop | browser audio decodes; no web framework installed; run files hold no reply audio | — | — |
| 4 | 09-27, night | the demo on the laptop's CPU, scratch | the page, the measured part, and a live turn through three rungs | `src/s2slab/demo.py`, `demoweb.py`, `demo_page.html`, tests, `requirements/demo.txt` | 83a67e5 |
| 5 | 09-27, night | the demo on the laptop, by the author | the measured part loaded; the one live request was refused, 401; no live turn yet | — | — |
| 6 | 09-27, night | FT-2's recognizer as a rung, laptop, scratch, on a stand-in model | the rung loads from a folder and answers; refuses a model that is not PR-011's | `src/s2slab/r2verbatim.py`, `rungs.py`, `demo.py`, `demoweb.py`, `demo_page.html`, tests | this commit |

## 1. The design

Asked for by the author on 2026-09-27: the ladder shown on demand, for the portfolio. Neither plan
has a live demo. v2's week 8 says "publish"; v1 §1 calls "here's my voice bot, talk to it" the demo
trap, and §6 scopes the client transport out. So the demo shows measured turns first and live
turns second, and the live part says what it is: one turn on whatever machine is awake.

| Decision | Chosen |
|---|---|
| Who can use the live part | an access code the author hands out; with none configured, nobody |
| Daily spend cap | $1 a day, enforced in the app |
| Audio for the measured part | four LibriSpeech test-clean clips, with attribution (CC BY 4.0) |
| A visitor's audio | decoded in memory, handed to the rungs, discarded; said so on the page |
| Where the page lives | served by the Modal app itself |

| Phase | Adds | GPU |
|---|---|---|
| 1 | the measured part; live R0, R1, R2, and R2 with FT-2's recognizer | T4 |
| 2 | live R3 | A10 |
| 3 | live R4 and R5 (R5 over WebSocket) | L40S, A10 |
| 4 | barge-in in live playback | T4 |

**"The better versions".** What the author asked for. Ready to show: R1 over R0, R2's slow tier,
PersonaPlex beside Moshi. Measured but not wired into a rung: FT-2's verbatim recognizer (phase
1), the barge-in policy with truncation (phase 4). Fine-tunes that did not improve what they were
tested on, which the demo can show as such: FT-1's LoRA, FT-3's voice, 8-bit Kokoro.

## 2. The documents that were owed (6037535)

The repo had no `README.md` at its root and no backlog file, after 124 commits. The author had
asked more than once for every document to be kept, in Notion and in the repo. Mine to have kept.
Both written, shown and committed. From here every milestone's hand-over carries a documentation
check.

Found while checking, and not fixed: six report files under `runs/` (M16 to M20) and three
comments in `src/s2slab/` hold the author's local path. In `docs/backlog.md`, as a decision for
the author before any push.

## 3. The pilot

| Question | Answer |
|---|---|
| Does browser audio decode? | yes: WebM and Ogg Opus, MP4 AAC, all to 16 kHz by PyAV 17.1.0, which `faster-whisper` already brings, here and in the Modal image |
| Is a web framework installed? | no: `fastapi`, `uvicorn`, `python-multipart` installed by the author, pinned in `requirements/demo.txt` |
| Do the run files hold what the measured part needs? | each turn's transcript, reply text and stage times, yes; **the reply's audio, no** |

**One change to the design followed.** The measured part plays the caller's clip and shows what
was heard, what was replied and where the time went; it does not play a reply. Rendering the reply
again would not be the recorded turn: Kokoro does not speak a sentence the same way twice.

## 4. Commit 1: the demo on the laptop

- `src/s2slab/demo.py`: decoding, the access code, the spend cap, the measured data, a live turn
  as a record. No web framework.
- `src/s2slab/demoweb.py`: the FastAPI app around it, and `laptop`, the app on the CPU rungs.
- `src/s2slab/demo_page.html`: one page, two tabs, no script or font fetched from anywhere.
- Nine tests, on made-up run files, the stub rung, and audio encoded in the test.

**The spend cap counts the time the GPU is up**, not the time a request takes: the first request
of a burst is charged a cold start (45 s, an estimate until it is measured), its own time and the
scale-down window (60 s); a request that finds the container up is charged only the time it adds.
A turn on a sleeping T4 comes to about two cents.

**Run before the hand-over**, in a scratch copy of the tree on the laptop's CPU:

| What | Result |
|---|---|
| the nine tests | passed |
| the page in a browser, from the real run files | all seven rows shown, the four clips listed |
| a live turn without the code, and with a wrong one | refused, 401 |
| a live turn from a WAV, through `r0`, `r1`, `r2` | three replies, each with its stages and its audio |
| a live turn from a WebM (Opus) file | the same |
| anything written to disk by a live turn | nothing |

**Not run:** the microphone in a browser (the recording path ends in the same upload as a chosen
file); anything on Modal; any GPU.

**Seen on the laptop, for the record:** three rungs loaded at once share the CPU, and a turn took
8 to 44 s a rung. The laptop is for development; the page says which machine answered.

**A fault found by writing it:** FastAPI reads a handler's annotations as objects, and a module
with `from __future__ import annotations` hands it strings it cannot resolve when the types are
imported inside a function. The web app is its own module, without that import. It never ran
wrong; it was split before its first test.

## 5. The author's first try (after 83a67e5)

837 tests passed on the author's machine. The server's log of the try:

| Request | Answer |
|---|---|
| the page, `/api/measured`, `/api/status`, a clip | 200 |
| `/favicon.ico` | 404: the page named no icon |
| `POST /api/live`, once | **401**: the access code sent was not the one the server was started with |

So on the author's machine the measured part is confirmed, and the live part is confirmed only to
refuse. No live turn and no microphone recording has run there yet. Not a fault in the code as
far as the log shows: the same route answered in the scratch run (entry 4). Still to do, by the
author: one live turn with the code the server was started with, from the microphone.

The icon: the page now names an empty one, so a browser asks for nothing.

## 6. Commit 2: FT-2's recognizer as a rung

**What was missing.** PR-011's model had been run once, through the recognizer class alone, on
Modal. No rung loaded it, so nothing could put it in a conversation.

- `src/s2slab/r2verbatim.py`: `r2-verbatim` (CPU, int8) and `r2-gpu-verbatim` (fp16). R2 with one
  change, the recognizer: `VerbatimASR` loads the export from a folder (`S2SLAB_FT2_CT2`, or
  `data/ft2/ct2`).
- **The model is held to PR-011's.** The folder's `model.bin` must have the SHA-256 the
  registration recorded (`b31833fe...a51cff`), checked before anything loads. A test holds the
  constant to `runs/ft2/pr011.json`.
- `demo.py`: the measured part gains a panel from `runs/ft2/pr011.json`: two disfluent clips, what
  was said, what `small.en` wrote, what the LoRA wrote, and the cohort's numbers beside them.
- `demoweb.py`: the laptop offers "R2 verbatim" only where PR-011's export is; the clips the page
  may play are the four clean ones and the panel's two, and no other.

**The two clips are chosen for the size of the difference, and the page says so.** The rule:
the two disfluent clips where the LoRA removed the most raw errors (`pick_verbatim`, held by a
test): `disfluent-29` and `disfluent-42`. They are the best cases, not typical ones, so the
cohort's numbers stand beside them: raw WER 9.9% to 2.3% over 50 clips, fillers kept 1 of 29 and
29 of 29. The page also carries PR-011's limit: one speaker, performing disfluency, trained on and
tested on. Nothing measured says the model writes down a visitor's "um".

The clips are DisfluencySpeech's test split, Apache-2.0, attributed on the page.

**Run before the hand-over**, in a scratch copy of the tree on the laptop's CPU. The export is on
the Modal volume `ft2`, not on the laptop, so the loading route ran on a STAND-IN: the cached
`small.en` folder, which has the same layout.

| What | Result |
|---|---|
| the new tests (4 in `test_r2verbatim.py`, 3 in `test_demo.py`) and the rest of the suite | 839 passed, 5 skipped for audio the scratch copy does not hold |
| `r2-verbatim` given the stand-in | refused: not PR-011's SHA-256 |
| `r2-verbatim` given no export | refused, with where to put it |
| the class told the stand-in's own hash | loaded from the folder; one turn on `disfluent-29`, recognizer 2.9 s, first audio 9.4 s |
| the web app, no export; and another model in the export's place | three rungs offered; "R2 verbatim" refused, 400 |
| the web app, the stand-in passed off as the export | four rungs offered; one upload answered by `r2` and `r2-verbatim` |
| the panel, from the committed run file | two clips, the cohort's numbers, both clips served; `disfluent-47` not served |

**Not run:** PR-011's model in a rung, anywhere. The stand-in is `small.en`, so what it heard says
nothing about the LoRA. The model's first turn inside a rung will be on Modal (commit 3), or on
the laptop if the author brings the export down (`ft/README.md`).

**A fault found by the pilot, fixed before the hand-over.** The first version offered the tier
wherever a `model.bin` was, and checked the hash only at load: with another model in the folder
the page offered a tier that then failed, a 500. Now the tier is offered only where the hash
matches. It never ran outside scratch.

**Left for commit 3:** a rung that fails to load during a live turn still answers 500. The
visitor sees no detail, but the page should say the rung is not available.
