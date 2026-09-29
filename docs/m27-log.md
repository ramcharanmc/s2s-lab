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
| 6 | 09-27, night | FT-2's recognizer as a rung, laptop, scratch, on a stand-in model | the rung loads from a folder and answers; refuses a model that is not PR-011's | `src/s2slab/r2verbatim.py`, `rungs.py`, `demo.py`, `demoweb.py`, `demo_page.html`, tests | ef3d827 |
| 7 | 09-28 | the demo on the laptop, by the author, from the microphone | a live turn answered by the rungs: the live part and the microphone confirmed | — | — |
| 8 | 09-28 | the Modal app, laptop, scratch, with a function in the T4's place | the web app, the door and the spend counter run; nothing has run on Modal | `modal_app/demo.py`, `demo.py`, `demoweb.py`, `demo_page.html`, tests | 15a0aa3 |
| 9 | 09-28 | `modal serve modal_app/demo.py`, by the author | both images built; the page and the measured part served from the CPU container; one live turn answered by all four rungs on a T4, PR-011's model among them; the wake 72 s | — | — |
| 10 | 09-28 | the tests, laptop, scratch | Modal's warning in that run: the live handler was `async` and called the spend counter, which blocks. Mine: written and tested against a dict | `demoweb.py`, a test | ea1d37a |
| 11 | 09-28 | a count over `runs/ft2/pr011.json`, laptop | on 100 read clips with no filler in the reference the LoRA wrote none; on 50 accented clips it wrote 20 to the references' 31, 3 of them not in the reference | — | bd529bc |
| 12 | 09-28 | the tests, laptop, scratch | wrong codes are counted and close the door; a code under ten characters is no code | `demo.py`, `demoweb.py`, `demo_page.html`, tests | c58e588 |
| 13 | 09-28 | `modal serve`, by the author, with a new code | the new code was shorter than ten characters: the live part stayed closed and said so; the one live request refused, 401; no turn, the T4 never woke | — | — |
| 14 | 09-28 | the page in a browser, laptop, scratch, a made-up microphone | the author: "I dont know when it started and stopped". The page now shows that it is recording, the time, the limit, and plays the recording back before it is sent | `demo_page.html`, a test | 22fce45 |
| 15 | 09-28 | `modal serve`, by the author, a code of ten characters or more | two live turns through four rungs on a T4: one that woke it (the wake 47 s) and one with it up (3.1 s for all four); Modal's warning gone; the recorder used with a real microphone | — | 430c522 |
| 16 | 09-28 | the tests and the page in a browser, laptop, scratch | prosody in the demo: a measured panel from PR-006's run, and a live card of the visitor's pitch and range beside the words the rungs received | `demo.py`, `demoweb.py`, `demo_page.html`, `modal_app/demo.py`, tests | cdb890b |
| 17 | 09-28 | `modal serve`, by the author, his own voice | the voice card and the prosody panel on Modal; two live turns, and **both woke the T4**: it had gone to sleep in between | — | 2661338 |
| 18 | 09-28 | the tests, laptop, scratch | the T4's window 300 s, the author's choice after entry 17 | `modal_app/demo.py`, `demo_page.html`, a test | 87ba16c |
| 19 | 09-28 | the plan for the rest of the demo | seven decisions by the author; R5 as a conversation and barge-in live left out, and why | — | — |
| 20 | 09-28 | the tests and the page in a browser, laptop, scratch | R3 and R4 live, each on a machine of its own that wakes only when asked; one cap over three machines | `demo.py`, `demoweb.py`, `demo_page.html`, `modal_app/demo.py`, tests | ab45907 |
| 21 | 09-28 | `modal serve`, by the author | R3 and R4 on Modal for the first time, both asked for at once; a second turn that found all three machines up | — | — |
| 22 | 09-28 | the tests and the page in a browser, laptop, scratch | R5 on the visitor's recording: Moshi, on an A10 of its own, hears it as one side of a call | `demo.py`, `demo_page.html`, `modal_app/demo.py`, tests | 7650d36 |
| 23 | 09-28 | `modal serve`, by the author | the sixth session: Moshi in the demo for the first time, one recording answered on four machines; the call ran at 1.67 of real time and was not answered | — | — |
| 24 | 09-28 | the tests and the page in a browser, laptop, scratch | the measured panels for five more registrations, from their run files; the note on R3's card; a throwaway call when Moshi loads | `demo.py`, `demo_page.html`, `modal_app/demo.py`, tests | b4b4dd7 |
| 25 | 09-28 | the plan for what is left | the author's questions (the portfolio; whether Moshi was tuned as a patient) and seven decisions | — | — |
| 26 | 09-28 | the tests and the page in a browser, laptop, scratch | one request a machine: each machine's cards appear when it has answered | `demoweb.py`, `demo_page.html`, tests | 97e7af3 |
| 27 | 09-28 | the tests, the render on the laptop's CPU, and the page in a browser; scratch | the severity render, to be heard: PR-009's sentences, voices and scores, and no other | `demo.py`, `demoweb.py`, `demo_page.html`, `modal_app/demo.py`, tests | fb601a5 |
| 28 | 09-28 | `modal serve`, by the author | the seventh session: the severity render on the T4, one render; no live turn, the day's budget being nearly spent | — | 79c42eb |
| 29 | 09-28 | the tests, laptop, scratch | before the repo is public: the MIT licence; the author's local path out of nine files; what stays in the history | `LICENSE`, six reports, three source files, `runs/README.md` | ff405b4 |
| 30 | 09-29 UTC | `modal serve`, by the author | the eighth session: two severity renders, one that woke the T4 (50.5 s) and one with it up (1.77 s) | — | — |
| 31 | 09-29 | `modal serve`, by the author | the ninth session: two live turns, every box ticked, on four machines; commits 11 and 12 confirmed on Modal; Moshi at 0.68 of real time after its throwaway call | — | eb1974a |
| 32 | 09-29 | `modal deploy`, by the author | the demo deployed at a lasting address; checked from a browser, no GPU woken; the portfolio's link drafted | — | this commit |

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

## 7. The author's live turn (after ef3d827)

844 tests passed on the author's machine. Then one turn from the microphone, on the laptop's CPU,
with the code the server was started with. Read from the author's screenshot, which shows the
first two cards:

| Rung | Recognizer | LLM | Speech | First audio | Whole reply |
|---|---:|---:|---:|---:|---:|
| R0 | 1.25 s | 11.71 s | 3.66 s | 16.62 s | 16.62 s |
| R1 | 2.91 s | 16.00 s | 3.82 s | 22.73 s | 22.73 s |

The recording: 233 kB, 14.82 s. Both heard the same words and gave the same one-sentence reply.
R2's card is below the screenshot's edge: not seen by me.

What this confirms: the microphone path, the access code, and a live turn through the rungs, on
the author's machine. Entry 5's open item is closed.

What it does not show, and why:

- **R1 is not sooner than R0 here.** The reply was one sentence, so R1 had one clause to hand
  out: its first audio is its whole reply. And the rungs ran one after another on a CPU they
  share with each other's loaded models. One turn on a laptop is not a measurement: PR-002's is.
- **The recording stopped at the 15 s limit**, in the middle of the author's sentence. The page
  stops the recorder at 15 s and does not say so beforehand. Noted for the page; not changed in
  this commit.

The author asked whether the wait is all right. On the laptop it is what a CPU gives: the LLM is
most of it. On a T4 the same rungs measured 0.66 s and 0.49 s to first audio (p50, PR-002).

## 8. Commit 3: the Modal app

`modal_app/demo.py`. Built, and run on the laptop with a function in the T4's place. **Nothing in
it has run on Modal**: no image built, no secret read, no volume mounted, no GPU.

**One change to the design the author approved, made while building it, and shown to him before
it was applied.** The design note had one web app on the T4. That wakes the GPU for every visit
to the page and every crawler, outside the access code and outside the cap. So there are two
containers:

| Container | Holds | Woken by |
|---|---|---|
| web, CPU | the page, the measured data, six clips, the status; the access code, the budget, the decoding | any request |
| rungs, T4 | the four rungs, loaded at start | a live request that has passed the code and the budget |

The five decisions, as approved: R0, R1, R2 (`small.en`) and R2 verbatim live; no turbo; all four
loaded at start; the wait said on the page, with the last measured wake; `modal serve` before any
`modal deploy`.

- `demo.py`: `Live.load_all()` and `Live.turns()`; a rung that does not load, or fails on a
  recording, is a card with a sentence while the others answer; `Remote` and `answer()`, the two
  ends when the rungs are on another machine.
- `demoweb.py`: the turns run on a worker thread, one recording at a time, so the page is served
  while they work; a machine that does not answer is a 503 with a sentence; the wake is measured
  on every cold turn (the wait, less the turns' own time) and kept for the page.
- `demo_page.html`: a failed rung's card; the wait and the wake in words; the note on R2.
- The spend counter is a `modal.Dict`; the access code a Modal secret the author creates.

**The cap on Modal** counts the wall time of a request that woke the T4, plus the 60 s window: the
wake is inside the wall time, so the 45 s estimate is not added. A request that fails after the
door is charged too. Not counted: both containers' CPU and memory. Modal's dashboard is the bill.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the new tests (4 in `test_modal_demo.py`, 2 in `test_demo.py`) and the rest of the suite | 845 passed, 5 skipped for audio the scratch copy does not hold |
| `modal_app/demo.py` imported with Modal 1.5.5 | one app, the class `Rungs`, the web endpoint `web` |
| its web app, the T4's end played by the CPU rungs `r0`, `r1`, `r2`; the verbatim rung without its export | four rungs offered |
| a live request with a wrong code | 401; the far end was not called |
| a live request with the code, twice | 200 both; the first marked cold, the second not; both charged |
| the verbatim rung, its export absent | a card with a sentence; the other three answered |
| anything written to disk by the turns | nothing |
| local paths in the changed files | none |

The laptop was busy during the pilot: the three rungs took 636 s to load and 40 to 62 s a turn.
Those times say nothing about the code or about a T4.

**Not run, and first run by the author:** the image builds; the secret; the volume mount and the
hash check of the export on it; four rungs in one T4's memory (about 1.7 to 1.9 GB each when
measured alone; not measured together); the wake; a request that waits longer than Modal's 150 s
limit for a web request; the microphone over the public URL.

## 9. The first run on Modal (at 15a0aa3)

850 tests passed on the author's machine. Then `modal serve modal_app/demo.py`, with the secret
`s2s-lab-demo` created by the author beforehand. The first time any of it ran on Modal. No stage
failed.

| What had not run before | What happened |
|---|---|
| the two image builds | both built |
| the secret | read: the status reported a code as set, and the author's code was accepted |
| the spend counter, a `modal.Dict` | \$1.00 before the turn, \$0.98 after |
| the page and the measured part, from the CPU container | served: seven rows, four clips, the verbatim panel with its two clips. Opened by me, read-only, before any live turn; the T4 stayed asleep (the dashboard showed `Rungs.*` inactive, no activity) |
| the volume `ft2` and the export's hash check | the verbatim rung loaded and answered, which it does only if `model.bin` has PR-011's SHA-256 |
| four rungs on one T4 | all four loaded and answered. GPU memory was not measured; the dashboard showed 5.2 GiB of the container's memory |
| the wake | **72 s**, measured by the app: the wait, less the turns' own time. Of it 62.8 s was the four rungs loading (the container's own line; none failed to load), so about 9 s was the container starting. The 45 s in the laptop's budget was an estimate, and low |
| Modal's 150 s limit on a web request | not reached: the request took 81.4 s end to end |
| the microphone over the public address | worked: recorded, 163 kB, 10.32 s |

**The one live turn**, from the author's screenshot. One recording, the T4 just woken, the rungs
one after another in this order:

| Rung | Recognizer | LLM | Speech | First audio | Whole reply |
|---|---:|---:|---:|---:|---:|
| R0, `r0-gpu` | 137 ms | 760 ms | 184 ms | 1.08 s | 1.08 s |
| R1, `r1-gpu` | 99 ms | 443 ms | 460 ms | 517 ms | 1.00 s |
| R2, `r2-gpu` | 181 ms | 417 ms | 333 ms | 485 ms | 931 ms |
| R2 verbatim, `r2-gpu-verbatim` | 195 ms | 340 ms | 284 ms | 476 ms | 819 ms |

Charged to the day's budget by the app: \$0.0223. On the laptop the same rungs had taken 16.6 s
and 22.7 s to first audio (entry 7).

**How to read it: as one turn, not as a measurement.**

- The LLM's time fell from each rung to the next (760, 443, 417, 340 ms) although three of the
  four replies begin with the same words. That is the shape of a machine warming up, in the order
  the rungs ran, and it is mixed into every difference between the cards. PR-002 and PR-004 are
  the measurements: 200 turns each.
- R1 against R0 is the one difference the turn shows by construction: R1's first audio came
  before its reply was whole (517 ms of 1.00 s); R0's could not.
- **The verbatim rung wrote "uh" twice and "wanna"; R0, R1 and R2 wrote none and "want to".**
  PR-011's model, on a speaker it was not trained on. It is one recording, and what the author
  said is known only to the author: no reference exists for it. It is not evidence for the
  claim PR-011 did not make; it is the first time the question has an example.
  **Asked afterwards, the author is not sure whether he said them**, and is not to be held to
  either answer. So what was said is not known. PR-011 tested whether fillers that are there are
  kept (29 of 29); whether fillers that are not there are written had not been looked at.
  Entry 11 looks.
- The recognizers of R0 and R1 are the same model and heard the same words.

**Addresses.** The author first looked for the demo at the laptop server's address, and then at
the link `modal serve` prints last, which is Modal's dashboard. The demo's own address is printed
once, above the list of watched folders, and had scrolled away. My hand-over said only that a URL
would be printed. The README now says which is which.

**Seen in the terminal:** a warning from Modal, eight times, once for each call to the spend
counter from the live handler, an `async` function. Entry 10. And after Ctrl+C, `FATAL: exception
not rethrown` and a core dump from Modal's client as it shut down; the app had stopped.

**The author's paste of the run showed the access code** (an `echo` of the laptop's variable).
It is in no file of the repo. To be replaced by a longer one before any deploy; the app has no
limit on wrong attempts yet.

**Not seen by me:** a second turn, with the T4 already up. The author sent one.

## 10. Commit 4: the live handler is not `async`

**The fault.** `POST /api/live` was an `async` handler, and it reads and writes the spend counter.
On Modal the counter is a `modal.Dict`, whose plain calls block; from an `async` function they
hold up everything else the web container is serving. Modal warned of it on the first live
request. The turn was answered and charged correctly.

**Cause, and whose.** Mine. The handler was written for a dict, and tested with a dict, which
cannot block. Commit 3 moved the rungs to a worker thread and left the counter's calls where they
were.

**The fix.** No handler is `async`. FastAPI runs a plain function on a worker thread, so the
counter, the decoding and the rungs are all off the event loop, and the page is served while a
turn waits. A test holds every handler to it.

**Run before the hand-over:** the full suite in a scratch copy, 845 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold). **Not run:** the
fixed handler on Modal; the warning's absence is for the author's next `modal serve` to show.

## 11. Does FT-2's model write fillers that were not said? A look, not a test

Raised by entry 9. Answered as far as committed data can: PR-011's run file holds what `small.en`
and the LoRA wrote on all 200 eval clips. Counted: the words uh, um, uhm, er, erm, ah, hmm, mm,
mhm in each transcript, against the same words in the clip's reference. No model ran; nothing
was written. Not registered: the question came after the data.

| Cohort | Clips | In the references | `small.en` wrote | The LoRA wrote | Of those, not in the reference |
|---|---:|---:|---:|---:|---:|
| clean, read | 50 | 0 | 0 | 0 | 0 |
| slow, read | 50 | 0 | 0 | 0 | 0 |
| accented, spontaneous | 50 | 31 | 0 | 20 | 3 |
| disfluent | 50 | 29 | 1 | 29 | 0 |

- **On read speech it adds none**: 100 clips, no filler written.
- **On spontaneous speech from other speakers it writes fillers `small.en` drops**: 20 of the
  references' 31, where `small.en` wrote none. PR-011 claimed nothing beyond its one speaker;
  this is the first sign the behaviour carries.
- **The three not in the reference are weak**: "ah" where the reference has "oh"
  (`accented-32`), and one more "uh" (`accented-26`) and "um" (`accented-10`) than the
  reference, in clips whose references hold fillers already. A reference can miss a filler.
- "wanna": once in 200 clips, where the reference has it too.

**For the author's recording**, the likelier reading is that the fillers were said and the three
other recognizers dropped them, as `small.en` does in every row above. Not shown: there is no
reference for that turn.

**Mine to correct:** I wrote "the eval set's 150 fluent clips" in the backlog before looking.
100 are read; the accented cohort is spontaneous speech.

## 12. Commit 5: the door, before any deploy

**What was open.** The live part took any number of wrong codes, and the first code was five
digits: a hundred thousand tries find it, and nothing counted them. The code also appeared in a
paste (entry 9).

- **Wrong codes are counted** (`demo.Door`). Five inside ten minutes close the live part for ten
  minutes, to everyone, the right code included. A request with no code is not counted. A closed
  door answers 429 before it looks at the code, reaches no rung and charges nothing.
- **A code shorter than ten characters is treated as no code.** The live part stays closed and
  the page says why. So the five-digit code cannot be deployed by oversight.
- The count lives where the spend does: a dict on the laptop, the `modal.Dict` on Modal.

**What it costs, said plainly.** The count is one for all visitors. Behind Modal's proxy a
visitor's address is what a header says, so a count for each address would be a count an attacker
resets at will. The price: someone who sends wrong codes can keep the live part closed. They
cannot open it, and closed it costs nothing. The measured part is not behind the door.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite,
848 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold); three new tests among them (the count and its window; the locked door refusing the
right code and opening again; the short code).

**Not run:** the door on Modal, where its count is in the `modal.Dict`. **For the author:** a
new code of ten characters or more, put in the secret with `--force`; the old one is refused now.

## 13. The second session on Modal: the door held, and no turn ran (at c58e588)

853 tests passed on the author's machine. A new code went into the secret with `--force`, which
Modal accepted. Then `modal serve`.

| What | Result |
|---|---|
| the images | not rebuilt: both came from the first run |
| the page, the measured part, the status | served |
| the status line | "The access code configured is shorter than 10 characters, so the live part is closed." |
| `POST /api/live`, once | **401** |
| the T4 | never woke: one container, the web one |
| Modal's warning (entry 10) | not printed; but no turn ran, so the calls that raised it were not all reached. Not settled |

**Commit 5 did what it was written for**, on Modal, the first time it ran there: a code too
short was refused, and the page said why.

**What the page did badly.** It let the author record and send although the live part was
closed, and answered "the live part needs an access code" under a status line that had already
said why. Entry 14 turns Send off while the live part is closed.

**Counted by the door:** one wrong code. **Cost:** the web container's CPU for some minutes;
no GPU. No ledger row: nothing was billed at a GPU's rate.

**Still open:** the second live turn, with the T4 up. It needs a code of ten characters or more.

## 14. Commit 6: the recorder says what it is doing

**The author, after two sessions:** "The recording is very primitive - I dont know when it
started and stopped". The page changed a button's word from Record to Stop and showed nothing
else; at 15 s it stopped without a word (entry 7).

| Before | Now |
|---|---|
| the button says Stop | the button turns red and says Stop; a line says **Recording**, with a blinking dot, the time running, the limit, and a bar filling |
| it stops at 15 s, silently | the limit is said before recording; it stops half a second before the server's limit, and says "it was stopped at the limit" |
| "recorded, 163 kB" | "Recorded 9.4 s, 163 kB. Listen, then send." and a player with the recording |
| Send is on whenever a recording exists | Send is off while the live part is closed, while recording, and while a turn is out |
| "Sent." for however long it takes | "Sent 12 s ago." counting |

**Why half a second early:** the server refuses a recording longer than 15 s, and a recorder
stopped at 15 s by a timer can hold a little more.

**The visitor's recording is still not stored.** The player reads it from the page's memory;
nothing is sent until Send.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 848 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the page in a browser, from the scratch copy's server, the recorder fed a tone | while recording: the button red and "Stop", the line "Recording", the seconds counting, "of 14.5 s", the bar filling, Send off |
| Stop pressed after 3.3 s | "Recorded 3.3 s, 42 kB. Listen, then send.", the player holding the recording, Send on, the button "Record again" |
| left to run | stopped by itself: "Recorded 14.5 s, 227 kB: it was stopped at the limit." |
| that 14.5 s recording sent, with a made-up code, to two stub rungs | answered: "One turn of 14.4 s", two cards; the server did not refuse it for length |
| the same page with a code of five characters configured | the status line in the warning colour, saying why; a recording made; Send off |
| errors in the browser's console | none |
| "Sent 12 s ago", counting | not seen: the stub rungs answer at once |

**Not run:** a real microphone: the browser I can drive has none, so the recorder was fed a made-up
one (a tone). The author's microphone is the test.

## 15. The third session on Modal: two turns, one of them warm (at 22fce45)

853 tests passed on the author's machine. A code of ten characters or more in the secret, then
`modal serve`. No stage failed.

| What | Result |
|---|---|
| the door | open: the code was accepted |
| the four rungs | loaded in 40.8 s, none failed (62.8 s in the first run) |
| the first turn, the T4 asleep | 200, 51.9 s end to end; the wake **47 s** as the app measures it (72 s in the first run) |
| the second turn, the T4 up | 200, **3.1 s end to end for all four rungs**, the recording's upload and the replies' download included |
| Modal's warning (entries 9, 10) | **gone**: two turns ran, every call to the spend counter was reached, nothing was printed. Commit 4 is confirmed on Modal |
| the recorder, with the author's microphone | used: "Recorded 8.7 s, 134 kB. Listen, then send.", the player holding the recording, the button "Record again". The author has not said whether it is now clear when it starts and stops |
| the budget | \$0.96 of \$1.00 left after the session; the first turn was charged \$0.0182 |

**The first turn's cards**, from the author's screenshot. One recording of 8.5 s, the T4 just
woken, the rungs one after another:

| Rung | Recognizer | LLM | Speech | First audio | Whole reply |
|---|---:|---:|---:|---:|---:|
| R0, `r0-gpu` | 78 ms | 808 ms | 172 ms | 1.06 s | 1.06 s |
| R1, `r1-gpu` | 63 ms | 421 ms | 799 ms | 562 ms | 1.28 s |
| R2, `r2-gpu` | 159 ms | 425 ms | 471 ms | 347 ms | 1.06 s |
| R2 verbatim, `r2-gpu-verbatim` | 139 ms | 404 ms | 387 ms | 325 ms | 930 ms |

**The second turn's cards were not sent to me.** Both screenshots show the first turn (its line
says the machine was asleep). So the warm turn is known by its total only: 3.1 s for four
replies, against 4.3 s for the four in the first turn.

**How to read it.** As entry 9: one turn, not a measurement. The same shape again: R0, the
first rung to run, has the slowest LLM (808 ms against 404 to 425 ms), which is warm-up, not R0.
Two first turns now agree on it, so the first rung's card after a wake should not be compared
with the others. (Entry 17 has a wake where it did not happen, and a second cause: the length
of the reply.)

**The verbatim rung wrote no filler this time.** All four heard the same words: "Hi, this is
[a first name]. Let's see if the service is able to answer. Thank you." The verbatim rung differs
by a comma.
Its reply differs from the other three, which is the LLM answering a different punctuation. One
recording; it agrees with entry 11 (none added where none is said), and whether the author chose
the sentence beforehand is not known to me.

**A name was said in the recording.** It is left out here: the repo holds no personal detail,
and a visitor's words are not kept. The screenshot with it is the author's, outside the repo.

**Cost.** About \$0.02 of T4 time: the app's count for the first turn, \$0.0182, and a second
turn inside its window. The dashboard's figure was not sent.

## 16. Commit 7: prosody in the demo

**The author, after the third session:** "are we also focusing on prosody, pitch, emotional
variations for live". The live part did not: its four rungs are cascades, and a cascade hands its
language model words. How they were said is gone by then. That is the project's first point
(v2 §2, the input bottleneck), and the page did not show it.

**The measured panel, "What the words do not carry".** From `runs/prosody/pr006.json`: R3 asked
"Is the person speaking upset?" of two deliveries of one sentence by one actor. Three pairs, one
for each emotion, and the registered result beside them: the loaded delivery scored higher in 13
of 13 pairs for each emotion; Yes to it and No to the plain one in 11, 11 and 9 of 13.

**The live card, "What your voice carried".** Median pitch, pitch range, and words a second,
beside the words the rungs received. A second recording shows the change from the first, so one
sentence said two ways shows a voice that moved and words that did not. The comparison is in
the page's memory; nothing is kept by the server.

| Decision, as approved | Built |
|---|---|
| CREMA-D's audio is not served | the panel shows the sentence and R3's scores; no clip of it is in the image or on the page |
| the pair with the middle difference in each emotion | `pick_pairs`, held by a test: `sad-t05`, `fea-t04`, `ang-t10` |
| no emotion label on the live card | numbers only; the card says it cannot tell how anyone feels; a test holds that no label is sent |
| no loudness | not sent |
| compare with the recording before | in the page's memory |
| measured in the CPU container | beside the door, before the rungs: shown even if they fail; a measure that fails costs no turn |

**One change to the design the author approved, made while building it.** He approved "median
pitch, pitch range, speech rate, and pauses". Speech rate and pauses from the audio are left
out. `prosodycheck.py` records why, from this project's own dev pairs: both ways of timing speech
by its energy turned a difference in LEVEL into a difference in LENGTH, and a sentence said upset
is 10 to 15 dB louder than the same sentence said plainly. That is the comparison the card
invites. In their place: words a second, from the first rung's transcript over the recording's
whole length, silences included, and said so on the card.

**Limits, said on the page.** Pitch is looked for between 75 and 600 Hz. CREMA-D is acted
emotion, and a benchmark the model may have met. R3 ranks; its Yes and No are not calibrated.
Words a second counts the silence before and after the speech.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 852 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the measured panel in a browser, from the committed run file | the three pairs with their sentences and R3's scores (0.39 and 0.98; 0.20 and 0.99; 0.22 and 0.98), the registered counts, the citation; no audio element in it |
| a live turn in a browser, the recorder fed a tone of 130 Hz, through WebM to the server and Praat | the card: "Median pitch 130 Hz", range 0.0 semitones, 2.4 s with pitch |
| a second turn, a tone of 195 Hz | "Median pitch 195 Hz (+7.0 semitones)": the change from the first, which is what 195 over 130 is |
| errors in the browser's console | none |
| words a second | not seen: the stub rungs hear no words, and the card then leaves the rate out |

**Not run:** a real voice through the card; the CPU image's rebuild on Modal with Praat in it.
Both are the author's next `modal serve`.

## 17. The fourth session on Modal: the voice card, and a T4 that slept between turns (at cdb890b)

857 tests passed on the author's machine. Then `modal serve`. No stage failed.

| What | Result |
|---|---|
| the CPU image, with Praat added | rebuilt in 2.6 s |
| the prosody panel and the voice card | served; the card filled from the author's voice, twice |
| a live request with a wrong code | 401, once; counted by the door |
| the first turn | 200, 63.7 s end to end; the rungs loaded in 50.1 s; the wake 59 s |
| the second turn | 200, 63.0 s end to end; **the rungs loaded again**, in 45.0 s; the wake 54 s |
| Modal's warning (entry 10) | not printed |
| the budget | \$0.93 after the first turn, \$0.91 after the second; charged \$0.0200 and \$0.0193 |

**Both turns woke the T4.** It sleeps 60 s after its last turn. Between sending one recording
and sending the next the author listened to four replies, read four cards and recorded again,
which took longer than that. So a visitor who uses the page as it is meant to be used waits
about a minute for every turn, and pays a wake each time. The warm turn of entry 15 (3.1 s)
was a recording sent again at once. Not a fault in the code: the window was chosen for batch
runs, where the next call comes at once. For the author to decide: a longer window costs more
for each burst of use and less waiting inside it.

**What the voice card showed**, from the author's screenshots:

| Turn | Recording | Median pitch | Pitch range | Words a second | Seconds with pitch |
|---|---:|---:|---:|---:|---:|
| 1 | 5.2 s | 133 Hz | 3.1 semitones | 1.7 | 1.6 |
| 2 | 7.0 s | 126 Hz (−0.9 semitones) | 6.1 semitones (+3.0) | 2.7 (+1.0) | 3.6 |

The card worked on a real voice, through the browser's recorder, the decoder and Praat, in the
CPU container. **The two recordings were different sentences**, so the change in brackets is
between two sentences and says nothing of delivery: the comparison the card is for, one sentence
said two ways, has not been made yet.

**The cards**, the rungs one after another, each turn on a T4 just woken:

| Turn | Rung | Recognizer | LLM | Speech | First audio | Whole reply |
|---|---|---:|---:|---:|---:|---:|
| 1 | R0 | 67 ms | 348 ms | 122 ms | 536 ms | 536 ms |
| 1 | R1 | 62 ms | 228 ms | 458 ms | 301 ms | 748 ms |
| 1 | R2 | 115 ms | 238 ms | 410 ms | 352 ms | 764 ms |
| 1 | R2 verbatim | 131 ms | 264 ms | 414 ms | 379 ms | 809 ms |
| 2 | R0 | 95 ms | 790 ms | 250 ms | 1.13 s | 1.13 s |
| 2 | R1 | 138 ms | 391 ms | 299 ms | 652 ms | 829 ms |
| 2 | R2 | 168 ms | 424 ms | 282 ms | 733 ms | 875 ms |
| 2 | R2 verbatim | 165 ms | 473 ms | 274 ms | 777 ms | 913 ms |

**A correction to entries 9 and 15.** I read R0's slow LLM after a wake as warm-up. In turn 1
here R0's LLM took 348 ms after a wake, not 760 to 808: its reply was one short sentence. So the
LLM's time follows the length of what it says as well as the order the rungs ran in, and one
turn cannot separate the two. The reading that stands: a card is one turn, and the differences
between cards are not the differences between rungs. The measurements are PR-002 and PR-004.

**The two tiers heard different words**, in turn 2: R0 and R1 (`base.en`) wrote "on how it
responds. I want to see"; R2 and R2 verbatim (`small.en`, and `small.en` with the LoRA) wrote
"and how it responds. I wanna see". Which is right is not known: no reference. It is R2's point
(the slow tier hears other words than the fast one) seen live for the first time. The verbatim
rung wrote no filler in either turn; in turn 1 it differed from the others by one comma.

**Cost.** About \$0.04 of T4 time: two wakes, by the app's count \$0.0200 and \$0.0193. The
dashboard's figure was not sent.

**Said by the author afterwards:** "recorder is clear now". Entry 14's open question is closed.

## 18. Commit 8: the T4 stays up 300 s

**Why.** Entry 17: with a window of 60 s every turn of a person using the page woke the T4, a
minute's wait each. The author chose 300 s from three options.

| Window | The wait between turns | An idle window's cost | Bursts of use in the day's dollar |
|---|---|---:|---:|
| 60 s | about a minute, every turn | \$0.01 | about 50 |
| 180 s | none inside three minutes | \$0.03 | about 25 |
| **300 s** | none inside five minutes | \$0.05 | about 17 |

The costs are arithmetic from the T4's list rate, a wake of a minute included in a burst; none
is measured. The cap counts the window already (`Budget.charge`), so it stays true: a test holds
that the first turn of a burst is charged the window.

**A rule of the plan, set aside here and said so.** v2 §5.1 rule 3 is `scaledown_window=60`.
It is kept in every batch run. The demo is the one place where the next call comes from a person.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite,
852 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold). **Not run:** on Modal. The author's next `modal serve`: a second turn, sent after
listening to the first one's replies, should take seconds and not a minute.

## 19. The plan for the rest of the demo

The author, 2026-09-28: "Lets get everything related to demo part get done". What was left, and
what he decided:

| Commit | What | GPU |
|---|---|---|
| 9 | R3 and R4 live, each asked for by the visitor | A10, L40S |
| 10 | R5 on the visitor's recording: Moshi answers it as it answered the eval clips | A10 |
| 11 | measured panels for the rest: barge-in (PR-003), FT-1, severity, quantization | none |
| 12 | `modal deploy`, and what is owed before a public link | none |

| Decision | Chosen |
|---|---|
| Do the bigger GPUs wake on every turn? | no: each is asked for, and the page says what it wakes |
| One cap or one for each machine? | one, \$1 a day, each machine charged at its own rate |
| The window of the A10 and the L40S | 120 s, not the T4's 300: their idle time costs two and three times as much |
| R3's "upset?" score for a visitor's voice | not shown: an uncalibrated label on a real person. It stays in the measured panel |
| R5: a conversation, or on a recording | on a recording first: `duplex.py`'s tested protocol |
| Barge-in live | a measured panel in its place |
| PersonaPlex beside Moshi | not now: its own image, gated weights |

**Left out, and said so.** A conversation with R5, and barge-in while a reply plays, both need
the microphone open while the machine speaks: a streaming client in the browser, a WebSocket
through Modal, and headphones at the visitor's end. Nothing in the repo has tested any of it.
They can follow the list above. (In the hand-over the numbering ran 9 to 13, with R3 and R4 apart;
they are built as one commit, being one pattern.)

## 20. Commit 9: R3 and R4 live

**What was missing.** The live part had the three cascades. The rung that hears the audio (R3)
and the one that speaks the reply itself (R4) were on the measured tab only.

- `modal_app/demo.py`: two more classes, `Hears` (R3, an A10) and `Speaks` (R4, an L40S: 22.5 GB
  does not fit an A10), their images pinned as `r3_probe.py` and `r4_bench.py` pin theirs.
- `demo.py`: `Several`, the live part over more than one machine. The first answers every
  recording; the others only what is asked of them. They are asked at the same time, so the wait
  is the longest of theirs and not the sum. `Budget` charges each machine at its own rate, with
  its own window, under the one cap.
- `demoweb.py`: a recording with nothing asked for goes to the T4's four rungs, as before. Each
  machine is charged for its own time, and its wake is measured and kept.
- `demo_page.html`: "Also ask:" with a box for each; the box says which machine wakes, how long
  it took the last time, and what its idle window costs.

**A change in what a visitor sees when a machine fails.** Before, a T4 that did not answer was
a 503 and no cards. Now every machine that does not answer is a sentence on its rungs' cards,
and the rest answer, with the voice card above them. It is still charged: it was up.

**What a burst costs**, as arithmetic from list rates, a wake of a minute and one turn; none
measured:

| Asked for | Machines | About |
|---|---|---:|
| nothing | T4 | \$0.06 |
| R3 | T4, A10 | \$0.12 |
| R4 | T4, L40S | \$0.16 |
| both | all three | \$0.22 |

So the day's dollar is four bursts with everything asked for, or about seventeen with nothing.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 858 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the page in a browser, three made-up machines with stub rungs (wakes of 3 s and 5 s played) | "Also ask:" with R3 and R4, each saying its machine, "a minute or more", and what its idle window costs |
| a recording sent with nothing ticked | two cards, from the first machine only; 0.2 s; charged for the T4's window |
| the same recording with R3 and R4 ticked | four cards, R3's and R4's naming their machines; 5.9 s, the longest wake and not the sum of the two; "Sent 1 s ago" counting while it waited; both wakes said above the cards; the boxes still ticked, now saying "about 3 s the last time" |
| errors in the browser's console | none |

**Not run, and first run by the author:** the two image builds (16 GB and 22 GB of weights,
fetched on CPUs); R3 and R4 loading in their containers; their wakes, which may be longer than
the T4's; a request that waits past Modal's 150 s limit for a web request; the L40S's
availability.

## 21. The fifth session on Modal: six rungs answer one recording (at ab45907)

863 tests passed on the author's machine. Then `modal serve`. **The terminal's output of this
session was not sent to me**: what follows is read from two screenshots of the page, small
ones, so the times of the single stages are left out where I could not read them with
certainty. Not known: how long the two images took to build, how long the rungs took to load,
how long each request took end to end.

| What had not run before | What happened |
|---|---|
| the A10's and the L40S's images | built: both machines answered |
| R3 and R4 loading in their containers | both loaded |
| their wakes | the A10 37 s, the L40S 51 s; the T4 49 s in the same turn |
| a request past Modal's 150 s limit | not reached: the machines are asked together, and the longest wake was 51 s |
| the 300 s window (commit 8) | **held**: the second turn found all three machines up, and no wake was reported |

The author ticked R3 and R4 together in both turns. I had asked for one at a time, so that a
failure in one would not hide the other. Neither failed.

| | Turn 1, every machine woken | Turn 2, every machine up |
|---|---|---|
| the recording | 8.6 s | 6.3 s |
| median pitch, range, words a second | 118 Hz, 5.4 semitones, 1.6 | 135 Hz, 9.4 semitones, 1.7 |
| R0: first audio, whole reply | 839 ms, 839 ms | 1.34 s, 1.34 s |
| R1 | 305 ms, 625 ms | 276 ms, 1.34 s |
| R2 | 368 ms, 702 ms | 359 ms, 945 ms |
| R2 verbatim | 410 ms, 1.18 s | 339 ms, 1.13 s |
| R3, on the A10 | 1.13 s, 1.13 s | 2.51 s, 2.51 s |
| R4, on the L40S | 14.81 s, 14.81 s | 7.02 s, 7.02 s |
| charged by the app | \$0.2063 | \$0.0829 |
| the budget after it | \$0.70 | \$0.62 |

**The first warm turn whose cards were kept.** Turn 2's replies are longer than turn 1's, and
its times follow them: R0's 1.34 s with nothing to wake is slower than its 839 ms after a wake.
So again: a card is one turn. What holds in both turns is what holds by construction. R1's first
audio comes before its reply is whole and R0's cannot; R3 and R4 have no transcript; R4 is
slowest by far, because its wait is its own speech (9.17 s at the median of its 200 measured
turns).

**R3 answered about the speaker, not to him**, in turn 1: "The speaker is expressing frustration
about the time it's taking for something to happen or be observed." Two things in one sentence.
It described the recording instead of replying to it: in the measured run 50 of R3's 200
replies begin that way ("The speaker ...", counted from `runs/r3-gpu/evalset-modal-a10.json` for
this entry). And it named a feeling in a visitor, unasked: 5 of those 50 name one, by a list of
words of mine. The author's decision (entry 19) keeps R3's "upset?" score
off the live part; this is the rung's own reply, which that decision does not cover. Whether
the word came from the voice or from the words ("it's taking a long time") cannot be told from
one turn. For the author: whether the page should say, on R3's card, that its replies may
describe the speaker.

**The verbatim rung wrote a filler and a repetition the others did not**, in turn 1: "how it,
uh, how it behaves" where R0, R1 and R2 wrote "how it behaves". As in entry 9: what was said is
not known. Entry 11's count says the model adds no filler to read speech.

**The two recordings were again different sentences**, so the voice card's brackets compare
sentences. One sentence said two ways is still owed.

**Cost.** \$0.29 by the app's count: \$0.2063 for the turn that woke three machines, \$0.0829
for the one that kept them up. My arithmetic before the run said about \$0.22 for a burst with
both ticked: low by the idle time between the two turns, which the app counts on three machines
at once. The dashboard's figure was not sent. The day's dollar stood at \$0.62.

## 22. Commit 10: R5 on the visitor's recording

**What it is.** Moshi has no turn to answer, so it is not a rung. The visitor's recording is
the caller's side of one call, streamed through `duplex.converse` exactly as the 200 eval clips
were: 4.8 s of silence in which the model may open the call, the recording, 8 s of silence in
which to answer. The card shows what `duplex.measure` measures (the response gap, whether it
opened the call, how much of the caller it spoke over, the real-time factor) and plays the call,
the caller on the left and the model on the right.

**What it is not, and the card says so.** A conversation. The visitor cannot hear the model
while speaking, so nothing it says can change what the visitor says. A conversation needs the
microphone open while the model speaks (entry 19).

- `demo.py`: `Call`, a full-duplex model as the live part holds it; `any_rung`, which finds a
  rung in the registry and a duplex model in `r5.MODELS`.
- `modal_app/demo.py`: `Talks`, Moshi on an A10 of its own, the image pinned as `r5_bench.py`
  pins it. Its own, because Moshi's 16.6 GB and R3's 17.0 GB do not share 24 GB, and because
  Moshi needs a torch R3's libraries do not.
- `demo_page.html`: a third box under "Also ask:"; the card for a call.

**The caller's recording comes back in the reply**, as the left channel of the call. It goes to
the visitor's own browser and nowhere else; nothing is stored.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 860 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the page in a browser, a made-up duplex model on a made-up machine, the recorder fed a tone | "Also ask:" with R5, "on a second A10"; ticked and sent: R5's card under the first machine's, naming its machine |
| that card | "One call of 15.2 s: 4.8 s of silence in which the model may open the call (it did), your recording, then silence in which to answer", the sentence that it is not a conversation, and a player holding the call in stereo (1.9 MB in the reply) |
| its response gap | "It did not answer this caller": the tone fed to the recorder is not speech to the voice detector, so no caller was found to answer. The gap itself is held by a test, on a voice |
| errors in the browser's console | none |

**Not run, and first run by the author:** Moshi itself in the demo: the image build, the load,
a call on a real recording, and how long a call of up to 28 s of stream takes (the eval run's
real-time factor says less than its length).

## 23. The sixth session on Modal: Moshi answers a call, slowly (at 7650d36)

865 tests passed on the author's machine, and `7650d36` was committed. The record of the session
came a message late and without the terminal's output: three screenshots of the page. (I had
first written this entry as "nothing reached me"; that was true for one message.) Not known: how
long Moshi's image took to build and the model to load, how long the request took end to end.

One recording of 12.1 s, every box ticked, every machine asleep:

| Machine | Its wake | Rung | First audio | Whole reply |
|---|---:|---|---:|---:|
| T4 | 52 s | R0 | 966 ms | 966 ms |
| | | R1 | 548 ms | 814 ms |
| | | R2 | 527 ms | 1.21 s |
| | | R2 verbatim | 585 ms | 1.03 s |
| A10 | 33 s | R3 | 1.36 s | 1.36 s |
| L40S | 50 s | R4 | 5.99 s | 5.99 s |
| a second A10 | 82 s | R5, Moshi | not answered | a call of 25.0 s, in 41.8 s |

Charged by the app: \$0.2754. The budget after it: \$0.34 of the day's dollar. The voice card:
108 Hz, 4.9 semitones, 2.1 words a second.

**Moshi ran in the demo**: its image built, it loaded, and it took the recording as one side
of a call. Commit 10 is confirmed that far. Three things about the call:

- **It ran at 1.67 of real time: 41.8 s for 25.0 s of stream.** Slower than the speech it
  handles, which a full-duplex model cannot be. The cause is in the committed eval run: its first
  clip ran at 1.677 and the 199 after it at a median of 0.69. The first call after loading pays
  for start-up. Mine to have seen: the number was in `runs/r5-moshi/evalset-modal-a10.json` when I
  wrote `Call.load()`, and the rungs take a throwaway turn at load for this very reason. Fixed in
  entry 24.
- **It did not answer.** It opened the call, spoke over none of the recording, and said nothing
  in the 8 s after it. In the eval run it answered 96 of 200 callers: not answering is what Moshi
  does about half the time, and the card says it as it is.
- **The wait came near Modal's limit.** Moshi's wake, 82 s, and its call, 41.8 s, are 124 s of
  one request, and a web request may take 150 s. A recording of 15 s makes a call of 28 s.
  With the throwaway call moved into the wake the call itself takes about 17 s, but the wake is
  longer by as much: the sum is not shorter. Not fixed. What would fix it is answering the page
  first and the slow machines after, which is a change to how the page asks; in the backlog.

**The two tiers heard different words, and this time one of them is plainly right.** The author
said the rungs' names. R0 and R1 (`base.en`) wrote "are three are four and are five"; R2
(`small.en`) wrote "R3, R4 and R5"; the verbatim rung wrote "R three, R four and R five, all the
rungs, uh, testing". R2's point, on the project's own vocabulary. R3, which has no transcript,
replied "We're recording and testing R3, R4, and R5": it heard them too.

**R3 restated the recording this time**, and named no feeling. The session ran before the note
on its card (entry 24).

**Cost.** \$0.28 by the app's count, one turn on four machines. The day's spend by the app's
count: \$0.66.

## 24. Commit 11: what else was claimed, and what happened

**What was missing.** The measured tab showed the six rungs, FT-2's recognizer and the prosody
probe. Five more registrations had run and were in `runs/` only: barge-in, FT-1, the two severity
renders, the quantization ladder. Four of the five are fine-tunes or methods that did not do
what was claimed for them, which is what the author asked the demo to show as such (entry 1).

| Panel | From | As it came out |
|---|---|---|
| When the caller breaks in, PR-003 | `runs/bargein/pr003-70ms.json` | supported: wrong on 1 of 20, against 8 for the best duration threshold |
| A LoRA on a full-duplex model, PR-010 | `runs/ft1/pr010.json` | not supported: 496 ms from the human gaps became 1035 ms |
| A severity score, put onto speech, PR-009 | `runs/severity/pr009.json` | not supported: four criteria of five; pitch offset 88.0% against 90% |
| The same targets, as a synthesizer's own inputs, PR-012 | `runs/ft3/pr012.json` | not supported: all five failed |
| A synthesizer made smaller, PR-013 | `runs/m26/pr013.json` | Claim A supported: 3.96 Hz lower, words unchanged; Claim B not |

- `demo.py`: `findings()`. Every number is read from the run file's own criteria, or counted from
  its outcomes; nothing is typed in. A test holds each against the registered result.
- **A fault found by checking, before the hand-over.** I had written one criterion's bar into a
  label from memory, "at least 24 of 32"; the registered bar is 28. It was a label, not a verdict,
  and it was wrong. The bars are now read from `severity.py`, where the criteria are.
- **No audio.** The barge-in set is the author's own voice; the renders are synthetic speech
  kept out of git. The panels are numbers and sentences.
- The severity panels carry the project's boundary: synthetic speech, no clinical validity
  claimed or implied.
- The five run files go to the web container with the rest; none holds a local path (checked).

**A throwaway call when Moshi loads** (`Call.load`): one second of silence and four after it,
through `duplex.converse`, so that the first call a visitor makes runs at the model's speed and
the card's real-time factor is the model's. It lengthens the wake by what it takes.

**The note on R3's card**, the author's decision after entry 21: "Its reply may describe you,
how you sound included: it is not asked to, and nothing here checks what it says."

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 862 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the measured tab in a browser, from the committed run files | five panels under "What else was claimed, and what happened", each with its registration and "supported" or "not supported" (two and three), its rows, its note and its file |
| the numbers on them | as registered: 1 of 20 against 8; 496 ms and 1035 ms; 169 of 192 (88.0%): failed; five failed; -3.96 Hz, 10 errors against 8, 42 of 48 |
| audio in the panels | none |
| errors in the browser's console | none |
| the throwaway call at load | on a made-up model, by a test: one call of 63 frames before the visitor's |

**Not run:** on Modal. The panels need no GPU: the CPU container serves them. The throwaway
call has run on a made-up model only: whether one call is enough to bring Moshi to 0.7 is for
the next session to show.

## 25. What is left, and two questions of the author's

**"Isnt moshi finetuned for emotions for psychiatric patient simulation"**: no, and the repo says
why. FT-1 is a LoRA on Moshi trained on DailyTalk's dialogues for turn-taking; the plan chose that
target in place of a patient model (v2 §7.1, quoted in PR-010's registration), because a patient
model needs clinical data that does not exist in a form that can be shared. What the project has
that simulates psychiatric states in speech is the severity render: a PHQ-9 or YMRS score becomes
targets for pitch, range, rate and pauses, rendered onto neutral Kokoro speech in Praat (PR-009,
four criteria of five) and given to FastPitch as its own inputs (PR-012, none of five).

**The portfolio.** The author's site is a GitHub Pages repository with one `index.html`. Three
ways to join it to the demo were put to him: a link, a frame, or the page itself in the
portfolio calling Modal. Chosen: a link.

| Commit | What |
|---|---|
| 12 | answers arrive as they are ready: one request a machine |
| 13 | a severity render that can be heard: a sentence, a condition and a score; neutral and rendered speech; targets against what was measured |
| 14 | `modal deploy`, the link from the portfolio, and what is owed before it |

| Decision | Chosen |
|---|---|
| The severity render's text | a fixed list of neutral sentences, not free text: free text would let anyone make the demo say anything in a "patient" voice |
| Its scores | the six registered steps of each condition, not a slider: every render is a measured case |
| Its label | on the card: synthetic speech edited to a specification; no clinical validity claimed or implied; not a diagnosis |
| PR-009's result | beside the player, the miss included |
| Extreme scores | a warning: at the top of the mania scale the mapping asks for +45 Hz at 1.8 times the speed, which sounds processed |
| The portfolio | a link |
| The measured tab | open without a code |

Left out: FastPitch's voice as a player (it failed every criterion; it stays a measured panel), and
Moshi as a patient (nothing here supports it).

## 26. Commit 12: one request a machine

**What was wrong.** One request carried a recording to every machine and came back when the
slowest had answered. In the sixth session that was Moshi: 82 s to wake and 41.8 s for the call,
124 s of a request that may take 150 (entry 23). And a visitor saw nothing for two minutes,
though the T4's four rungs had answered in the first.

**Now.** The page asks the door once (`POST /api/enter`), then sends the recording to each
machine in a request of its own, all at once. Each machine's cards say "Sent N s ago" until it
answers, and then hold its answer. No request waits for another machine.

- `demoweb.py`: `/api/enter`, the door alone; `voice_too`, so the voice is measured once a
  recording and not once a machine; a lock for each machine in place of one for all.
- `demo_page.html`: the cards as places that fill; the line above them adds up what has been
  charged and which machines were woken, as the answers come.
- **Why the door is asked first.** Every request checks the code. Four requests with a wrong
  code would be four wrong codes, and five close the live part (entry 12): one mistyped code
  would all but lock the visitor out. Asked once, it is one.
- The budget is as it was: each request is charged for its machine.

**What it does not change.** A machine's own wait. Moshi asleep is still two minutes; the page
now shows the rest meanwhile.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 864 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the page in a browser: three made-up machines with stub rungs, their wakes played as 2 s, 6 s and 4 s, the third made to fail | — |
| a wrong code, then Send | "the live part needs an access code", no card, one request to the door and none to a machine |
| the right code, R3 and R4 ticked, 1.2 s after Send | four cards waiting, each counting "Sent 1 s ago" |
| at 3.4 s | the voice card and the first machine's two cards filled; R3 and R4 still waiting; the line above them: charged so far, and the first machine's wake |
| at 5.2 s | R4's card: "The machine this rung runs on did not answer."; the rest as they were; the charge grown by that machine's |
| at 7.8 s | R3's card filled, naming its machine; every card settled |
| errors in the browser's console | one: the 401 of the wrong code, as meant |

**Not run:** on Modal, where the requests go through Modal's own web layer at the same time.

## 27. Commit 13: the severity render, to be heard

**What was asked.** The author, entry 25: a demonstration of "TTS simulation of different
psychiatric states". The project has one: PR-009's render. Until now it could be read about on
the measured tab and not heard: its renders are kept out of git.

**What it is.** On the live tab, behind the door: a condition (depression on PHQ-9, mania on
YMRS), one of the six scores the registration swept, one of its two voices, one of its eight
sentences. Kokoro says the sentence; Praat edits pitch, range, rate and pauses to the score's
targets; the four are measured again from the render. The card plays both and sets three
columns side by side: what was asked for, what this render measured, what the registered run
measured for the same voice, sentence and score.

| Decision, as approved | Built |
|---|---|
| a fixed list of sentences, not free text | `render_severity` refuses anything outside `render_choices()`; a test sends it free text, a dev sentence, a score between the steps, another condition and another voice |
| the six registered scores, not a slider | the page's list is `severity.SCORES`; the server checks again |
| the label on the card | "Synthetic speech, edited to a specification. No clinical validity is claimed or implied, and this is not a diagnosis of anyone" |
| PR-009's result beside the player | read from the run file: not supported, four criteria of five, pitch in 169 of 192 checks |
| a warning on extreme scores | shown where the target passes 30% of rate or 40 Hz: depression from 20, mania from 26 |

- **The files say what they are.** Each WAV carries "synthetic" in its title and the project's
  note in its comment, as every file `severity.py` writes does: a render saved from the page
  stays labelled.
- **Where it runs.** On the T4, where Kokoro is; the edit is Praat's, on that machine's CPU.
  The T4's image gains PR-009's two voices. Charged to the budget as a turn is.
- **This render and the registered one differ**, and the card says why: Kokoro does not say a
  sentence the same way twice, so they start from different neutral speech.

**What it is not.** A model of a patient, and not evidence that depressed or manic speech
sounds like this. The mapping is one page's hand-written anchors (`severity.py`'s own words),
and PR-009 tested only whether the render does what the mapping asks.

**Run before the hand-over**, in a scratch copy of the tree on the laptop:

| What | Result |
|---|---|
| the full suite | 869 passed, 5 skipped, the skipped ones for audio the scratch copy does not hold |
| the render with the real Kokoro on the laptop's CPU, `am_adam`, sentence t3, mania 26 | asked for +19.5 Hz, range 1.35, rate 1.35, pauses 0.70; measured +20.5, 1.31, 1.34, 0.68; every target met. The registered run for the same case: +20.5, 1.23, 1.34, 0.68 |
| the page in a browser, the real Kokoro behind it: depression, PHQ-9 20, `af_sarah`, sentence t1 | the lists hold the two conditions, the six scores, the two voices, the eight sentences; the render came back in 8.0 s, Kokoro's loading included: neutral 4.3 s, rendered 8.0 s, two players, the three columns (pitch -19.82 asked, -19.55 here, -20.64 registered), "Every target met", the warning, PR-009's result, the charge |
| errors in the browser's console | none |
| the renders, by ear | **not listened to**: I cannot hear. How they sound is for the author |

**Not run:** on Modal: the T4's image with the two voices, Kokoro on the GPU speaking them,
the render through the door there.

## 28. The seventh session on Modal: the render, and nothing else (at fb601a5)

874 tests passed on the author's machine. Then `modal serve`. Sent to me: one screenshot of the
live tab. The terminal's output came a day later, ahead of the eighth session's (entry 30); I had
written here that it was not sent, which was true until then. It shows the T4's image rebuilt for
PR-009's voices (its two new layers in 21.1 s and 2.3 s), two requests to render refused with 401
before one was let in (two wrong codes, counted by the door), the four rungs and Kokoro loaded in
42.9 s, and the render's request 57.3 s end to end, nearly all of it the wake.

**What ran: one severity render.** Depression, PHQ-9 10, `af_sarah`, sentence t1, on the T4.

| | Asked for | Measured in this render | Measured in the registered run |
|---|---:|---:|---:|
| pitch, Hz | -12.41 | -9.69 | -10.52 |
| pitch range, times | 0.78 | 0.78 | 0.78 |
| rate, times | 0.81 | 0.81 | 0.81 |
| pauses, times | 2.11 | 2.08 | 2.08 |

Neutral 4.3 s, rendered 5.9 s. "Every target met in this render, within its tolerance": the
pitch is 2.72 Hz from its target, inside the 3 Hz allowed, and further from it than the
registered run's. Charged \$0.0584. No warning, as none is due at this score.

So commit 13 is confirmed on Modal: the T4's image with PR-009's voices built, Kokoro spoke
`af_sarah` on the GPU, Praat edited it, the door let the request in, the run file was read
beside it.

**What did not run, and why.** No recording was sent: the answers' place is empty, no box is
ticked, and the wakes the boxes quote are the sixth session's. The budget stood at \$0.34 before
the session and \$0.29 after it; a turn with every box ticked costs about that much. The author
wrote "live turns done"; the screenshot shows a render done. Still unconfirmed on Modal:

| Commit | What is unconfirmed |
|---|---|
| 11 | the five measured panels served from the CPU container; the note on R3's card; Moshi's speed after its throwaway call |
| 12 | the cards filling as each machine answers, through Modal's own web layer |

**How the render sounds**: "Render sounded well", the author, asked afterwards. One render,
depression at PHQ-9 10, one listener, the person who asked for it: not a listening test. I
cannot hear it. No one has yet listened at the scores where the card warns that the edit may
sound processed.

**Cost.** \$0.06 by the app's count: the T4 woken for one render, and its window of 300 s.

## 29. Commit 14: before the repo is public

The author's decisions, 2026-09-28, on what was owed (`docs/backlog.md`):

| Item | Chosen |
|---|---|
| One more session on Modal before deploying | yes, with a day's budget unspent: commits 11 and 12 are unconfirmed there |
| `modal deploy` | after that session; the author runs it |
| The portfolio | a link in its `index.html`; the author commits and pushes it |
| A licence for the code | MIT |
| A remote | a public repository, pushed by the author |
| The local paths | out of the files, in a new commit; the history left as it is |
| `probes/README.md`, the recording notes | the author's to write |

**In this commit:**

- `LICENSE`: MIT. The copyright line names the author's public handle, not his name: his to
  change. Models and datasets keep their own licences, as the README says.
- **The local path, out of nine files**: 17 places in six reports under `runs/` and 5 in the
  comments of three source files, each now `<repo>`. Replaced as bytes, so nothing else in a
  file moves. `runs/README.md` says it was done, as the one exception to "never overwritten":
  a path is not a result, and the reports as printed are in the history.

**What stays, and the author should know before he pushes:**

| What | Where | Why it stays |
|---|---|---|
| the local path | the history: every commit before this one | to take it out is to rewrite every commit, and the log, the ledger, the registrations and the results pages cite commits by hash |
| the author's name and e-mail address | every commit's author line | it is what git records; to change it is the same rewrite |
| the Modal workspace's name | the first line of every Modal report | it is the author's public handle, and the demo's address holds it too |

**Not done here, and the author's:** `modal deploy`; the push; the link in the portfolio; the
recording notes.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite,
869 passed, 5 skipped (the skipped ones for audio the scratch copy does not hold); a search of every tracked file for the local path: none left.

## 30. The eighth session on Modal: two renders (at fb601a5)

The day's budget had reset at midnight UTC. Sent to me: the terminal's output, with the seventh
session's ahead of it (entry 28), and one screenshot of the live tab.

| | Took |
|---|---:|
| the four rungs and Kokoro, loaded on the T4 | 40.4 s |
| the first render, the T4 asleep: its request end to end | 50.5 s |
| the second render, the T4 up | 1.77 s |

The screenshot is of the second render, with the seventh session's settings: depression, PHQ-9
10, `af_sarah`, sentence t1. The first render's numbers are not in it.

| | Asked for | This render | The seventh session's | The registered run |
|---|---:|---:|---:|---:|
| pitch, Hz | -12.41 | -10.84 | -9.69 | -10.52 |
| pitch range, times | 0.78 | 0.78 | 0.78 | 0.78 |
| rate, times | 0.81 | 0.81 | 0.81 | 0.81 |
| pauses, times | 2.11 | 2.07 | 2.08 | 2.08 |

"Every target met in this render." The same settings twice, and the pitch 1.57 Hz from its
target this time, 2.72 Hz the time before: what the card says, that Kokoro does not say a
sentence the same way twice, so each render starts from different neutral speech. With the T4
up a render takes under 2 s; a cold one's 50 s is the wake.

**Cost.** \$0.06 by the app's count: \$0.94 of the day's dollar left after it, the second render
\$0.0027 of it.

## 31. The ninth session on Modal: every box ticked, twice (at ff405b4)

874 tests passed on the author's machine, and `ff405b4` was committed. Then `modal serve`. Sent
to me: the terminal's output and four screenshots of the live tab. Two recordings, R3, R4 and R5
ticked for both. While it was served I read `/api/status` at its address, which wakes no machine.

**Turn 1, every machine asleep.** The recording was about 7.3 s: the page's line with its length
is not in the screenshots, and this is Moshi's call, 20.2 s, less the 4.8 s before it and the
8.1 s after.

| Machine | Its wake, the app's count | Its model loaded in, the terminal | Its request, end to end |
|---|---:|---:|---:|
| T4: R0, R1, R2, R2 verbatim | 49.6 s | 41.8 s | 55.2 s |
| A10: R3 | 31.8 s | 24.4 s | 36.6 s |
| L40S: R4 | 66.5 s | 27.9 s | 77.6 s |
| a second A10: R5 | 50.5 s | no line reached the terminal | 67.3 s |

The terminal does not say which request was whose. Each is its machine's wake and replies plus
2.6 to 3.0 s, and this pairing is the only one that fits. Most of the L40S's wake came before
its model began to load: 66.5 s, of which 27.9 s loading.

**Turn 2, 8.82 s, every machine up.** The page named no wake. The four requests came back at
4.38, 7.89, 18.6 and 20.4 s. The T4's is the first: its four replies add up to 4.07 s. R3's is
the second, its card 589 ms; R4's and R5's are the last two, their cards 14.6 s each. So R3,
R4 and R5 each took 4 to 7.3 s more than the times on their cards, and the T4 0.3 s. The cards
time the models; where the rest went, I do not know. Once; not looked into.

| Rung | Turn 1: first audio | Whole reply | Turn 2: first audio | Whole reply |
|---|---:|---:|---:|---:|
| R0 | 806 ms | 806 ms | 573 ms | 573 ms |
| R1 | 441 ms | 716 ms | 516 ms | 516 ms |
| R2 | 461 ms | 750 ms | 966 ms | 2.38 s |
| R2 verbatim | 634 ms | 634 ms | 599 ms | 599 ms |
| R3 | 2.07 s | 2.07 s | 589 ms | 589 ms |
| R4 | 8.53 s | 8.53 s | 14.60 s | 14.60 s |
| R5, Moshi | not answered | a call of 20.2 s in 13.76 s | a gap of 100 ms | a call of 21.7 s in 14.59 s |

**What this confirms:**

| Commit | Confirmed on Modal |
|---|---|
| 11 | the five measured panels, read from the page at the session's address earlier on 09-29: all five, their tables, no error in the console; R3's note on its card; **Moshi at 0.68 and 0.67 of real time**, turn 1's the first call after its throwaway one (the sixth session's first call, with no throwaway before it: 1.67; the eval run's median: 0.69) |
| 12 | four requests a turn, one a machine, each back when its machine was done: 36.6, 55.2, 67.3 and 77.6 s in turn 1, where one after another they would have taken 237 s. The longest is about half the 150 s a web request may take; Moshi's wake and call, 124 s in one request in the sixth session, are 67.3 s here |

Nothing of the demo is left unconfirmed on Modal.

**Moshi answered one call of two.** In turn 1 it opened the call and said nothing after the
recording; in turn 2 it answered 100 ms after it. In the eval run it answered 96 of 200.

**The words the cascades heard decided their replies.** Turn 2:

| Rung | Heard |
|---|---|
| R0, R1 (`base.en`) | "Let's test on taking I won't help with GPS. Can you share where you are located?" |
| R2 (`small.en`) | "Let's test turn taking I won't help with GPS. Can you share where you are located?" |
| R2 verbatim | "Let's test the untaking. Uh, I want help with G P S. Can you share where you're located?" |

The four cascades share one language model, Qwen2.5-0.5B-Instruct, greedy, with one system
prompt: given the same words it gives the same reply. In turn 1 R0, R1 and R2 heard "How do you
want to serve?" and replied "I'm here to help!"; the verbatim rung heard "Uh, ... How do you
wanna serve?" and replied "I'm sorry, but I can't assist with that request." In turn 2 the same
refusal came to R0, R1 and the verbatim rung, and R2's words got a long answer. Four of the
eight cascade replies were that refusal, to harmless questions. It is the model the ladder
measured, and the demo shows it as it is: a larger model would make the live rungs differ from
the measured ones. Not changed.

**Two models named a city.** Asked where it is, R3 said "I am currently located in New York
City" and Moshi "Sure, I'm currently in the city of Paris"; R4 said it did not know. Nothing in
the demo gives a model a place: both cities are made up, about the model itself.

**The voice card**, both turns: 136 Hz, 7.6 semitones, 2.7 words a second, then 122 Hz
(-1.8 semitones), 6.4 semitones (-1.2), 1.8 words a second (-0.9). Two different sentences
again: one sentence said two ways is still owed.

**In the terminal, and not failures.** "Unclosed connection: Channel('input-plane.us-east.modal.com', ...)"
three times from the web container, each just before a request came back 200: printed by
Modal's client; not looked into. Warnings from the libraries as they loaded (Moshi's loader,
Qwen2.5-Omni's config) and CUDA's banner. No request failed; Modal's async warning did not come
back.

**Cost.** By the app's count the day's spend after the session was \$0.505 (\$0.495 left, read
from `/api/status`). Less the eighth session's \$0.06: \$0.445 for this one, turn 1 about
\$0.27 with four machines woken, turn 2 \$0.1785 (the page's line). Turn 2 woke nothing and
cost two thirds of turn 1: each machine's window runs from its last answer, so the minutes
between the turns were paid on four machines at once, \$4.74 an hour together. Modal's own
bill is on its dashboard.

**Left of the demo, all the author's:** `modal deploy`; the push; the link in the portfolio; one
sentence said two ways; a listen at the scores that carry a warning.

## 32. Deployed (at eb1974a)

874 tests passed on the author's machine, and `eb1974a` was committed. Then `modal deploy
modal_app/demo.py`, by the author: "App deployed in 2.823s". The address:

https://ramcharanmc--s2s-lab-demo-web.modal.run

It is `modal serve`'s address without `-dev`, and it stays up without the author's terminal.

**Checked there, the same day, from a browser, with no GPU woken:** the page and the measured
part (the seven rung rows, the five panels); `/api/status` (the code set, \$0.495 of the day's
dollar left, nothing kept); one request to the door with no code, refused with 401 and not
counted as a wrong code, since a request with no code is not a guess. The author's screenshot
shows the live tab there, the three boxes quoting the ninth session's wakes (32, 67 and 51 s).

**What it costs when nobody visits: nothing.** Every container scales to zero: the page's and the
T4 300 s after their last request, the other three 120 s after theirs. A visit wakes the page's
CPU container only. A GPU wakes only behind the door, and the app refuses a turn once \$1 of a
UTC day is spent, by its own count. Not under that cap: the CPU container, woken by any visitor
to the page, at CPU rates. To take the demo down: `modal app stop s2s-lab-demo`.

**The portfolio's link, drafted, not committed**: the portfolio is the author's repository. A card
in the style of its "Key Projects" cards (`project-card`, `project-tag`, `btn btn-secondary`),
saying that the live tab needs a code, since a visitor who is not told would find a locked door:

```html
<div class="project-card">
    <h3>s2s-lab: Speech-to-Speech, Measured</h3>
    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
        Six speech-to-speech architectures, from a naive ASR&rarr;LLM&rarr;TTS cascade to full-duplex
        Moshi, measured on one frozen benchmark, every claim registered before its test. The live
        demo shows the measured turns to anyone, and answers your own recording on up to four GPUs
        (access code on request: see Contact).
    </p>
    <div style="margin-bottom: 1rem;">
        <span class="project-tag">Whisper</span>
        <span class="project-tag">Qwen2-Audio</span>
        <span class="project-tag">Qwen2.5-Omni</span>
        <span class="project-tag">Moshi</span>
        <span class="project-tag">Modal</span>
    </div>
    <a href="https://ramcharanmc--s2s-lab-demo-web.modal.run" target="_blank" rel="noopener" class="btn btn-secondary">&#9654; Live demo</a>
</div>
```

A link to the code goes beside it once the repository is public.

**Left, the author's:** the push to a public remote; the link in the portfolio; one sentence said
two ways; a listen at the scores that carry a warning.
