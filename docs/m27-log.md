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
| 12 | 09-28 | the tests, laptop, scratch | wrong codes are counted and close the door; a code under ten characters is no code | `demo.py`, `demoweb.py`, `demo_page.html`, tests | this commit |

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
