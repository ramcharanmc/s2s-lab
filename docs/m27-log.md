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
| 17 | 09-28 | `modal serve`, by the author, his own voice | the voice card and the prosody panel on Modal; two live turns, and **both woke the T4**: it had gone to sleep in between | — | — |

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

**Still not said by the author:** whether the recorder now makes clear when it starts and stops.
