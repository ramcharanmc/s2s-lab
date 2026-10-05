# M32 run log: R4 streamed, in the demo (M32a), then sooner (M32b)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m31-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design (M32a): R4 and R4 streamed in the demo on one set of weights; the stream heard as it arrives; the author's go, with real streaming to the browser, the door at 30 wrong codes in 5 minutes and the day's cap at \$3 | — | — |
| 2 | 10-04 | the demo's tests, laptop | the pair on one model; both R4 cards from one request to the L40S; PR-014 and PR-015 read from their run files | `r4stream.pair`, `demo` (measured row, two panels, door), `modal_app/demo.py` (the L40S's two rungs, the cap), the page | 39f0c6a |
| 3 | 10-04 | the demo's tests, laptop; the page on localhost with fake rungs, in a browser | the stream reached the browser a piece at a time (an HTTP client saw the pieces at 0.96, 1.76 and 2.56 s, as the fake rung made them); the page played them as they came and timed the first at 940 ms after Send, the server's own 801 | `demo` (`stream_turn`, `Live.stream`, `Several.stream`), `demoweb` (`/api/stream`), `modal_app/demo.py` (`Speaks.stream`), the page (it plays the pieces as they arrive), tests | this commit |

## 1. The design (M32a)

**What the demo had.** The L40S offered R4 alone (`r4-gpu`, eager), woken on request; a live turn came
back as one answer a machine, each card showing "first audio · whole reply". The measured part's
panels stopped at PR-013.

**Two R4s do not fit one card.** The demo loads a model per rung: R4 and R4 streamed would be two sets
of 22.5 GB on the L40S's 48. And plain R4 cannot share its weights with the stream: the stream's
compiled Talker layers would take over R4's eager decoding. So the L40S offers:

| card | rung | what it is |
|---|---|---|
| R4 | `r4-gpu-fast` | the reply decoded whole; on the eval set 1.03 of R4's time, the same text (PR-015) |
| R4 streamed | `r4-gpu-stream` | the same Talker and codes, decoded a block at a time, its first half second held |

on one set of weights (`r4stream.pair`). The codes are the same, so a visitor hears the same reply in
the same voice twice, one of them sooner.

**Commit 1:** the pair; an "R4 streamed" row in the measured part (from `runs/r4-gpu/pr015.json`) and
two panels, PR-014 (reported as *not run*) and PR-015; the door at **thirty wrong codes in five
minutes, closed for five** (it was five in ten, for ten: thirty tries still find no code of ten
characters, and strangers guessing rarely close it on the people given the code); the day's cap at
**\$3** (it was \$1), the author's choice.

**Commit 2:** the stream heard as it arrives: the L40S yields each block as it is made, the web
container passes it on as it comes, the page plays it at once and times the first sound in the
browser, network included. Until then the page plays the streamed reply when all of it has arrived,
and the card says when its first sound existed on the server.

**Then** a `modal serve` session (about \$0.50): the wake with the compiles (about two minutes cold,
against about one), both R4 cards on one recording, the door, the measured part. Then `modal deploy`,
the author's.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 968 passed, 2 skipped.

## 2. Commit 2: the stream heard as it arrives

**The path.** `Speaks.stream` on the L40S is a generator: `demo.Live.stream` yields `{"cold"}`, then for
R4 streamed `{"start", "sr"}`, a `{"piece", "made_ms"}` for each block the rung emits (16-bit PCM,
base64), and `{"done"}` with the turn's record; R4, decoded whole, comes after it as one `{"done"}` with
its WAV. The rung runs on a worker thread and its `emit` is wrapped for the turn, so each block leaves
the moment it exists. The web container calls it with Modal's `remote_gen` and passes each event on as
a line of `/api/stream` (newline-delimited JSON), then a last line with what was charged.

**The rules are /api/live's.** The door first; one machine a request, and only one with a streaming
rung on it; the machine's lock held until the stream ends; the budget charged in a `finally`, so a
visitor who leaves mid-reply still pays for the machine that was up; a wake time recorded when it was
cold. A machine that fails says so in a line, without its error.

**The page.** One request a machine (the L40S's two cards in one, the streamed one first). Each piece
is scheduled on Web Audio right after the one before; the AudioContext is made on the click, as
browsers allow sound only after a gesture. The first piece's time is taken in the browser, from Send to
when it plays, so the network and any wake are in it. When the turn is done the card shows the server's
numbers, the browser's first-audio time and a player with the pieces joined.

**Checked on the laptop**, the real page and web app with fake rungs (a tone in three blocks, 0.8 s
apart): an HTTP client read the cold line at 0.15 s and the pieces at 0.96, 1.76 and 2.56 s, so nothing
held them back; in a browser both L40S cards came from one request, the streamed card read "First audio
801 ms · whole reply 2.40 s" (server) and "first audio after 940 ms, counted from Send" (browser), and
R4's card answered whole with its WAV.

**Not checked until Modal:** `remote_gen` through a real container, the compiles in the wake, and the
browser's time against the server's on a real network.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 978 passed, 2 skipped.
