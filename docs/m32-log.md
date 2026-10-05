# M32 run log: R4 streamed, in the demo (M32a), then sooner (M32b)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m31-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design (M32a): R4 and R4 streamed in the demo on one set of weights; the stream heard as it arrives; the author's go, with real streaming to the browser, the door at 30 wrong codes in 5 minutes and the day's cap at \$3 | — | — |
| 2 | 10-04 | the demo's tests, laptop | the pair on one model; both R4 cards from one request to the L40S; PR-014 and PR-015 read from their run files | `r4stream.pair`, `demo` (measured row, two panels, door), `modal_app/demo.py` (the L40S's two rungs, the cap), the page | this commit |

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
