# M32 run log: R4 streamed, in the demo (M32a), then sooner (M32b)

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m31-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-04 | — | the design (M32a): R4 and R4 streamed in the demo on one set of weights; the stream heard as it arrives; the author's go, with real streaming to the browser, the door at 30 wrong codes in 5 minutes and the day's cap at \$3 | — | — |
| 2 | 10-04 | the demo's tests, laptop | the pair on one model; both R4 cards from one request to the L40S; PR-014 and PR-015 read from their run files | `r4stream.pair`, `demo` (measured row, two panels, door), `modal_app/demo.py` (the L40S's two rungs, the cap), the page | 39f0c6a |
| 3 | 10-04 | the demo's tests, laptop; the page on localhost with fake rungs, in a browser | the stream reached the browser a piece at a time (an HTTP client saw the pieces at 0.96, 1.76 and 2.56 s, as the fake rung made them); the page played them as they came and timed the first at 940 ms after Send, the server's own 801 | `demo` (`stream_turn`, `Live.stream`, `Several.stream`), `demoweb` (`/api/stream`), `modal_app/demo.py` (`Speaks.stream`), the page (it plays the pieces as they arrive), tests | ec2713b |
| 4 | 10-05 | `modal serve modal_app/demo.py`, the author, T4 and L40S; a cold turn (R4 streamed) and a warm one (R4 and R4 streamed) | the stream through Modal: `Speaks.stream` and `remote_gen` worked; the L40S loaded both rungs in 152.7 s with the compiles; R4 streamed's first audio 3.14 s (cold turn) and 3.06 s (warm; R4 3.67 s); but the browser's first audio 164.59 s and 8.91 s after Send, about when the request ended; the door held (two wrong codes 401) | — | — |
| 5 | 10-06 | a probe outside the repo (an HTTP client reading `/api/stream` line by line, with and without gzip), the author | the lines left Modal as they were made: on the warm run the blocks made at 3.06, 4.73 and 6.15 s reached the client at 4.30, 5.83 and 7.25 s; no compression either way | — | — |
| 6 | 10-06 | a second probe: HTTP/2 alone, HTTP/1.1 beside a T4 request, HTTP/2 beside a T4 request (the page's set-up), the author | every run streamed; beside the T4's request the stream's first lines came at 4.17 s instead of 1.1, its second block with the T4's answer | — | — |
| 7 | 10-12 | the page in the author's browser, its network panel open; a cold turn and a warm one, R4 and R4 streamed | the warm turn: the stream request 13.34 s; R4 streamed's first audio 3.41 s on the L40S, 5.83 s after Send in the browser; R4 5.81 s: the browser gets the blocks as they come | — | — |
| 8 | 10-12 | the voice measurement timed on the laptop beside a thread that ticks every 5 ms | `voice()` on a 6 s recording took 0.23 s and held every other thread up to 101 ms at a time; on Modal's default share of a core, seconds | the page's container a full core (`cpu=1.0`) | f5d22a5 |
| 9 | 10-12 | the second probe again, the author, after the fix | beside the T4's request the stream's first lines at 1.53 s (4.17 before), as with nothing beside it; each block reached the client 0.6 to 0.8 s after it was made in every run, the T4 beside it or not | — | — |
| 10 | 10-12 | `modal deploy modal_app/demo.py`, the author; the deployed page's status and measured data read from the laptop | the public demo offers R4 and R4 streamed on the L40S, the streamed one marked to play as it arrives; the measured part has the R4 streamed row and the PR-014 and PR-015 panels; the cap \$3. M32a done | — | this commit |

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

## 3. On Modal (commit 2, then commit 3)

**The first session (10-05, row 4).** Everything ran: the door, the T4's wake (41.5 s of loading), the
L40S's (159 s, 152.7 of it loading, against 67 s in M27, before the compiles), R4 streamed beside R4 on one set of
weights, the same reply from both. But the browser's first audio came about when the request ended:
164.59 s after Send on the cold turn (the request 164.3 s), 8.91 s on the warm one (R4 streamed 4.48 s
and R4 3.67 s, one after the other). It looked as if the stream were held somewhere between the L40S
and the page.

**Two probes (10-06, rows 5 and 6)** read `/api/stream` from the laptop line by line. Every line left
Modal as it was made: neither `remote_gen`, nor Modal's proxy, nor compression, nor HTTP/2 held it. One
thing did slow it: with the T4's request running beside it, as the page sends it, the stream's first
lines came at 4.17 s instead of about 1.1, and its second block with the T4's answer.

**The browser (10-12, row 7).** With the network panel open, a warm turn: the stream request lasted
13.34 s, and the browser played R4 streamed's first audio 5.83 s after Send (3.41 s on the L40S). The
browser gets the blocks as they come. The 8.91 s of row 4 is not explained by a run of its own: the
slowing of row 6, larger on a 6 s recording, is the likeliest part of it.

**The cause of the slowing (row 8).** The web container measures the visitor's voice (Praat) during
the T4's request, and it had Modal's default share of a core. On the laptop the measurement takes
0.23 s and holds every other thread up to 101 ms at a time; on a fraction of a core, seconds. So the
page's container gets a full core (`cpu=1.0`): a few cents a day while the page is used, outside the
day's cap, which counts the GPUs.

**Run before the hand-over**, in a scratch copy of the tree on the laptop: the full suite, 978 passed, 2 skipped.

## 4. After the fix, and M32a done

**The check (row 9)**, the page's set-up (HTTP/2, the T4's request beside the stream):

| | before the fix | after |
|---|---:|---:|
| the stream's first lines | 4.17 s | 1.53 s |
| its first block reached the client | 5.14 s (made at 3.14) | 5.76 s (made at 3.41) |
| the first block's lag after it was made | about 0.9 s | about 0.8 s |

The fix did what it was for and no more: the opening lines are no longer held while the voice is
measured, but a block reaches a client 0.6 to 0.8 s after the L40S makes it, with the T4 beside it or
not, so that lag is not the web container's CPU. Not measured: how it splits between the two hops (the
L40S to the web container, the web container to the client); a block is about 150 kB of base64.
Reading the second block's arrival "with the T4's answer" (row 6) as a cause was wrong: here it came
on time against its making.

**What a visitor hears**, on a warm L40S: R4 streamed after about 1.5 s (upload and the call) + 3.1 to
3.4 s (the L40S) + 0.7 s (delivery), 5.3 to 5.6 s from Send; R4, decoded whole, after it is done and R4
streamed with it, 5 s or more later. The browser's number on the card is this whole wait, as measured
in row 7 (5.83 s).

**Deployed (row 10)** by the author on 10-12: https://ramcharanmc--s2s-lab-demo-web.modal.run. M32a is
done. Next, M32b: a Talker decode loop of our own, for the 3.1 s the L40S takes to make the first block;
a design note first.
