# M41 run log: the cold-start write-up

What happened, in order. The earlier logs: `docs/m40-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-19 | — | chosen by the author: v1's and v2's "cold-start economics", the last of week 8 but publishing; the design and the author's go | — | — |
| 2 | 10-19 | — | `docs/cold-start.md`, from the runs already made, \$0: every measured number cited to its log line, every computed one marked arithmetic | `docs/cold-start.md` | this commit |

## 1. The design

One document, no new run. What a wake is and how the demo measures it; each machine's wake over time and what
changed it; the L40S's load by phase; what a cold turn costs, the wake and the idle window apart; cold against warm,
with the day's cap beside it; the idle window's choice; and what is not measured. The numbers were gathered from
`docs/m27-log.md` to `docs/m39-log.md`, `docs/cost-ledger.md` and `modal_app/demo.py`, and each one cited was read
again at its line before it went in.

## 2. What it found

- **The idle window, not the wake, is most of what a cold turn costs** since M36: the T4's 300 s window about
  \$0.049 against its 50 s wake's \$0.008; the L40S's 120 s about \$0.065 against its 72 to 93 s wake's \$0.04 to
  0.05 (arithmetic from list rates).
- **On the L40S a wake is mostly compiling**: the weights load in about 4 s of 185.7 (M35); a compile cache took the
  load to 72 to 93 s (M36).
- **Scale-to-zero is two orders of magnitude cheaper** for a demo a few people open: a warm T4 is \$14.16 a day, about
  250 cold turns; the day's \$3 cap is about 50.
- **Not known**: the CPU container's cost, the image builds, the bill against the estimates (checked once), and
  whether Modal's 150 s web-request limit applies to a streamed response: cold streams of 164 to 357 s did not fail.
