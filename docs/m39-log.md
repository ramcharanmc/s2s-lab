# M39 run log: the trim in the demo's R1

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m38-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-18 | — | chosen by the author after M38 (`docs/backlog.md`); the design (below) and the author's go | — | — |
| 2 | 10-18 | the tests, laptop | the demo's R1 is `r1-gpu-trim`; its measured row and a PR-018 panel from the registered run; first sound on every turn card | `modal_app/demo.py`, `demo.py`, `demo_page.html`, tests | this commit |

## 1. The design

**What changes.** The demo's R1 becomes `r1-gpu-trim`, PR-018's rung: the same models, so nothing new reaches the T4.
R0, R2 and R2 verbatim keep Kokoro's silence, as does R3 on the A10: PR-018 tested R1 alone, and a trim anywhere else
would be untested. R1's measured row is now PR-018's run, the rung offered (as R4 streamed's row is PR-017's), and a
PR-018 panel joins PR-015 to PR-017's.

**What a visitor sees.** The T4's rungs answer whole: the reply reaches the browser as one recording, not clause by
clause. The trim does not change when it arrives; it makes the recording start about 0.3 s sooner. So every turn card
now says when the reply's own audio stops being silence ("first sound") beside first audio, and R1's says what was cut.

**The rule, written before the check.** In a `modal serve` session, a live turn's R1 card shows first sound within
50 ms of first audio, R0's card still about 0.3 s after it, and no rung fails. Then `modal deploy`, the author's.
