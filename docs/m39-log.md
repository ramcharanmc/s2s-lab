# M39 run log: the trim in the demo's R1

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m38-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-18 | — | chosen by the author after M38 (`docs/backlog.md`); the design (below) and the author's go | — | — |
| 2 | 10-18 | the tests, laptop | the demo's R1 is `r1-gpu-trim`; its measured row and a PR-018 panel from the registered run; first sound on every turn card | `modal_app/demo.py`, `demo.py`, `demo_page.html`, tests | 3c54204 |
| 3 | 10-18 | `modal serve modal_app/demo.py`, the author, at 3c54204: the Measured tab, then one live turn through the T4's four rungs | the T4 loaded the four in 37.7 s (R1 4.8 s), the cold Send 48.8 s. R1's card: first audio 351 ms, first sound 371, 325 ms of silence cut; R0's 402 and 722, R2's 470 and 750, R2 verbatim's 403 and 703; no rung failed. The Measured tab: R1's row from PR-018's run, its panel beside PR-015 to PR-017's. **The rule holds** | — | — |
| 4 | 10-18 | `modal deploy modal_app/demo.py`, the author | deployed at https://ramcharanmc--s2s-lab-demo-web.modal.run; its `/api/measured` and `/api/status` read from the laptop: R1's row from `runs/r1-gpu-trim/pr018.json`, PR-018's panel supported, R1's note served | — | — |
| 5 | 10-18 | — | M39 closed: these records | records | bc6f795 |
| 6 | 10-18 | — | the plans (Notion pages 16 and 17) checked in full against what was built: the README's "Not built: T4" corrected (built in M30); its Limits name the cascades' 0.5B model and the caricature risk (v1 §9); three v1 items the backlog lacked listed (the plans' model sizes, three harness metrics, P1c's latency); the data the plans named and did not use recorded | records | this commit |

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

## 2. The check (row 3)

One live turn, the author's recording ("Sweet home Alabama"), on a cold T4:

| card | first audio | first sound | between them | silence cut |
|---|---:|---:|---:|---:|
| R0 | 402 ms | 722 ms | 320 ms | — |
| **R1, trimmed** | **351 ms** | **371 ms** | **20 ms** | **325 ms** |
| R2 | 470 ms | 750 ms | 280 ms | — |
| R2 verbatim | 403 ms | 703 ms | 300 ms | — |

By the rule written before it: R1's first sound within 50 ms of its first audio, R0's still about 0.3 s after, no
rung failed. One turn: it shows the trim working in the demo, not a rate; the rate is PR-018's.

**Left as it was:** the laptop's version of the demo (`demoweb.laptop`) still offers `r1`: PR-018 ran on the T4.

**Found on the way:** `modal serve` now prints its temporary address with the region in it
(`ramcharanmc--s2s-lab-demo-web-dev.us-east.modal.run`); the deployed one is unchanged. The README said the lasting
address is the temporary one "without `-dev`", no longer true; corrected.

M39 cost about \$0.06: one cold T4 wake, one turn, and its 300 s window.
