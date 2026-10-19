# M42 run log: CI and the severity badge

What happened, in order. The earlier logs: `docs/m41-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-19 | the suite on a fresh checkout, laptop, scratch: no audio; then with the model libraries hidden from it | without the audio git does not hold, 2 tests failed where 8 others skip; with only `base.txt`'s packages, 8 R4 test files failed to import torch and about 110 tests failed on transformers, Modal, FastAPI and torch; `tests/test_severity.py` passed alone, 19 tests in 2.1 s | — | — |
| 2 | 10-19 | — | the design (section 1) and the author's go | — | — |
| 3 | 10-19 | the tests, laptop | `requirements/test.txt`; the two tests skip without their audio; the two workflows; the README's badges | `requirements/test.txt`, `.github/workflows/`, tests | this commit |

## 1. The design

v1 §4 (week 6) asked for the monotonicity test "in CI" and its badge, "severity monotonicity: 24/24 passing"; v1
§10's third sentence says it "runs in CI in ninety seconds". The repository had no CI.

- **The environment, written down.** The laptop's environment had five packages that no requirements file named:
  torch 2.14.0 (CPU; `r0.txt` says to install it, but pins no version), onnxruntime 1.23.2 (M26), piper-tts 1.8.0
  (M38), httpx 0.28.1 (FastAPI's test client) and av 17.1.0 (the demo). `requirements/test.txt` names everything the
  suite imports, at the versions it passes with.
- **Two tests skip** when their audio is not on the machine, as eight others already did: the eval set's manifest
  check, and R3's probe lock, whose code checks the frozen set before the lock.
- **Two workflows.** `tests.yml`: the suite on Ubuntu 24.04, Python 3.10, on every push to `main` and every pull
  request, pip's downloads cached. `severity.yml`: `tests/test_severity.py` with `base.txt` alone; its badge is the
  plan's. ft/'s own 33 tests (NeMo) are not run.
- **The author's account.** The repository is private, so the runs use the free Actions minutes of the author's
  GitHub plan, and the badges show only to those who can see it.

**The rule, written before the first run:** both workflows finish green on their first push. If one fails, the
cause is found and the fix shown before it is applied. The severity job's time is recorded against v1's "ninety
seconds".
