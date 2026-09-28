# s2s-lab

Six speech-to-speech architectures, from a naive batch cascade to a full-duplex model with a LoRA,
built and measured against one frozen benchmark, with every claim registered before its test.

The point is the measurement, not the demo. Each rung is thin: get it working, measure it, say what
it bought and what it cost, and report the result whether or not it went the way it was predicted.

**Status, 2026-09-28** (commit `b4b4dd7`): rungs R0 to R5 built and measured; three fine-tunes run;
thirteen registrations, twelve of them reported, four with a supported claim. In progress: a demo
that shows the measured turns and answers a live one (`docs/m27-log.md`); run on Modal with
`modal serve`: a recording answered by R0 to R5 on four machines (R3, R4 and R5 woken only when
asked for; 33 to 82 s to wake), with what the visitor's voice carried beside the words the
cascades received; not deployed. About \$34 spent: \$9.37 on Modal, \$24.29 on Vast.ai (`docs/cost-ledger.md`).

## The ladder

Every rung is defined by which information-destroying boundary it removes.

| Rung | What it is | Machine | Time to first audio, p50 | WER, normalized | Memory |
|---|---|---|---:|---:|---:|
| R0 | naive cascade: Whisper, then an LLM, then Kokoro, all blocking | Modal T4 | 0.66 s | 11.9% | 1.9 GB |
| R1 | streaming cascade: the reply handed out clause by clause | Modal T4 | 0.49 s | 11.9% | 1.7 GB |
| R2 | two-tier: a slow recognizer (large-v3-turbo) drives the reply | Modal T4 | 0.89 s | 8.6% | not measured |
| R3 | speech-LLM: Qwen2-Audio hears the turn, no transcript in the loop | Modal A10 | 1.06 s | none in the loop | 17.0 GB |
| R4 | speech-to-speech, half duplex: Qwen2.5-Omni speaks the reply itself | Modal L40S | 9.17 s | none in the loop | 22.5 GB |
| R5 | full duplex: Moshi; PersonaPlex | Modal A10 | 150 ms; 460 ms response gap | none in the loop | 16.6; 18.1 GB |

**Read the table with care.** The rows come from different GPUs and different days, and the same
rung measured 17% slower three days apart. Only rungs run in one container are comparable: R1 is
26% sooner than R0 in theirs; R2's slow tier costs +0.31 s against the R1 beside it; R3 is 2.4
times R0 in theirs. R5's number is a response gap in stream time, not a turn's latency, and Moshi
answered 96 of 200 callers. The full table, with every caveat, is `runs/README.md` and the results
page it cites.

## What was claimed, and what happened

Thirteen registrations (`docs/preregistration.md`). Criteria are committed before the data is seen;
results are appended whatever they say.

| | Question, in short | Outcome |
|---|---|---|
| PR-002 | Does streaming the reply cut the wait for first audio? | not supported: 26% sooner, under its bars |
| PR-003 | Does a barge-in policy that reads the caller's words beat a duration threshold? | **supported**: wrong on 1 of 20 against 8 |
| PR-004 | Does a slow-tier recognizer cut WER by 3 points, and 5 on accented speech? | not supported: -3.4 overall, -4.3 accented |
| PR-005 | Does a \$0 tone tag let the LLM tell an upset caller from a calm one? | not supported, by one pair on fear |
| PR-006 | Does a speech-LLM that hears the audio rank the upset clip higher? | **Claim A supported**: 39 of 39; Claim B not |
| PR-007 | Does a per-caller baseline help the tone tagger, on fresh pairs? | not supported: 26 of 38 either way |
| PR-008 | Does a speech-to-speech model's voice move with the caller's? | not supported: no consistent direction |
| PR-009 | Does a severity score render onto speech as the mapping says? | not supported: pitch offset 88.0% against 90% |
| PR-010 | Does a LoRA move Moshi's turn gaps toward its training dialogues'? | not supported, moved away |
| PR-011 | Does a LoRA make Whisper write disfluent speech as spoken? | **supported**: raw WER 9.9% to 2.3% |
| PR-012 | Do the severity targets work as FastPitch's own inputs? | not supported: rate exact, pitch on target 57 to 60% |
| PR-013 | Quantized to 8 bits, does Kokoro's pitch drift while its words stay? | **Claim A supported**; fp16 not usable on a CPU |

PR-001 is registered and open. Of the twelve reported, eight were not supported, and each entry
says why.

## Run it

Python 3.10, Linux or WSL. The tests and the CPU track need no GPU and no account. The packages
are pinned in `requirements/`, one file per purpose, each with a header that says what it is for:
`base.txt` and `r0.txt` for the library and its models (CPU torch first), `affect.txt`, `data.txt`,
`demo.txt` and `modal.txt` as needed.

```bash
PYTHONPATH=src python -m pytest -q                      # the tests; the models are faked
PYTHONPATH=src python -m s2slab.rungs r0                # R0 on the laptop: the per-stage table
PYTHONPATH=src python -m s2slab.severity dev --listen   # the severity sweeps on Kokoro speech
PYTHONPATH=src python -m s2slab.quantladder dev         # the quantization ladder (needs the ONNX files)
```

The demo, on the laptop's CPU (`requirements/demo.txt`; the measured part needs no model). The
access code must be ten characters or more, or the live part stays closed; five wrong codes in
ten minutes close it for ten:

```bash
PYTHONPATH=src DEMO_ACCESS_CODE=... python -m uvicorn s2slab.demoweb:laptop --factory --port 8000
```

The live part offers R0, R1 and R2, and R2 with PR-011's recognizer (`r2-verbatim`) where its
export is: in `data/ft2/ct2`, or the folder `S2SLAB_FT2_CT2` names (`ft/README.md` says how to
fetch it). A folder holding any other model is refused.

On Modal (`requirements/modal.txt`, your own account; each command prints what it cost):

```bash
modal run modal_app/r0_bench.py --rung r0-gpu           # a T4, about \$0.01
modal serve modal_app/demo.py                           # the demo, at a temporary URL; needs the secret `s2s-lab-demo`
```

`modal serve` prints two addresses. The demo is the one ending in `modal.run`, printed once near
the top; the one printed last, on `modal.com/apps`, is Modal's dashboard. The laptop's address
(`localhost:8000`) answers only while the laptop's server runs.

The fine-tunes are their own project with their own environment: `ft/README.md`.

## Where things are

| Path | What |
|---|---|
| `src/s2slab/` | the library: rungs, turn-taking, barge-in, probes, measures, statistics |
| `modal_app/` | the GPU entry points, one per family of rungs, and the fine-tunes' training |
| `ft/` | the fine-tunes (Whisper LoRA, FastPitch), on the Lightning-Hydra template |
| `vast/` | the kit that trained the Moshi LoRA on a rented H100 |
| `tests/` | the tests; nothing in them needs a GPU |
| `probes/`, `evalset/` | scripts, labels and SHA-256 manifests of the test sets; the audio stays out of git |
| `runs/` | the complete output of every reportable run; `runs/README.md` says what each file holds |
| `docs/` | see below |

## The documents

| Document | What it records |
|---|---|
| `docs/preregistration.md` | every registration: criteria, predictions, deviations, results |
| `docs/cost-ledger.md` | one row per paid run, failed runs included |
| `docs/ft1-log.md`, `ft2-log.md`, `ft3-log.md`, `m26-log.md`, `m27-log.md` | run logs, by event: what ran, what failed, the cause, the fix |
| `docs/backlog.md` | what is not done, and what was chosen next |
| `docs/colab.md` | the Colab route for interactive GPU work |
| `runs/README.md` | every committed run, and how to re-score it without audio or a model |
| `probes/README.md` | how each probe set was built, recorded and frozen |
| `ft/README.md`, `vast/README.md`, `evalset/README.md` | the fine-tunes, the training kit, the eval set |

## Limits, stated

- **No clinical validity is claimed and no diagnostic capability is implied.** The severity render
  maps a score to acoustic targets and measures whether the audio reached them. Every synthesized
  file is named and tagged synthetic.
- **No patient data, ever.** Every dataset is public; each licence is named where the data is used.
- **Latency is measured server-side** with a synthetic audio driver. The client transport is out of
  scope: Modal offers HTTP and WebSocket, not WebRTC.
- **One speaker's voice** is in the hand-recorded probe sets, the author's.
- **Not built:** T4, the voice-activity-projection turn-taking model.

## Licences

The code has no licence file yet. Models and datasets keep their own: each is named in the README
of the part that uses it.
