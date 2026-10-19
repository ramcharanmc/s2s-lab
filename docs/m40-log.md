# M40 run log: P1b on real disfluent speech

What happened, in order: every run, every failure, its cause, the code that changed, and the commit.
Kept as events happen. The earlier logs: `docs/m39-log.md` and those it names.

| # | When | What ran | Outcome | Code change | Commit |
|---|---|---|---|---|---|
| 1 | 10-18 | — | chosen by the author after the plan review (`docs/backlog.md`, "From the plans, not done"); the design (below) and the author's go | — | — |
| 2 | 10-18 | the tests, laptop | DisfluencySpeech's validation and test splits as endpointing turns; the DEV sweep and its rule, written before it | `dsturns.py`, tests | 28657de |
| 3 | 10-18 | `python -m s2slab.dsturns build`, the author, laptop | DEV (the validation split): 215 clips, 74.1 MB fetched; TEST (the test split): 207 clips, 70.8 MB, 50 of them the eval set's; none outside 3 to 15 s. The manifests: `probes/disfluencyspeech/manifest-{dev,test}.json` | — | — |
| 4 | 10-18 | `python -m s2slab.dsturns dev`, the author, laptop, \$0 | 215 DEV turns, 29 settings (section 2). T0 at 300 ms cut off 155 (72.1%): **GO by the rule**. The run file's commit reads `28657de-dirty`: the stamp counts untracked files, and the only ones were the manifests the build had just written; no tracked file differed | — | — |
| 5 | 10-18 | — | **PR-019 registered** before any test turn was read; T2's setting from DEV by the entry's rule | `pr019.py`, `docs/preregistration.md`, tests | this commit |

## 1. The design

**The question, v1 §8's P1b:** "Prosodic policy dominates the fixed threshold on the SEP-28k slice at some
operating point with cutoff rate < 5%." Tested so far on the synthetic disfluent cohort only, where it was not met
(28% at best; Notion page 18 §7).

**Not SEP-28k.** The eval set's disfluent cohort was never SEP-28k: it is 50 clips of DisfluencySpeech's test split
(`src/s2slab/evalset.py`). SEP-28k cannot carry P1b: its audio is not distributed (it is fetched from the podcasts,
which keep their copyright), its labels are CC BY-NC, its clips are fixed 3-second windows whose end is where the
window stopped rather than where a speaker finished, and it has no words for T1 to read (PR-011's background found
the same; LibriStutter's stutters are synthetic, FluencyBank needs a membership). An endpointer is scored against a
turn's end, so a clip with no end cannot test one.

**DisfluencySpeech instead** (Apache-2.0, the eval set's own pin `b7da294`): one speaker re-performing Switchboard
utterances with their fillers, repetitions and restarts. Real recorded speech with real pauses, each clip one whole
utterance, so its end is a turn's end. PERFORMED, not clinical, and one speaker: the limit PR-011 stated.

- **DEV: the validation split.** Every clip passing the eval set's disfluent rules (the verbatim transcript differs
  from the fluent one, at least 6 words, 3 to 15 s).
- **TEST: the test split**, the same rules: the eval set's 50 and the rest of the split, never used by anything
  here. All of them, so a cut-off rate under 5% can be told apart from one of 7%; with the eval set's 50 alone,
  even no cut-off at all leaves a 95% interval reaching 7%.
- **The turns: M30's harness** (`realturns.turn`): cropped before the first speech, ended at the last stretch of
  speech the policies' own VAD finds, padded 2.5 s with noise at the clip's own floor; words from faster-whisper
  base.en with word times, as T1 reads them.
- **The download is the author's** (`python -m s2slab.dsturns build`, about 140 MB); the audio stays out of git
  (`probes/disfluencyspeech/audio/`), the manifests with every file's SHA-256 are committed.

**The DEV sweep** (`dsturns.dev_policies`): T0 at 150 to 1,500 ms; T1 at 300 to 900 ms; T2 at 300 to 900 ms with
slope thresholds of 20, 40 and 80 Hz/s (40 was set on the synthetic cohorts, page 18 §9); T3 at cost ratios 2 to 32.
Each: cut-offs with a Wilson interval, turns never ended, the median and p90 wait.

**The rule, written before the DEV run** (`dsturns.gate`): **GO** to PR-019 if T0 at 300 ms cuts off at least
**20%** of the DEV turns, so the clips pause long enough mid-turn for the endpointing question to have an answer;
otherwise **STOP**, nothing registered, and the reason recorded. PR-019's criteria, T2's setting and the bars are
set from the DEV run, before the test turns are read; the test turns are refused without `confirm="PR-019"`.

## 2. The DEV run (row 4)

`runs/turn-taking/p1b-dev.json`: 215 turns of DisfluencySpeech's validation split; 147 pause mid-turn, every one of
them for more than 300 ms, the longest pause 360 ms at the median; the turn 6.8 s.

| setting | cut off, of 215 | median wait | p90 |
|---|---:|---:|---:|
| T0 at 500 ms | 47 (21.9%) | 500 ms | 500 |
| T0 at 700 ms | 6 (2.8%) | 700 | 700 |
| T0 at 900 ms | 2 (0.9%) | 900 | 900 |
| T0 at 1,100 ms | 0 | 1,100 | 1,100 |
| T1 at 700 ms | 5 | 700 | 1,100 |
| T2 at 700 ms, slope 20 / 40 / 80 Hz/s | 13 / 10 / 9 | 700 | 960 to 1,100 |
| T2 at 900 ms, slope 20 / 40 / 80 Hz/s | 1 / 1 / 1 | 900 | 1,160 / 1,300 / 1,300 |
| T3, cost ratio 32 | 2 | 840 | 840 |

**What it shows.**

- **These pauses are short.** A fixed 700 ms cuts off 2.8% of these real disfluent turns; on the synthetic
  disfluent cohort, 49% (Notion page 18 §5). The performed disfluencies here pause for about a third of a second.
- **T2 never answers sooner.** At every setting its median wait is its base threshold, the same as T0's. Its
  prosodic cue moves only the tail, later (p90 1,160 ms against 900).
- **At 700 ms T2 cuts off more than T0** (9 to 13 against 6). At 900 ms it cuts off one turn fewer (1 against 2).

**T2's setting for PR-019**, by the rule written in its entry before any test turn: among the T2 settings that cut
off under 5% of DEV turns and are at least as safe and at least as fast at the median as T0 at the same base,
better in one, the lowest median wait, then the fewest cut-offs, then the lowest p90. Only the three at 900 ms
qualify (one fewer cut-off, the same median); the 20 Hz/s slope has the lowest p90. So: **T2 at 900 ms, 20 Hz/s,
against T0 at 900 ms**.

**Then:** PR-019's one run, the author's, on the laptop: `python -m s2slab.pr019 --confirm PR-019`.
