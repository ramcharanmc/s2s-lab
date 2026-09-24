# Probe sets

Hand-recorded audio for testing the policies on **real speech**, from the v2
Engineering plan (Notion page 17, §6.3). The simulator can show how a policy
behaves when pauses fall where we put them. Only real recordings show what real
acoustics, real pitch and a real VAD do at those pauses.

| Set | Items | Serves | When |
|---|---:|---|---|
| **Endpointing** | 30 | Turn-taking, T0–T3 and PR-001 | **Now** |
| Prosody | 30 | R3 speech-LLM probe (sincere vs distressed delivery) | Before R3 |
| **Barge-in** | 20 | R1 part 2: stop for an interruption, keep going for a backchannel | **Now** |

## Rules

- **Never record real personal details.** The natural items ask for made-up
  addresses, numbers and names. This repository is meant to be public.
- **Audio stays out of git.** `*.wav` is already ignored. What gets committed
  is the script, the labels, and a manifest of SHA-256 checksums. That freezes
  and versions the set without publishing a voice recording by accident.
- **Once frozen, a probe set never changes.** Additions become a new version.
  From v2 §6.2: *silent test-set changes are how people accidentally claim
  improvements.*
- **Know what this is and isn't.** It is one fluent speaker *performing*
  hesitations. It tests real audio at pause positions we chose. It is not a
  sample of disfluent speakers; that is what SEP-28k is for.

---

## Endpointing probe set

`endpointing/script.tsv` has 30 items in three groups of equal weight to the
question they answer:

| Type | Items | Pause falls right after… | What it tests on real audio |
|---|---:|---|---|
| `function` | 12 | a function word ("is", "the", "about") | Does T1's hold fire and save the turn? |
| `content` | 12 | a content word ("seven", "road", "david") | The failure `diagnose.py` found: T1 cannot see these |
| `natural` | 6 | wherever you actually pause | Where does a real speaker pause when not told to? |

The first two groups replay the simulator's finding on real speech. The third
is the question the simulator cannot answer at all.

### How to record

1. **Software:** Audacity (free). Set the project rate to **16000 Hz**, and the
   recording channel to **mono**.
2. **Room:** quiet, same room and microphone for every item. Note the
   microphone at the bottom of this file.
3. **Each item:**
   - say the line, pausing at each `|` for the length its number calls for
     (table below),
   - finish the sentence, then **stay silent for at least 3 seconds** before
     stopping.

   The trailing silence matters. The slowest possible decision on the ladder
   needs just over a second of silence, and a stop click inside that window
   resets it. Stop too early and the item never gets a decision.

   | Items | Pause at each mark | Silently count |
   |---|---|---|
   | 01, 04, 07, 10 | **long**, about 1.2 s | "one-and-two-and" |
   | 02, 05, 08, 11 | **short**, about 0.4 s | "one" |
   | 03, 06, 09, 12 | **medium**, about 0.7 s | "one-and" |

   The same rule covers both `f` and `c` items. You won't hit the lengths
   exactly, and don't need to: the loader measures every real pause from the
   labels, and the analysis groups turns by the pause you actually made. The
   rotation only makes sure the pauses spread across the range where the
   policies disagree.
4. **Natural items** have no `|` marks. Don't plan pauses; say it as you would
   on the phone.
5. **Export** each item as WAV (signed 16-bit PCM), named by its id:
   `endpointing/audio/f02.wav`, and so on. **Never edit the audio** — no
   trimming, no noise removal. The loader crops before your first word itself,
   and a frozen set's checksums must describe what you actually recorded.

### Labelling

1. **Draft:** `PYTHONPATH=src python -m s2slab.probes propose f02 f03 ...`
   It skips any item that already has a label file, so it never overwrites your
   corrections. Use `--force` only to redraft one item on purpose.
2. **Correct in Audacity:** open the WAV, import its label file, and make each
   region **start at the first word of its chunk and end where its last word
   ends**. The end of the last region is the true end of turn. Keep breaths and
   clicks outside every region.
3. **Natural items:** replace each `?` with the words you said, writing numbers
   as spoken ("seven four two", not "742").
4. **Export** the labels over the same file, then run
   `PYTHONPATH=src python -m s2slab.probes check`.

### Freezing

When every item loads, `PYTHONPATH=src python -m s2slab.probes freeze` writes
`endpointing/manifest.json`: the SHA-256 of the script and of every recording
and label file. Commit it. From then on `PYTHONPATH=src python -m s2slab.probes verify`,
and one test in `pytest`, fail if a single byte changes. To change the set,
make a new version. Never re-freeze this one.

### Pilot result (2026-09-13) — why the pause rule changed

The pilot (f01, c01, n01) was recorded with every pause at "about one second".
The measured pauses were 1.14–1.52 s, longer than the longest wait of any
policy tested (900 ms), so **every policy interrupted every item**:

| item | pauses s | T0 500 | T0 700 | T0 900 | T1 500 | T2 500 |
|---|---|---|---|---|---|---|
| f01 | 1.34 | CUT after 'the' | CUT | CUT | CUT | CUT |
| c01 | 1.52, 1.26 | CUT after 'seven' | CUT | CUT | CUT | CUT |
| n01 | 1.14 | CUT after 'terrace' | CUT | CUT | CUT | CUT |

A probe set on which every policy fails the same way cannot tell them apart.
The policies disagree for pauses between about 0.5 and 0.9 s — T0 at 500 ms
cuts, T1 holds after a function word — so the pause rule now spreads pauses
across that band. The pilot takes are kept: f01 and c01 are "long" items under
the new rule. The one unprompted pause (n01, 1.14 s, after "terrace") is a
single data point that a natural hesitation can outlast every threshold.

The pilot also caught three problems, all fixed before the other 27:

- a click before c01's first word, followed by 860 ms of silence, which would
  have made every policy "end the turn" before speech began (the loader now
  crops, and labels must start at the first word);
- a `propose` command that overwrote hand-corrected labels (it now refuses);
- a first n01 take with only 0.74 s after the last word (the loader now refuses
  any item with too little).

### Layout

```
probes/
├── README.md                 this file
└── endpointing/
    ├── script.tsv            what to say (committed)
    ├── audio/                your recordings (NOT committed -- *.wav is ignored)
    ├── labels/               where each pause and the true end are (committed)
    └── manifest.json         SHA-256 of every WAV, written when the set freezes
```

### Recording notes

- Microphone: Earbuds
- Room: My Personal Bedroom
- Date recorded: 09/20/2026

---

## Barge-in probe set

While the agent is speaking, the caller makes a sound. Either the agent should
**stop** (a genuine interruption) or **keep going** (a backchannel such as
"mm-hm"). `bargein/script.tsv` has 20 items, half of each, crossed with how
long the caller speaks:

| Cell | Items | Example | A duration threshold… |
|---|---:|---|---|
| `stop-long` | 5 | "no sorry, that is the wrong day, I asked for Thursday" | gets it right |
| `stop-short` | 5 | "no, wait" | misses it, if set long |
| `continue-short` | 5 | "mm-hm", a cough | false-stops, if set short |
| `continue-long` | 5 | "okay yeah, that is fine, got it" | false-stops at any setting |

The first policy anyone writes is "stop if the caller speaks for more than N
ms". This set says in advance where that policy must fail: whatever N is, it
gets `stop-short` or `continue-short` wrong, and `continue-long` always. A
policy that reads the words, or the pitch, has something to win. Two of the
`continue-short` items are not speech at all (`kind` = `noise`: a cough, a
short laugh), the classic cause of a false barge-in; they are reported
separately from the spoken backchannels.

**What it makes measurable:** false barge-in rate, missed-interruption rate,
stop latency (caller starts → agent stops), and which of the agent's words had
been played when it stopped, which is what state truncation needs (v2 §4, R1).

### How it is built

- **The agent's side is pre-rendered.** `bargein/agent.tsv` holds five replies
  of 10–12 s. They are synthesized once with Kokoro `af_heart`, the voice R0
  and R1 use, into `bargein/agent/A1.wav` … `A5.wav`. Each reply is used by four
  items, one per cell; the `after` column is the phrase the caller comes in
  after, and it occurs exactly once in its reply.
- **Only the caller's side is recorded**, over headphones, while the agent WAV
  plays. The recording starts when the agent's reply starts, so the recording's
  timeline *is* the playback timeline, and "when did the caller come in" is a
  label time.
- **Out of scope, stated loudly: echo.** The two channels here are clean and
  separate. On a real phone the agent's voice leaks into the caller's
  microphone, and barge-in detection needs echo cancellation in front of it.
  This set tests the decision, not the echo canceller (v1 §6 scopes the client
  transport out).
- **One speaker performing interruptions**, as in the endpointing set. It is
  not a sample of how callers interrupt.

### How to record

1. **Render the agent lines, once:**
   `PYTHONPATH=src python -m s2slab.bargein render`
2. **Audacity:** project rate **16000 Hz**, mono recording. Turn **overdub on**
   (Transport → Transport Options → "Hear other tracks during recording") and
   software playthrough **off**. Listen on **earbuds, at the lowest volume you
   can still follow** (the pilot used system volume 1): the agent must not
   reach the microphone. **Test it before each session:** record
   `agent/A3.wav` playing while you stay silent, export it as
   `bargein/audio/leaktest.wav`, and run
   `PYTHONPATH=src python -m s2slab.bargein leak`. It must say "clean".
3. **Calibrate, once per session.** Import `agent/A3.wav`, put the cursor at
   the start (Home), turn the volume up for this take only, hold an earbud
   against the microphone, and record while the line plays. Export the recorded
   track as `bargein/audio/calibration.wav`, then
   `PYTHONPATH=src python -m s2slab.bargein calibrate` reports how many
   milliseconds the recording lags the playback. **Turn the volume back down.**
   The set uses one lag for all 20 items: if a later session's figure differs
   from the committed `calibration.json` by more than 10 ms, stop and find out
   why before recording.
4. **Each item:**
   - new project; import the item's agent WAV (`cs03` uses `agent/A3.wav`);
   - cursor at the start (Home), press Record, listen;
   - right after the `after` phrase, say the item's `text`, **talking over the
     agent**: it will not stop for you, which feels odd and is correct;
   - for `[cough]` and `[short laugh]`, do that, once;
   - then stay silent until the agent's reply has finished, plus two seconds.
5. **Export the recorded track only:** mute the agent track, then export the
   whole project as WAV (signed 16-bit PCM) to `bargein/audio/<id>.wav`. The
   export must start at time zero. **Never edit the audio**: no trimming, no
   noise removal, no re-timing.
6. **After each session:** `PYTHONPATH=src python -m s2slab.bargein check`. It
   flags a take to re-record (agent leak, a peak at full scale), a short item
   that ran long, and any file in `audio/` that is not an item id. **The ids
   use the letter l, not the digit 1:** `sl03`, not `s103`.

### Pilot result (2026-09-19) — why the protocol changed

Four items on A3 (`sl03`, `ss03`, `cs03`, `cl03`), recorded twice.

**Round one, over-ear headphones: the agent leaked into the microphone.** With
the caller silent, the level while the agent spoke was 7 to 19 dB above the
room, and it lined up with the agent's audio (cross-correlation sharpness 24 to
76). It was 30 dB under the caller's voice, inaudible on playback. But the VAD
every policy uses calls anything 10 dB over the noise floor speech, so a
barge-in policy would have heard the *agent* as the caller: the echo problem
this set scopes out, arriving through the headphones. It also broke the draft
labels: leak and voice fused into one 3-second "sound".

**Round two, earbuds at system volume 1: clean.** Level while the agent speaks
+2.5 to +7.3 dB, match with the agent's audio 3 to 6. The +7.3 dB take was two
breaths, which is why a take counts as leaking only if it is loud **and**
matches the agent.

| item | lasts | came in after the phrase by | agent words heard |
|---|---:|---:|---:|
| `sl03` stop-long | 3.80 s | −157 ms | 11 of 33 |
| `ss03` stop-short | 0.70 s | +240 ms | 9 of 33 |
| `cs03` continue-short | 0.26 s | −97 ms | 11 of 33 |
| `cl03` continue-long | 2.04 s | +308 ms | 17 of 33 |

(From draft labels, before hand correction.) Both interruptions of "half past
two" begin before the agent has finished the word: real callers anticipate.
Already on four items, no duration threshold is right: 200 ms stops for
"right", 800 ms misses "no, wait", and the long acknowledgement fools all four.

What the pilot changed:
- **Playback is earbuds at the lowest volume that can be followed**, with a
  silent leak test before each session (step 2).
- **The overdub lag belongs to the machine, not the headphones:** 202.1 ms with
  the over-ear pair, 202.4 ms with the earbuds, and 202 to 206 ms recovered
  from the leak in each round-one take. One calibration per session is enough.
- Tools: `leak`, the leak and clipping checks in `check` and `freeze`, draft
  labels gated on the caller's own level, and a warning for misnamed files
  (two pilot takes were saved as `s103.wav` and `c103.wav` and silently skipped).
- One take peaked at 0.997 of full scale; `check` now flags that too.

### Labelling

One region per item: from the start of your first sound to the end of your
last. `python -m s2slab.bargein propose` drafts it, but **the draft's edges are
approximate**: it only keeps sound within 25 dB of the take's loudest moment,
which clips the soft start and tail of a word (the pilot's "right" was drafted
at 0.26 s). Correct every region in Audacity as for the endpointing set,
keeping breaths and clicks outside it. The region's start is the interruption
onset; its length decides whether a "short" item really was short, and the
analysis uses the measured length, not the cell's name.

### Freezing

`python -m s2slab.bargein freeze` writes `bargein/manifest.json`: checksums of
both TSV files, the five agent WAVs, the 20 recordings and the 20 label files.
The agent WAVs are hashed because another machine's Kokoro may differ in the
last bit, and the labels are only meaningful against the audio that was playing.

### Layout

```
probes/bargein/
├── agent.tsv          what the agent says (committed)
├── script.tsv         what the caller does, and when (committed)
├── agent/             A1.wav … A5.wav, rendered once (NOT committed)
├── audio/             your recordings, and calibration.wav (NOT committed)
├── labels/            one region per item (committed)
└── manifest.json      SHA-256 of all of the above, written when the set freezes
```

### Recording notes

- Microphone: earbuds' built-in microphone · Playback: **earbuds at system volume 1** (see Pilot result)
- Room: personal bedroom, door closed · measured room level −59 to −70 dB re full scale
- Date recorded: pilot and the other 16 takes 2026-09-19; `cs01` restored from a same-day copy on 2026-09-20
- Overdub lag from `calibrate`: **202.4 ms** (sharpness 78), 2026-09-19

## Prosody probe set

`probes/prosody/` · 60 pairs · built by `src/s2slab/prosodyset.py` · frozen 2026-09-21

The v2 plan, §6.3: *"Neutral words, loaded delivery … R1 cannot distinguish the
pairs by construction — this is the input bottleneck, demonstrated rather than
argued."* One item is a **pair**: the same actor saying the same sentence, once
neutrally and once with feeling. A cascade hands its LLM a transcript, and both
clips of a pair have the same transcript.

**Nothing here was recorded for this project.** A label that is a perception
needs listeners, and one untrained speaker has none. The clips come from
**CREMA-D**, where 2,443 raters said what they heard:

> Cao H, Cooper DG, Keutmann MK, Gur RC, Nenkova A, Verma R. *CREMA-D:
> Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Transactions on
> Affective Computing 5(4):377–390, 2014. github.com/CheyneyComputerScience/CREMA-D,
> pinned at commit `1658cd342dff90010aa843eaeebd53610a08b1dc`.

91 actors (48 male, 43 female, aged 20 to 74) speak 12 sentences whose words
carry no emotion ("It's eleven o'clock", "I think I have a doctor's
appointment", "Don't forget a jacket"…), each in six acted emotions.

**Licence.** CREMA-D's database is under the Open Database License (ODbL 1.0),
its contents under the Database Contents License (DbCL 1.0). `pairs.tsv` and
`manifest.json` contain information from CREMA-D, which is made available under
the ODbL, and are offered under the same licence. The audio and the ratings
files are not redistributed here: the builder fetches them from the source.

### "Distress" is a proxy, three times over

CREMA-D has no such category. **Sad, fearful and angry** delivery are three
separate stand-ins, reported one by one and never pooled. This is acted emotion
by consenting actors; nothing here claims clinical validity or diagnoses anyone
(v1 plan, ethics boundary). Each pair keeps the raters' mean intensity for the
loaded clip — a number beside the label.

### Selected by rule, from the ratings alone

Nobody listened first. The rule was committed (`f56c55d`) before any rating was
read:

1. **Voice-only ratings** — the raters who heard the clip and saw no face.
2. A pair qualifies if the neutral clip was heard as neutral and the loaded clip
   as its intended emotion, each by at least the threshold, of at least 8 raters.
   A tie is not a majority.
3. **Actors are split into dev (30) and test (61) before any pair is drawn**, by
   the seed alone (20260921). No voice is on both sides.
4. Each emotion's threshold is the highest of 90 / 80 / 70 / 60 / 50% that fills
   both sides (7 dev + 13 test pairs).
5. A seeded draw: one pair per actor per emotion, no neutral clip in two pairs,
   at most 2 (dev) or 3 (test) pairs of one sentence per emotion.
6. Every chosen clip is cross-checked against CREMA-D's own `summaryTable.csv`.

The counting that chose the thresholds (dev + test pairs the draw could fill):

| raters agreeing | SAD | FEA | ANG |
|---|---|---|---|
| at least 90% | 1 + 2 | 7 + 6 | **7 + 13** |
| at least 80% | 1 + 4 | **7 + 13** | · |
| at least 70% | 6 + 12 | · | · |
| at least 60% | **7 + 13** | · | · |

**Sadness is rarely heard from the voice alone**: of 91 actors, three sad pairs
reach 90% agreement. The sad pairs are the weakest proxy, and say so: loaded
clips heard as sad by 60 to 100% of raters (median 67%), against 80 to 100% for
fear (median 90%) and 90 to 100% for anger (median 100%).

**One change to the rule, recorded.** The first rule used one threshold for all
three emotions and landed at 60%, because sadness needs it; 8 of the 20 anger
pairs then sat under 70% although anger fills at 90%. Rule 4 was changed to one
threshold per emotion (`e2d1436`) after seeing the counting table and nothing
else: no clip had been fetched, no system run.

### Dev and test

21 dev pairs (14 actors) for developing anything that reads prosody; **39 test
pairs (29 actors) are a test set**: `prosodyset.load("test")` refuses without
`confirm` naming a registration in `docs/preregistration.md`, like the
simulator's EVAL sentences and the frozen barge-in recordings.

### Layout, and rebuilding it
probes/prosody/
├── pairs.tsv the rule's output: 60 pairs, agreement, raters, intensity (committed)
├── manifest.json source revision, seed, SHA-256 of pairs.tsv and of all 120 clips (committed)
├── audio/ 120 clips, 16 kHz mono, 1.5 to 4.1 s, 9.8 MB (NOT committed)
└── source/ CREMA-D's two ratings files, 4.5 MB (NOT committed)

A clone has the first two. `python -m s2slab.prosodyset audio` fetches the 120
clips from the pinned revision (the full corpus is 7.5 GB; only these are
fetched), and `python -m s2slab.prosodyset verify` proves they are the frozen
ones.

### What it is not

- **Acted, not felt**, by North American actors reading fixed sentences. It
  tests whether a system can use delivery at all, not how callers sound.
- **Possibly seen before.** CREMA-D is a standard emotion-recognition benchmark;
  a speech-LLM may have trained on it. A good score here can be memory. A small
  set in the author's own voice, which no model has seen, is parked until a rung
  that could use it exists.
- **Three emotions as stand-ins for one idea**, and one of the three weakly
  validated. Results are per emotion.
- **60 pairs.** Enough for a large effect per emotion (13 test pairs each), not
  for a subtle one.

### First look — the 21 dev pairs only (2026-09-21)

`python -m s2slab.prosodycheck acoustics` and `… blind`. Descriptive; no claim
is tested, and the 39 test pairs are not loaded.

**The difference is in the sound.** Loaded minus neutral inside each pair: the
median change, and how many of the 7 pairs moved that way. Pitch is tracked
from 75 to 600 Hz — the 60 to 400 Hz that serves endpointing clips fearful and
angry speech.

| | median pitch | pitch range | level |
|---|---|---|---|
| SAD | +0.4 st (4 of 7) | +2.0 st (6 of 7) | −2.0 dB (6 of 7) |
| FEA | **+9.3 st** (6 of 7) | −1.6 st (5 of 7) | **+10.3 dB** (6 of 7) |
| ANG | **+7.7 st** (7 of 7) | +0.5 st (4 of 7) | **+14.7 dB** (7 of 7) |

Anger and fear raise the voice by more than half an octave and 10 to 15 dB.
Sadness barely shows in these numbers, as it barely showed to the raters. A
level difference of this size would not survive a telephone's gain control;
the pitch difference would. **Speaking rate is not reported:** two ways of
timing the spoken part both turned the level difference into a length
difference (one made anger 35% faster, the other 42% slower). Rate needs word
times from a recognizer.

**The difference is not in the words.** Both clips of each pair through R1's
recognizer (Whisper base.en):

| | pairs | same transcript for both clips | WER, neutral | WER, loaded |
|---|---:|---:|---:|---:|
| SAD | 7 | 5 | 2.3% | 2.3% |
| FEA | 7 | 6 | 7.0% | 0.0% |
| ANG | 7 | 7 | 0.0% | 0.0% |

In 18 of 21 pairs R1's LLM is given the same words for both clips, so it gives
the same reply: the delivery is gone before it sees anything. The three that
differ are recognition errors — two of them on the *neutral* clip ("…a doctor's
appointment to rate it") — and loaded speech is no harder to recognize than
neutral.

### The affect side-channel on the test pairs — PR-005 (2026-09-21)

`src/s2slab/affect.py` (the tagger: 88 eGeMAPS numbers into a class-balanced
softmax regression, fitted on 398 clips from the 16 dev actors who have no
pair) and `src/s2slab/sidechannel.py` (R1's LLM asked *"Judging by their tone
of voice, is the caller upset?"* with the tagger's word in the prompt). Full
registration and results: `docs/preregistration.md`, PR-005. Evidence:
`runs/prosody/`.

**Registered before the run, not supported.** Pairs answered right of 13 —
loaded clip Yes and its neutral twin No — against shuffled tags:

| | none (R1 as it is) | tagger | shuffled (mean of 1000 deals) | oracle | bar | |
|---|---:|---:|---:|---:|---|---|
| fear | 0 | 7 (p = 0.015) | 3.1 | 13 | ≥ 8, p < 0.05 | **fail** |
| anger | 1 | 11 (p < 0.001) | 3.1 | 13 | ≥ 7, p < 0.05 | pass |
| sadness | 0 | 7 (p = 0.025) | 3.2 | 13 | ≥ 7, p < 0.05 | pass (reported, not deciding) |

The LLM layer loses nothing (oracle 13 of 13); every miss is the tagger's, and
12 of its 15 misses are **neutral clips tagged emotional** — a
speaker-independent tagger does not know how this caller sounds when calm.
The 39 test pairs are now spent for this family of taggers.

### R3 on the test pairs — PR-006 (2026-09-22)

`src/s2slab/r3probe.py` and `modal_app/r3_probe.py`: Qwen2-Audio-7B-Instruct
on a Modal A10 is given each clip and asked *"Is the person speaking upset?"*
— no transcript, no tag — and its p(Yes) is read off one forward pass, as for
the side-channel. Full registration and results: `docs/preregistration.md`,
PR-006. Evidence: `runs/prosody/pr006.json`.

**The three points on the plan's axis**, same 39 pairs, same question
(SAD · FEA · ANG, of 13):

| | answered right | ranked right |
|---|---|---|
| R1 blind | 0 · 0 · 1 | 3 · 5 · 8 |
| R1 + $0 affect tag | 7 · 7 · 11 | 10 · 8 · 13 |
| R3, hears the audio | 11 · 11 · 9 | **13 · 13 · 13** |

R3 hears the difference in every pair (Claim A, supported) and every one of its
8 misses on the strict metric is a calm clip it calls upset (Claim B, not
supported: anger 9 against the tag's 11, sign test p = 0.21). It also
transcribed all 39 neutral clips word for word. The 39 pairs are now spent for
R3 as well; whether a perfect ranking is hearing or memory of CREMA-D is what
the own-voice mini set is for.

## Own-voice mini set (2026-09-22)

Ten takes by the author: the plan's lexically neutral phrases, each once calm
and once upset. It exists for one reason. R3 ranked the upset clip above its
calm twin on 39 of 39 CREMA-D pairs (PR-006), and CREMA-D is a standard emotion
benchmark whose presence in Qwen2-Audio's training data cannot be ruled out, so
a perfect ranking is consistent with hearing and with familiarity. These takes
are audio no model has seen -- a fresh voice, a fresh microphone, fresh
sentences. Five pairs cannot prove the CREMA-D result is clean; they can show
whether the ranking survives on them. They also give the tagger, fitted on 16
CREMA-D actors and one microphone, its first genuinely new speaker.

**Dev-only.** Nothing is registered, the audio is never committed (`*.wav` is
ignored), and whatever comes out is reported as descriptive. Code:
`src/s2slab/ownvoice.py`; R3 side: `modal run modal_app/r3_probe.py::own`.

### The takes

| take | phrase | delivery |
|---|---|---|
| `own_p1_CAL` / `own_p1_SAD` | I'm fine. | calm / sad |
| `own_p2_CAL` / `own_p2_ANG` | That's great. | calm / angry (bitter) |
| `own_p3_CAL` / `own_p3_ANG` | Sure, whatever you say. | calm / angry |
| `own_p4_CAL` / `own_p4_FEA` | I'll be there at six. | calm / fearful |
| `own_p5_CAL` / `own_p5_SAD` | Okay, I'll do that. | calm / sad |

Recorded in one session in Audacity at 16 kHz mono, one file per take,
exported as 16-bit PCM WAV. The gain was set on the loudest delivery first
(angry, p3, peaks near -6 dBFS) and not touched again, nor the microphone or
the distance to it, so a level difference between two takes is the voice's,
not the knob's. The five calm takes were recorded first, back to back, then
the five upset ones; one take each; no trimming, normalizing, noise reduction
or fades. The delivery cues came from the CREMA-D dev medians in "First look"
above: sad softer and slower with more pitch movement, fearful much higher and
louder, angry louder and clipped. Microphone and room: (fill in).

As recorded: 5.8-7.8 s per file with 0.8-2.3 s of speech inside, peaks -18.2
to -0.1 dBFS, no clipped samples, and the click of the record button at the
start of several. CREMA-D's clips are 1.5-4.1 s with little silence, and the
eGeMAPS functionals are taken over the whole clip, so every take is
**windowed in code** -- the longest run of 20 ms frames within 30 dB of the
take's loudest, runs closer than 0.5 s merged, 0.3 s of room tone kept on
each side -- identically for all ten, with the files on disk untouched. Both
the windowed and the as-recorded audio go to R3 and to the tagger; the
windowed one is primary. (`ownvoice look` prints the windows; p3's angry take
keeps a breath before the words and so starts at 0.00 s. The rule was not
adjusted for it.)

### The reading, fixed before any run

Chance ranking of 5 pairs is 1 in 32. R3 ranks 4 or 5 of 5 right -> "hearing
generalizes to a fresh voice"; 3 or fewer -> "the CREMA-D ranking may lean on
familiarity; this set cannot say which". Predictions, to be scored: R3 ranks 5
of 5 and answers 3 of 5 strictly right (its Yes-lean catches a calm take or
two); the tagger tags 3-4 of the 5 upset takes non-neutral and 2-3 of the 5
calm takes neutral.

### What the takes measure as (`ownvoice look`, windowed)

| pair | phrase | upset | pitch (st) | pitch range (st) | level (dB) |
|---|---|---|---:|---:|---:|
| sad-o01 | I'm fine | SAD | +1.4 | -4.5 | -0.8 |
| ang-o02 | That's great | ANG | +0.5 | -3.8 | +0.9 |
| ang-o03 | Sure, whatever you say | ANG | +1.6 | +3.7 | +9.7 |
| fea-o04 | I'll be there at six | FEA | +8.5 | -4.8 | +15.5 |
| sad-o05 | Okay, I'll do that | SAD | +0.5 | +0.4 | +6.6 |

Calm takes sit at 97-114 Hz and -21 to -31 dB. Fear moved as CREMA-D's actors
did (+8.5 st, +15.5 dB against their +9.3 st, +10.3 dB); one angry take did
(+9.7 dB) and the other, the bitter "That's great", barely moved at all; the
two sad takes did not get softer (-0.8 and +6.6 dB against CREMA-D's -2.0).
One untrained speaker, one attempt each -- the set is what it is.

### The tagger on the five pairs (2026-09-22)

Windowed: told apart SAD 0 of 2, FEA 1 of 1, ANG 1 of 2; both right only
ang-o03. Every upset take was tagged **angry** -- the sad ones too -- and so
were three of the five calm takes (p1 at 0.99, p2 at 1.00, p5 at 0.59); only
p3 and p4 calm were tagged neutral. As recorded, all ten takes are tagged angry.

Scored against the predictions: "3-4 of 5 upset takes non-neutral" -> 5 of 5,
over; "2-3 of 5 calm takes neutral" -> 2 of 5, at the bottom of the range.
The reading is the one PR-005 already gave, only louder: a speaker-independent
tagger does not know what this voice sounds like when nothing is wrong, and on
a new voice through a new microphone its baseline is off before the first
word -- here, towards angry. The per-caller baseline (parked) is the fix; the
window matters too (with the room tone in, nothing at all is told apart).

### R3 on the five pairs (2026-09-22)

`modal run modal_app/r3_probe.py::own --save runs/prosody/ownvoice-r3.json`,
Modal A10, 42 s in the function, about \$0.014; stamp `4c5905d-dirty` because
this section's files were in the tree uncommitted when the call was made
(dev-only, nothing registered; the run file is committed with them). Evidence:
`runs/prosody/ownvoice-r3.json` (every p(Yes) under every wording, windowed and
as recorded, and every transcript) and `runs/prosody/ownvoice-r3-report.md`.

Windowed, the registered wording *"Is the person speaking upset?"*:

| pair | p(Yes) calm | p(Yes) upset | ranked | answered |
|---|---:|---:|---|---|
| sad-o01 | 0.01 | 0.01 | wrong | wrong |
| ang-o02 | 0.01 | 0.02 | right | wrong |
| ang-o03 | 0.36 | 0.50 | right | wrong |
| fea-o04 | 0.39 | 0.77 | right | right |
| sad-o05 | 0.90 | 0.85 | wrong | wrong |

**Ranked right 3 of 5, answered right 1 of 5.** By the reading fixed above:
*the CREMA-D ranking may lean on familiarity; this set cannot say which.*
Predictions scored: "ranks 5 of 5" -> 3 of 5, wrong; "answers 3 of 5" -> 1,
wrong. (The tagger's two predictions were scored in the section above: one
over, one at the bottom of its range.)

What the run shows besides, descriptively:

- **The Yes-lean is gone.** On CREMA-D the upset clips got p(Yes) 0.94 to
  1.00 and 47 of 78 clips got Yes; here the five upset takes got 0.01, 0.02,
  0.50, 0.77 and 0.85, and 3 of 10 clips got Yes. Whatever made the model say
  Yes to nearly every acted clip, it does not do it to this voice.
- **The ranking follows the acoustics.** The pairs ranked right are the ones
  whose upset take moved in the `look` table (fear +8.5 st and +15.5 dB;
  ang-o03 +9.7 dB; ang-o02 barely, and ranked right by 0.01). The two sad
  pairs, which never got softer or slower, are the two ranked wrong -- and the
  tagger failed on exactly those two. That is what hearing the delivery would
  look like, and it is also what five pairs from one untrained speaker look
  like; the set cannot separate them.
- **Wording matters on a fresh voice as it did not on CREMA-D.** Ranked
  right, windowed: `not_words` 5 of 5, `tone` 4, `upset` 3, `emotion` 3. On
  the 21 dev pairs every wording ranked 21 of 21. Reported, not read: picking
  `not_words` now would be picking after seeing.
- **The window mattered for R3 too:** as recorded, `upset` ranks 2 of 5.
- The calm "Okay, I'll do that" is heard as upset by both systems (R3 0.90,
  the tagger angry at 0.59). One take, one listener; it may be the take.
- R3 transcribed all ten takes word for word.

**Where this leaves it.** M16's Claim A (13 · 13 · 13 on CREMA-D) stands as
registered; what this set adds is that the same model, same question, ranks a
fresh voice at 3 of 5 with no Yes-lean, so the CREMA-D figure should not be
read as "R3 hears upset in anyone". Settling familiarity against hearing needs
more speakers, or acted pairs from a corpus published after the model -- both
parked.

## The per-caller baseline, and the fresh pairs — PR-007 (2026-09-23)

`src/s2slab/baseline.py`. The affect tagger above is speaker-independent, and
its misses were calm clips tagged emotional (PR-005: 12 of 15; the author's
own calm voice: angry at 0.99). The per-caller baseline gives it the missing
piece without changing the model: every clip's 88 eGeMAPS features minus the
mean over *the same speaker's other neutral clips* (`probes/prosody/baseline/`
for the dev actors, 9–11 clips each; a training clip against its actor's other
neutral training clips). Saved as `probes/prosody/tagger-relative.json`.

**On the 21 dev pairs it looked like a win** — told apart 11 → 16 of 21,
sadness 1 → 5, held-out training actors' neutral clips right 69% → 81% —
so it was registered as PR-007. Not on the 39 locked pairs: the hypothesis
came from their misses, so they are spent for the tagger.

### The fresh pairs (`probes/prosody2/`)

The original rule applied to the 32 test-side actors of the original seeded
split that have no pair in `probes/prosody` — voices no system here had
scored. Per emotion, the highest threshold that fills 13 pairs, else the
lowest threshold and what it gives:

| raters agreeing | SAD (of 13) | FEA (of 13) | ANG (of 13) |
|---|---:|---:|---:|
| at least 90% | 1 | 0 | 3 |
| at least 80% | 1 | 6 | **13** |
| at least 70% | 4 | 12 | · |
| at least 60% | 8 | **13** | · |
| at least 50% | **12** | · | · |

38 pairs from 27 actors, ids `sad-f01` … `ang-f13`, split `test`, locked
behind `--confirm PR-007` by the same `prosodyset.load` lock. Thresholds are
lower than the original set's (60 / 80 / 90) because the clearer actors were
drawn first. Each fresh actor's baseline: their neutral clips heard as
neutral that are in no pair, 268 in all, 6–11 each
(`probes/prosody2/baseline/`). Layout as `probes/prosody/`: `pairs.tsv`,
`manifest.json`, `baseline/baseline.tsv`, `baseline/manifest.json` committed;
audio ignored. Rebuild: `baseline fresh`, then `baseline fetch-fresh`.

### The result — PR-007 not supported

Registered before any fresh clip was fetched; full text and verbatim output in
`docs/preregistration.md`, PR-007; evidence `runs/prosody/pr007.json`.

| 38 fresh pairs | told apart SAD · FEA · ANG | told apart, all | loaded clip tagged non-neutral |
|---|---|---:|---:|
| absolute tagger | 8 · 8 · 10 | **26 of 38** | 33 |
| relative tagger | 8 · 9 · 9 | **26 of 38** | 32 |

Fixed 2, broken 2, one-sided sign test p = 0.688. The dev gain did not
transfer at all. Both taggers miss the same way — a calm clip tagged
emotional (mostly "sad") in 7 pairs, the upset clip tagged neutral in 5 — and
the baseline size curve (1 / 3 / 5 / all clips: 23 / 24 / 25 / 26 told apart)
only climbs back to even. The reading: on actors recorded in one studio, the
tagger's calm-clip misses are clip-level, not a speaker offset a baseline can
remove. The absolute tagger, meanwhile, replicated PR-005's ceiling on a
second sample: 68% told apart against 62%.

What was wrong in the registration: dev's 7 fixed / 2 broken on 14 actors was
read as a rate (it is p = 0.09 one-sided), and "lower agreement means harder
pairs" was assumed for the tagger and was false. The one place a baseline
still plausibly helps — a new voice through a new microphone, the author's
five pairs, calm tagged neutral 2 → 3 of 5 — is one speaker and dev-only.

### Does R4's voice react? PR-008 (2026-09-23)

`src/s2slab/r4probe.py`, `modal_app/r4_bench.py::pairs`. The prosody pairs go
in as the caller's turn; R4 (Qwen2.5-Omni, speaking for itself) and R3
(Qwen2-Audio, Kokoro speaking its text) answer both clips of each pair, and
the F0 range of the reply to the upset clip is compared with the reply to the
calm clip: wider, narrower, or the same within +-0.5 st. Full registration
and results: `docs/preregistration.md`, PR-008; evidence
`runs/prosody/pr008.json`.

| 39 test pairs | wider | same | narrower |
|---|---:|---:|---:|
| R3 (Qwen2-Audio -> Kokoro) | 12 | 21 | 6 |
| R4 (Qwen2.5-Omni speaks) | 14 | 9 | 16 |

**Not supported, and no consistent direction** (two-sided p = 0.86). R4's
voice moves more than Kokoro's -- only 9 of 39 pairs inside the band -- but
either way, and its words do not change. On dev it had looked like
flattening (5 against 11); on the test pairs that did not survive. By the
plan's own sentence, on this measure the rung bought nothing. The 39 pairs
are now spent for R4's delivery as well.
