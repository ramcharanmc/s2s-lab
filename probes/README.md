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
