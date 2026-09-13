# Probe sets

Hand-recorded audio for testing the policies on **real speech**, from the v2
Engineering plan (Notion page 17, §6.3). The simulator can show how a policy
behaves when pauses fall where we put them. Only real recordings show what real
acoustics, real pitch and a real VAD do at those pauses.

| Set | Items | Serves | When |
|---|---:|---|---|
| **Endpointing** | 30 | Turn-taking, T0–T3 and PR-001 | **Now** |
| Prosody | 30 | R3 speech-LLM probe (sincere vs distressed delivery) | Before R3 |
| Barge-in | 20 | R1 barge-in (genuine interruption vs backchannel) | Before R1 |

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

- Microphone: *(fill in)*
- Room: *(fill in)*
- Date recorded: *(fill in)*
