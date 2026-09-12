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
   - about 1 second of silence,
   - say the line; at each `|`, **pause for about one second** (silently count
     "one-and-two"),
   - finish the sentence, then **stay silent for at least 2 seconds** before
     stopping.

   The 2 seconds matter. The slowest policy needs 900 ms of silence to decide.
   Stop too early and it never fires, and that item stops measuring anything.
4. **Natural items** have no `|` marks. Don't plan pauses; say it as you would
   on the phone. You'll mark where you actually paused when you label.
5. **Export** each item as WAV (signed 16-bit PCM), named by its id:
   `endpointing/audio/f01.wav`, `c01.wav`, `n01.wav`, and so on.

### Pilot first

Record **f01, c01 and n01 only**, then stop. The loader and the labelling tool
get built and checked against those three before you spend an afternoon on the
other 27. A wrong sample rate or a clipped microphone is cheap to find at three
items and expensive at thirty.

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
