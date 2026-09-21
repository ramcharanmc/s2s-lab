"""
The prosody probe set: the same words from the same person, said neutrally and said with feeling.

The v2 plan, section 6.3: "Neutral words, loaded delivery ... R1 cannot distinguish
the pairs by construction -- this is the input bottleneck, demonstrated rather
than argued." A cascade hands its LLM a transcript, and both clips of a pair
have the same transcript.

    one item = one PAIR    actor 1042, "I think I have a doctor's appointment"
                           neutral  1042_ITH_NEU_XX    raters: neutral, 10 of 11
                           loaded   1042_ITH_FEA_XX    raters: fear,     9 of 10

SOURCE: CREMA-D (Cao et al., IEEE Trans. Affective Computing 2014;
github.com/CheyneyComputerScience/CREMA-D), pinned to one commit. 91 actors
speak 12 sentences whose words carry no emotion, each in six acted emotions,
and 2,443 raters said what they HEARD. Open Database License for the database,
Database Contents License for its contents. The author of this repo records
nothing here: a label that is a perception needs listeners, and these clips
come with theirs.

"DISTRESS" IS A PROXY, THREE TIMES OVER. CREMA-D has no such category. Sad,
fearful and angry delivery are taken as three separate stand-ins, reported one
by one and never pooled. It is acted emotion by consenting actors; nothing here
claims clinical validity or diagnoses anyone (v1 plan, ethics boundary). Each
pair keeps the raters' mean intensity for the loaded clip, a number, beside
the label.

SELECTED BY RULE, FROM THE RATINGS ALONE -- nobody listens first.
  1. voice-only ratings (the raters who heard the clip and saw no face)
  2. a pair qualifies if the neutral clip was heard as neutral and the loaded
     clip as its intended emotion, each by at least `threshold` of at least
     MIN_RATERS raters
  3. actors are split into dev and test BEFORE any pair is drawn, so no voice
     is on both sides; the test pairs are locked like the other test sets
  4. the threshold is the HIGHEST of THRESHOLDS at which every emotion fills
     both sides -- decided by counting, reported, never by ear
  5. a seeded draw: one pair per actor per emotion, no neutral clip twice, at
     most PER_SENTENCE pairs of one sentence per emotion and side

Only the chosen clips are fetched (about 50 KB each); the full corpus is 7.5 GB.
Audio and the ratings file stay out of git; pairs.tsv, the rule's output, is committed.

    python -m s2slab.prosodyset ratings      fetch the two ratings files (4.5 MB), once
    python -m s2slab.prosodyset select       count, choose, write probes/prosody/pairs.tsv
    python -m s2slab.prosodyset audio        fetch the chosen clips
"""

from __future__ import annotations

import csv
import pathlib
import random
import urllib.request
from typing import NamedTuple

PROSODY_ROOT = pathlib.Path("probes/prosody")
REPO = "CheyneyComputerScience/CREMA-D"
REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"      # master on 2026-09-21; last changed 2025-03-21
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}/"    # the WAVs are git-lfs objects
RATINGS, SUMMARY = "processedResults/tabulatedVotes.csv", "processedResults/summaryTable.csv"

SEED = 20260921                       # fixed before any rating was read
NEUTRAL = "NEU"
EMOTIONS = ("SAD", "FEA", "ANG")      # the three proxies, always reported separately
VOTE = {"NEU": "N", "SAD": "S", "FEA": "F", "ANG": "A"}
INTENSITY = {"SAD": "meanSadResp", "FEA": "meanFearResp", "ANG": "meanAngerResp"}
VOICE_IDS = range(100_001, 200_000)   # tabulatedVotes numbers voice-only ratings 1xxxxx, face 2xxxxx, both 3xxxxx
THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5)
MIN_RATERS = 8
DEV_ACTORS = 30                       # of 91
PER_EMOTION = {"dev": 7, "test": 13}  # x 3 emotions = 21 dev pairs + 39 test pairs
PER_SENTENCE = {"dev": 2, "test": 3}
PAIR_COLUMNS = ("id", "split", "emotion", "actor", "sentence", "neutral", "loaded",
                "agree_neutral", "agree_loaded", "raters_neutral", "raters_loaded", "intensity")


class Rating(NamedTuple):
    """What the voice-only raters said about one clip."""

    clip: str             # 1042_ITH_FEA_XX
    actor: str
    sentence: str
    emotion: str          # what the actor was asked for
    vote: str             # what most raters heard: one letter, or a tie such as "N:S"
    agreement: float      # the share of raters who gave the majority answer
    raters: int
    intensity: float      # mean 0-100 intensity from raters who heard the INTENDED emotion; -1 if none did

    def heard_as_intended(self, threshold: float) -> bool:
        return self.vote == VOTE[self.emotion] and self.agreement >= threshold and self.raters >= MIN_RATERS


class Pair(NamedTuple):
    id: str               # sad-t01: emotion, side, number
    split: str            # dev | test
    emotion: str
    actor: str
    sentence: str
    neutral: str          # clip names, without .wav
    loaded: str
    agree_neutral: float
    agree_loaded: float
    raters_neutral: int
    raters_loaded: int
    intensity: float


# --------------------------------------------------------------------------- #
# Reading the ratings
# --------------------------------------------------------------------------- #


def read_ratings(path: pathlib.Path) -> dict[str, Rating]:
    """{clip name: Rating} for the voice-only ratings of the emotions this set uses.

    Takes:  CREMA-D's processedResults/tabulatedVotes.csv.
    Raises: ValueError if a column the rule reads is missing -- the file is not what it was.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        missing = {"fileName", "numResponses", "agreement", "emoVote", *INTENSITY.values()} - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: no column {sorted(missing)}")
        for row in rows:
            if int(row[""]) not in VOICE_IDS:
                continue
            actor, sentence, emotion, _level = row["fileName"].split("_")
            if emotion not in VOTE:
                continue
            intensity = float(row[INTENSITY[emotion]]) if emotion in INTENSITY else -1.0
            out[row["fileName"]] = Rating(row["fileName"], actor, sentence, emotion, row["emoVote"],
                                          float(row["agreement"]), int(row["numResponses"]), intensity)
    return out


def read_summary(path: pathlib.Path) -> dict[str, str]:
    """{clip name: VoiceVote} from summaryTable.csv: CREMA-D's own statement of
    what the voice-only raters heard, used to cross-check read_ratings()."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["FileName"]: row["VoiceVote"] for row in csv.DictReader(f)}


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #


def candidates(ratings: dict[str, Rating], threshold: float) -> list[tuple[Rating, Rating]]:
    """Every (neutral, loaded) pair that qualifies at this threshold, sorted.

    Same actor, same sentence. Where an actor said a sentence at several
    intensities (only "It's eleven o'clock" has low / medium / high), the one
    the raters agreed on most is the candidate; ties go to the first by name.
    """
    best: dict[tuple[str, str, str], Rating] = {}
    for r in sorted(ratings.values()):
        if r.emotion in EMOTIONS and r.heard_as_intended(threshold):
            key = (r.actor, r.sentence, r.emotion)
            if key not in best or r.agreement > best[key].agreement:
                best[key] = r
    out = []
    for (actor, sentence, _emotion), loaded in sorted(best.items()):
        neutral = ratings.get(f"{actor}_{sentence}_{NEUTRAL}_XX")
        if neutral is not None and neutral.heard_as_intended(threshold):
            out.append((neutral, loaded))
    return out


def split_actors(actors, seed: int = SEED) -> dict[str, str]:
    """{actor: "dev" | "test"}: a seeded shuffle of ALL actors, the first
    DEV_ACTORS to dev. Done before any pair is drawn and without looking at a
    rating, so which voices are test voices depends on nothing but the seed."""
    order = sorted(actors)
    random.Random(seed).shuffle(order)
    return {a: ("dev" if k < DEV_ACTORS else "test") for k, a in enumerate(order)}


def draw(cands: list[tuple[Rating, Rating]], sides: dict[str, str], seed: int = SEED) -> list[Pair]:
    """A seeded draw from the candidates, as many as the rule allows up to PER_EMOTION.

    One pair per actor per emotion; no neutral clip in two pairs; at most
    PER_SENTENCE pairs of one sentence per emotion and side. Emotions are
    drawn in EMOTIONS order from one shuffled list, so the result depends only
    on the candidates and the seed.
    """
    order = list(cands)
    random.Random(seed).shuffle(order)
    used_neutral: set[str] = set()
    pairs = []
    for emotion in EMOTIONS:
        for split in ("dev", "test"):
            actors, per_sentence, n = set(), {}, 0
            for neutral, loaded in order:
                if n == PER_EMOTION[split]:
                    break
                if (loaded.emotion != emotion or sides[loaded.actor] != split or loaded.actor in actors
                        or neutral.clip in used_neutral
                        or per_sentence.get(loaded.sentence, 0) >= PER_SENTENCE[split]):
                    continue
                n += 1
                actors.add(loaded.actor)
                used_neutral.add(neutral.clip)
                per_sentence[loaded.sentence] = per_sentence.get(loaded.sentence, 0) + 1
                pairs.append(Pair(f"{emotion.lower()}-{split[0]}{n:02d}", split, emotion, loaded.actor,
                                  loaded.sentence, neutral.clip, loaded.clip, round(neutral.agreement, 3),
                                  round(loaded.agreement, 3), neutral.raters, loaded.raters,
                                  round(loaded.intensity, 1)))
    return pairs


def filled(pairs: list[Pair]) -> dict[tuple[str, str], int]:
    """{(emotion, split): pairs drawn}."""
    out = {(e, s): 0 for e in EMOTIONS for s in ("dev", "test")}
    for p in pairs:
        out[(p.emotion, p.split)] += 1
    return out


def select(ratings: dict[str, Rating], seed: int = SEED) -> tuple[float, list[Pair], dict[float, dict]]:
    """Apply the rule. Returns the threshold used, the pairs, and for every
    threshold tried {(emotion, split): pairs the draw could fill} -- the
    counting that chose the threshold, kept so it can be reported.

    Raises: ValueError, with the counts, if no threshold fills every cell.
    """
    sides = split_actors({r.actor for r in ratings.values()}, seed)
    counts = {}
    for threshold in THRESHOLDS:
        pairs = draw(candidates(ratings, threshold), sides, seed)
        counts[threshold] = filled(pairs)
        if all(counts[threshold][(e, s)] == PER_EMOTION[s] for e in EMOTIONS for s in ("dev", "test")):
            return threshold, pairs, counts
    raise ValueError("no threshold fills every emotion on both sides:\n" + format_counts(counts))


def format_counts(counts: dict[float, dict], chosen: float | None = None) -> str:
    """The counting behind the threshold, as a table."""
    cells = [(e, s) for e in EMOTIONS for s in ("dev", "test")]
    lines = ["| raters agreeing | " + " | ".join(f"{e} {s} (of {PER_EMOTION[s]})" for e, s in cells) + " | |",
             "|---|" + "---:|" * len(cells) + "---|"]
    for threshold, got in counts.items():
        lines.append(f"| at least {threshold:.0%} | " + " | ".join(str(got[c]) for c in cells)
                     + (" | **chosen** |" if threshold == chosen else " | |"))
    return "\n".join(lines)


def cross_check(pairs: list[Pair], summary: dict[str, str]) -> None:
    """CREMA-D's summary table must say the same about every chosen clip. If it
    does not, read_ratings() is reading the wrong rows (not the voice-only ones)."""
    for p in pairs:
        for clip, expected in ((p.neutral, VOTE[NEUTRAL]), (p.loaded, VOTE[p.emotion])):
            if summary.get(clip) != expected:
                raise ValueError(f"{clip}: chosen as heard '{expected}', but summaryTable.csv says "
                                 f"VoiceVote = {summary.get(clip)!r}")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def write_pairs(pairs: list[Pair], path: pathlib.Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the selection is made once. Delete it yourself to select again.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(PAIR_COLUMNS)
        w.writerows(pairs)


def read_pairs(path: pathlib.Path) -> list[Pair]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Pair(r["id"], r["split"], r["emotion"], r["actor"], r["sentence"], r["neutral"], r["loaded"],
                     float(r["agree_neutral"]), float(r["agree_loaded"]), int(r["raters_neutral"]),
                     int(r["raters_loaded"]), float(r["intensity"]))
                for r in csv.DictReader(f, delimiter="\t")]


def fetch(url: str, dest: pathlib.Path) -> bool:
    """Download url to dest unless dest exists. Returns True if it downloaded."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    dest.write_bytes(data)
    return True


def fetch_ratings(root: pathlib.Path = PROSODY_ROOT) -> list[pathlib.Path]:
    """The two ratings files, into root/source/ (ignored by git). 4.5 MB, once."""
    paths = []
    for name in (RATINGS, SUMMARY):
        paths.append(root / "source" / pathlib.Path(name).name)
        fetch(RAW + name, paths[-1])
    return paths


def fetch_audio(pairs: list[Pair], root: pathlib.Path = PROSODY_ROOT) -> int:
    """Only the chosen clips, into root/audio/. Returns how many were downloaded now."""
    clips = sorted({c for p in pairs for c in (p.neutral, p.loaded)})
    return sum(fetch(f"{MEDIA}AudioWAV/{clip}.wav", root / "audio" / f"{clip}.wav") for clip in clips)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    source = PROSODY_ROOT / "source"
    if cmd == "ratings":
        for path in fetch_ratings():
            print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")
    elif cmd == "select":
        ratings = read_ratings(source / pathlib.Path(RATINGS).name)
        threshold, pairs, counts = select(ratings)
        cross_check(pairs, read_summary(source / pathlib.Path(SUMMARY).name))
        print(f"{len(ratings)} voice-only ratings of neutral, sad, fearful and angry clips; "
              f"{len({r.actor for r in ratings.values()})} actors; seed {SEED}\n")
        print(format_counts(counts, threshold))
        write_pairs(pairs, PROSODY_ROOT / "pairs.tsv")
        print(f"\n{len(pairs)} pairs -> {PROSODY_ROOT / 'pairs.tsv'}; every chosen clip agrees with summaryTable.csv")
        for emotion in EMOTIONS:
            mine = [p for p in pairs if p.emotion == emotion]
            print(f"  {emotion}: loaded clips heard as intended by {min(p.agree_loaded for p in mine):.0%} to "
                  f"{max(p.agree_loaded for p in mine):.0%} of raters; {len({p.sentence for p in mine})} "
                  f"of 12 sentences; intensity {min(p.intensity for p in mine):.0f} to "
                  f"{max(p.intensity for p in mine):.0f} of 100")
    elif cmd == "audio":
        pairs = read_pairs(PROSODY_ROOT / "pairs.tsv")
        print(f"downloaded {fetch_audio(pairs)} clips; {2 * len(pairs)} are in {PROSODY_ROOT / 'audio'}")
    else:
        sys.exit("usage: python -m s2slab.prosodyset ratings | select | audio")