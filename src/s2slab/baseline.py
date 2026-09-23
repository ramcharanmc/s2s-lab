"""
The per-caller baseline: the same tagger, told what this voice sounds like when nothing is wrong.

WHY. The affect tagger (affect.py) is speaker-independent: it calls a clip angry
without knowing how that person usually sounds. PR-005 showed where that
breaks -- 12 of its 15 misses were CALM clips tagged emotional -- and M17 made
it plain: the author's calm voice, through a new microphone, was tagged angry
at 0.99. This file gives the tagger the missing piece: the same speaker's
other neutral clips.

WHAT. Nothing about the model changes -- the 88 eGeMAPS functionals, the
class-balanced softmax regression, the 16 training actors -- except its input:

    x_relative = x - mean(eGeMAPS of the speaker's BASELINE clips)

"how far is this from this person's calm voice", delta only, so the relative
tagger and the absolute one are the same size and compared one for one. A
speaker's baseline clips are their neutral clips that the voice-only raters
heard as neutral (majority vote N), never a clip that is in a pair. When the
clip being tagged is itself a baseline clip (a neutral training clip), the
mean is taken over the OTHERS. The deployment analogue is a caller's earlier
calm turns; the baseline-size curve on the dev pairs says how many it takes.

WHO SUPPLIES THE BASELINE.
    16 training actors    their neutral training clips, already fetched (affect.py)
    14 dev-pair actors    9-11 clips each, fetched into probes/prosody/baseline/ -- for developing on the 21 dev pairs
    the PR-007 actors     come with the fresh set (probes/prosody2), never seen by anything before that run

WHY FRESH PAIRS. This hypothesis was read off PR-005's test misses. Confirming it
on the same 39 pairs would be confirming a post-hoc hypothesis on the data that
produced it; those pairs are spent for the tagger. PR-007 is registered on
pairs drawn from the 32 test-side actors nothing has used, and this file never
touches the 39 (the test split of probes/prosody is not read here).

    python -m s2slab.baseline clips    which baseline clips: the dev-pair actors' neutral clips -> baseline.tsv
    python -m s2slab.baseline fetch    fetch them (about 7 MB), write their manifest
    python -m s2slab.baseline train    the relative tagger on the training clips, held-out actors, save tagger-relative.json
    python -m s2slab.baseline dev      both taggers on the 21 DEV pairs, and told-apart against baseline size
    python -m s2slab.baseline own      the author's five pairs, each calm take's baseline = the other four
"""

from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

from .affect import (CACHE, LABELS, TAGGER, TRAIN_CLIPS, TRAIN_ROOT, Tagger, features_for, fit, leave_one_actor_out,
                     read_training_clips, tag_pairs, verify_training)
from .probes import sha256_file
from .prosodyset import MEDIA, PAIRS, PROSODY_ROOT, REPO, REVISION, Rating, fetch, read_pairs

BASELINE_ROOT = PROSODY_ROOT / "baseline"
BASELINE_CLIPS, BASELINE_MANIFEST = "baseline.tsv", "manifest.json"
RELATIVE_TAGGER = PROSODY_ROOT / "tagger-relative.json"     # the fitted relative model: committed
NEUTRAL_VOTE = "N"
CURVE = (1, 3, 5)                                           # baseline sizes tried on dev, besides "all"


# --------------------------------------------------------------------------- #
# Which clips make a baseline
# --------------------------------------------------------------------------- #


def baseline_clips(ratings: dict[str, Rating], pairs, actors) -> list[Rating]:
    """The baseline material of some actors, sorted by clip name.

    Takes:   the voice-only ratings; the probe set's pairs; the actors wanted.
    Returns: those actors' NEUTRAL clips that most raters heard as neutral,
             leaving out any clip that is in a pair (a pair's clips are the
             things being tagged, never the baseline).
    """
    actors, in_pairs = set(actors), {c for p in pairs for c in (p.neutral, p.loaded)}
    return sorted(r for r in ratings.values()
                  if r.actor in actors and r.emotion == "NEU" and r.vote == NEUTRAL_VOTE and r.clip not in in_pairs)


def write_baseline_clips(clips: list[Rating], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(("clip", "actor", "sentence", "agreement", "raters"))
        w.writerows((r.clip, r.actor, r.sentence, round(r.agreement, 3), r.raters) for r in clips)


def read_baseline_clips(path: pathlib.Path) -> list[tuple[str, str]]:
    """[(clip name, actor)] from baseline.tsv."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["clip"], r["actor"]) for r in csv.DictReader(f, delimiter="\t")]


def fetch_baseline(root: pathlib.Path = BASELINE_ROOT) -> int:
    """Fetch the clips in baseline.tsv from the pinned CREMA-D commit into root/audio/,
    then write the manifest (their SHA-256). Returns how many were downloaded now."""
    root = pathlib.Path(root)
    clips = [c for c, _ in read_baseline_clips(root / BASELINE_CLIPS)]
    n = sum(fetch(f"{MEDIA}AudioWAV/{c}.wav", root / "audio" / f"{c}.wav") for c in clips)
    (root / BASELINE_MANIFEST).write_text(json.dumps({
        "set": "prosody-baseline", "source": {"repo": REPO, "revision": REVISION},
        "files": {BASELINE_CLIPS: sha256_file(root / BASELINE_CLIPS, text=True)},
        "clips": {c: sha256_file(root / "audio" / f"{c}.wav") for c in clips}}, indent=2) + "\n", encoding="utf-8")
    return n


def verify_baseline(root: pathlib.Path = BASELINE_ROOT) -> list[str]:
    """One line per baseline clip that is missing or differs from the manifest."""
    root = pathlib.Path(root)
    frozen = json.loads((root / BASELINE_MANIFEST).read_text(encoding="utf-8"))
    problems = ([] if sha256_file(root / BASELINE_CLIPS, text=True) == frozen["files"][BASELINE_CLIPS]
                else [f"{BASELINE_CLIPS}: changed"])
    for clip, h in frozen["clips"].items():
        path = root / "audio" / f"{clip}.wav"
        if not path.exists():
            problems.append(f"audio/{clip}.wav: missing")
        elif sha256_file(path) != h:
            problems.append(f"audio/{clip}.wav: changed")
    return problems


# --------------------------------------------------------------------------- #
# The baseline, and features relative to it
# --------------------------------------------------------------------------- #


class Baseline:
    """Every actor's baseline clips' feature vectors, by clip name.

    Takes:   the baseline clips, their actors, their features (one row each).
    Gives:   mean_for(actor, exclude=clip) -- the mean over that actor's baseline
             clips, leaving `exclude` out when it is one of them, so a neutral
             training clip is never measured against itself.
    """

    def __init__(self, clips: list[str], actors: list[str], X: np.ndarray):
        self.rows: dict[str, dict[str, np.ndarray]] = {}
        for clip, actor, x in zip(clips, actors, np.asarray(X, dtype=np.float64)):
            self.rows.setdefault(actor, {})[clip] = x

    def mean_for(self, actor: str, exclude: str | None = None) -> np.ndarray:
        if actor not in self.rows:
            raise ValueError(f"no baseline clips for actor {actor}")
        kept = [x for clip, x in self.rows[actor].items() if clip != exclude]
        if not kept:
            raise ValueError(f"actor {actor} has no baseline clip other than {exclude}")
        return np.mean(kept, axis=0)

    def size(self, actor: str) -> int:
        return len(self.rows.get(actor, {}))


def make_baseline(clips: list[str], actors: list[str], X: np.ndarray) -> Baseline:
    return Baseline(clips, actors, X)


def relative_features(X: np.ndarray, clips: list[str], actors: list[str], baseline: Baseline) -> np.ndarray:
    """Each row minus its speaker's baseline mean -- the mean of the OTHER baseline
    clips when the row is one of them. Raises for a speaker without a baseline."""
    return np.vstack([np.asarray(x, dtype=np.float64) - baseline.mean_for(a, exclude=c)
                      for x, c, a in zip(X, clips, actors)])


def first_k(clips: list[str], actors: list[str], k: int) -> tuple[list[str], list[str]]:
    """The first k baseline clips of every actor, in clip-name order (deterministic, not chosen)."""
    keep, seen = [], {}
    for c, a in sorted(zip(clips, actors)):
        if seen.get(a, 0) < k:
            keep.append((c, a))
            seen[a] = seen.get(a, 0) + 1
    return [c for c, _ in keep], [a for _, a in keep]


# --------------------------------------------------------------------------- #
# Training, and the dev looks
# --------------------------------------------------------------------------- #


def train_relative(train_root: pathlib.Path = TRAIN_ROOT, cache: pathlib.Path = CACHE, extract=None):
    """The relative tagger from the training clips: baseline = each training actor's
    neutral training clips (leave-one-out for those clips themselves).
    Returns (tagger, held-out-actor results as leave_one_actor_out gives them)."""
    listed = read_training_clips(train_root / TRAIN_CLIPS)
    clips, y = [c for c, _ in listed], [e for _, e in listed]
    actors = [c.split("_")[0] for c in clips]
    X = features_for(clips, train_root / "audio", cache / "train.npz", extract=extract)
    neutral = [k for k, e in enumerate(y) if e == "NEU"]
    baseline = make_baseline([clips[k] for k in neutral], [actors[k] for k in neutral], X[neutral])
    X_rel = relative_features(X, clips, actors, baseline)
    from .affect import feature_names
    names = feature_names() if extract is None else [f"f{k}" for k in range(X.shape[1])]
    held_out = leave_one_actor_out(X_rel, y, actors, names)
    return fit(X_rel, y, names), held_out


def told_apart(tagger: Tagger, pairs, Xn: np.ndarray, Xl: np.ndarray) -> dict[str, tuple[int, int]]:
    """{emotion: (pairs told apart, pairs)}: neutral clip tagged neutral AND loaded clip tagged anything else."""
    out = {}
    for emotion in [e for e in LABELS if e != "NEU"]:
        rows = [(tagger.tag(xn).label, tagger.tag(xl).label) for p, xn, xl in zip(pairs, Xn, Xl) if p.emotion == emotion]
        if rows:
            out[emotion] = (sum(n == "NEU" and l != "NEU" for n, l in rows), len(rows))
    return out


def dev_report(absolute: Tagger, relative: Tagger, pairs, Xn: np.ndarray, Xl: np.ndarray,
               base_clips: list[str], base_actors: list[str], Xb: np.ndarray) -> str:
    """Both taggers on the pairs (affect.tag_pairs, twice), then told-apart totals
    when each speaker's baseline is only their first 1, 3, 5 clips -- how many
    calm turns a system would have to hear first."""
    actors_n = [p.actor for p in pairs]
    full = make_baseline(base_clips, base_actors, Xb)
    rel = lambda X: relative_features(X, ["-"] * len(X), actors_n, full)      # pair clips are never baseline members
    out = ["absolute tagger (tagger.json):", tag_pairs(absolute, pairs, Xn, Xl), "",
           f"relative tagger (tagger-relative.json), baseline = all of each speaker's baseline clips:",
           tag_pairs(relative, pairs, rel(Xn), rel(Xl)), "",
           "told apart against baseline size (first k clips of each speaker, by name):",
           "| baseline clips per speaker | " + " | ".join(e for e in LABELS if e != "NEU") + " | all |",
           "|---:|" + "---:|" * len(LABELS)]
    sizes = [(k, first_k(base_clips, base_actors, k)) for k in CURVE] + [("all", (base_clips, base_actors))]
    for k, (cl, ac) in sizes:
        b = make_baseline(cl, ac, Xb[[base_clips.index(c) for c in cl]])
        r = lambda X: relative_features(X, ["-"] * len(X), actors_n, b)
        t = told_apart(relative, pairs, r(Xn), r(Xl))
        out.append(f"| {k} | " + " | ".join(f"{t[e][0]} of {t[e][1]}" if e in t else "-" for e in LABELS if e != "NEU")
                   + f" | {sum(a for a, _ in t.values())} of {sum(n for _, n in t.values())} |")
    return "\n".join(out)


def own_report(relative: Tagger, absolute: Tagger, pairs, Xn: np.ndarray, Xl: np.ndarray) -> str:
    """The author's pairs: each pair's baseline is the OTHER calm takes (leave-one-out),
    so the take being tagged is never in its own baseline."""
    Xn, Xl = np.asarray(Xn, dtype=np.float64), np.asarray(Xl, dtype=np.float64)
    others = [(Xn.sum(axis=0) - Xn[k]) / (len(Xn) - 1) for k in range(len(Xn))]
    rn, rl = np.vstack([Xn[k] - others[k] for k in range(len(Xn))]), np.vstack([Xl[k] - others[k] for k in range(len(Xl))])
    return "\n".join(["absolute tagger:", tag_pairs(absolute, pairs, Xn, Xl), "",
                      f"relative tagger, baseline = the other {len(Xn) - 1} calm takes:", tag_pairs(relative, pairs, rn, rl)])


if __name__ == "__main__":
    import sys

    from .prosodyset import RATINGS, read_ratings

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    if cmd == "clips":
        ratings = read_ratings(PROSODY_ROOT / "source" / pathlib.Path(RATINGS).name)
        pairs = read_pairs(PROSODY_ROOT / PAIRS)
        clips = baseline_clips(ratings, pairs, {p.actor for p in pairs if p.split == "dev"})
        write_baseline_clips(clips, BASELINE_ROOT / BASELINE_CLIPS)
        per = sorted(sum(r.actor == a for r in clips) for a in {r.actor for r in clips})
        print(f"{len(clips)} baseline clips for the {len(per)} dev-pair actors -> {BASELINE_ROOT / BASELINE_CLIPS}; "
              f"per actor {per[0]} to {per[-1]}, median {per[len(per) // 2]}; "
              f"median agreement {np.median([r.agreement for r in clips]):.2f}")
    elif cmd == "fetch":
        n = fetch_baseline()
        print(f"downloaded {n} clips; {len(read_baseline_clips(BASELINE_ROOT / BASELINE_CLIPS))} are in "
              f"{BASELINE_ROOT / 'audio'}; manifest -> {BASELINE_ROOT / BASELINE_MANIFEST}")
    elif cmd == "train":
        problems = verify_training()
        if problems:
            sys.exit("the training clips are not the ones in the manifest:\n  " + "\n  ".join(problems[:10]))
        tagger, held_out = train_relative()
        print("leave one training actor out -- clips tagged right, on a voice the model had not heard, "
              "given that voice's neutral clips:")
        for label, (a, b) in held_out.items():
            print(f"  {label}: {a} of {b} ({100 * a / b:.0f}%)")
        tagger.save(RELATIVE_TAGGER)
        print(f"\nfitted on all training clips, relative to each actor's baseline -> {RELATIVE_TAGGER}")
    elif cmd == "dev":
        from .prosodyset import load
        problems = verify_baseline()
        if problems:
            sys.exit("the baseline clips are not the ones in the manifest:\n  " + "\n  ".join(problems[:10]))
        pairs = [lp.pair for lp in load("dev")]
        Xn = features_for([p.neutral for p in pairs], PROSODY_ROOT / "audio", CACHE / "dev.npz")
        Xl = features_for([p.loaded for p in pairs], PROSODY_ROOT / "audio", CACHE / "dev.npz")
        listed = read_baseline_clips(BASELINE_ROOT / BASELINE_CLIPS)
        bc, ba = [c for c, _ in listed], [a for _, a in listed]
        Xb = features_for(bc, BASELINE_ROOT / "audio", CACHE / "baseline.npz")
        print(f"the two taggers on the {len(pairs)} dev pairs; {len(bc)} baseline clips for {len(set(ba))} speakers\n")
        print(dev_report(Tagger.load(TAGGER), Tagger.load(RELATIVE_TAGGER), pairs, Xn, Xl, bc, ba, Xb))
    elif cmd == "own":
        from .affect import egemaps
        from .ownvoice import OWN_ROOT, load as load_own, windowed
        pairs = [p for p, _, _ in load_own()]
        ext = lambda a: egemaps(windowed(a))
        Xn = features_for([p.neutral for p in pairs], OWN_ROOT / "audio", CACHE / "own-windowed.npz", extract=ext)
        Xl = features_for([p.loaded for p in pairs], OWN_ROOT / "audio", CACHE / "own-windowed.npz", extract=ext)
        print(f"the author's {len(pairs)} pairs, windowed\n")
        print(own_report(Tagger.load(RELATIVE_TAGGER), Tagger.load(TAGGER), pairs, Xn, Xl))
    else:
        sys.exit("usage: python -m s2slab.baseline clips | fetch | train | dev | own")
