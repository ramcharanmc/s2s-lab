"""
The affect tagger: 88 numbers about a voice, and a small model that turns them into a word.

The v2 plan, R3: "Routing eGeMAPS tags around the text bottleneck is a poor
man's R3 ... how much of R3's benefit does a $0 side-channel recover?" This file
is the $0 part: one clip in, one tag out -- "neutral", "sad", "fearful" or
"angry" -- with no GPU and no neural network.

    clip, 16 kHz
      -> eGeMAPS (openSMILE, 88 functionals)   pitch, loudness, jitter, shimmer, formants,
                                               and SPECTRAL TILT: a raised voice is brighter
                                               whatever the recording gain, which a loudness
                                               number is not
      -> standardize, then softmax regression  88 x 4 weights, fitted with scipy's L-BFGS
      -> the most probable label, and its probability

ONE CLIP IN. The tagger never sees a clip's neutral twin: a deployed system has
no twin to compare with. It is speaker-INDEPENDENT -- it has to call a voice
angry without knowing how that person usually sounds. (A per-caller baseline
from earlier turns is the obvious next step, and is parked.)

WHO IT LEARNS FROM. The prosody probe set split CREMA-D's 91 actors before
anything was drawn: 30 dev, 61 test. Of the 30 dev actors, 14 have a dev pair.
The other 16 are the TRAINING actors: their neutral, sad, fearful and angry
clips that the voice-only raters HEARD as intended (a "sad" clip most raters
heard as neutral teaches the wrong thing). So:

    16 training actors   fit the model                     this file
    14 dev actors        the 21 dev pairs: develop, look   prosodyset.load("dev")
    29 test actors       the 39 locked pairs: tested once  prosodyset.load("test", confirm=...)

No voice is in two rows. Classes are unequal (sadness is rarely heard), so each
class weighs the same in the loss.

The model is saved as JSON -- feature names, means, scales, weights -- so it can
be read, diffed and committed, and tagging needs numpy alone.

    python -m s2slab.affect clips      which training clips: count them, write train.tsv
    python -m s2slab.affect fetch      fetch them (about 30 MB), write their manifest
    python -m s2slab.affect train      extract, check on held-out training actors, fit, save tagger.json
    python -m s2slab.affect dev        the saved tagger on the 21 DEV pairs (the test pairs stay locked)
    """

from __future__ import annotations

import csv
import functools
import json
import pathlib
from typing import NamedTuple

import numpy as np
from .probes import TARGET_SR, load_wav, sha256_file
from .prosodyset import (MEDIA, PAIRS, PROSODY_ROOT, RATINGS, REPO, REVISION, SEED, VOTE, Rating, fetch,
                         read_pairs, read_ratings, split_actors)

TRAIN_ROOT = PROSODY_ROOT / "train"
TRAIN_CLIPS, TRAIN_MANIFEST = "train.tsv", "manifest.json"
LABELS = ("NEU", "SAD", "FEA", "ANG")
WORDS = {"NEU": "neutral", "SAD": "sad", "FEA": "fearful", "ANG": "angry"}
TAGGER = PROSODY_ROOT / "tagger.json"         # the fitted model: committed
CACHE = pathlib.Path("results/affect")        # extracted features: ignored by git
N_FEATURES = 88
L2 = 1.0                  # on standardized features; fixed, not tuned


# --------------------------------------------------------------------------- #
# Who the tagger learns from
# --------------------------------------------------------------------------- #


def training_clips(ratings: dict[str, Rating], pairs, seed: int = SEED) -> list[Rating]:
    """The clips the tagger is fitted on, sorted by name.

    Takes:   the voice-only ratings and the probe set's pairs.
    Calls:   prosodyset.split_actors() -- the same seeded split the set was drawn with.
    Returns: clips of DEV-side actors who have NO pair in the set, whose
             majority vote is the intended emotion. Raises if a test actor or
             a paired actor slips in: that would be a leak.
    """
    sides = split_actors({r.actor for r in ratings.values()}, seed)
    paired = {p.actor for p in pairs}
    clips = sorted(r for r in ratings.values()
                   if sides[r.actor] == "dev" and r.actor not in paired and r.vote == VOTE[r.emotion])
    leaked = {r.actor for r in clips} & ({a for a, s in sides.items() if s == "test"} | paired)
    if leaked:
        raise AssertionError(f"training clips from actors that are tested on: {sorted(leaked)}")
    return clips


def write_training_clips(clips: list[Rating], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(("clip", "actor", "sentence", "emotion", "agreement", "raters"))
        w.writerows((r.clip, r.actor, r.sentence, r.emotion, round(r.agreement, 3), r.raters) for r in clips)


def read_training_clips(path: pathlib.Path) -> list[tuple[str, str]]:
    """[(clip name, emotion)] from train.tsv."""
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["clip"], r["emotion"]) for r in csv.DictReader(f, delimiter="\t")]


def fetch_training(root: pathlib.Path = TRAIN_ROOT) -> int:
    """Fetch the clips in train.tsv from the pinned CREMA-D commit into root/audio/,
    then write the manifest (their SHA-256). Returns how many were downloaded now."""
    root = pathlib.Path(root)
    clips = [c for c, _ in read_training_clips(root / TRAIN_CLIPS)]
    n = sum(fetch(f"{MEDIA}AudioWAV/{c}.wav", root / "audio" / f"{c}.wav") for c in clips)
    (root / TRAIN_MANIFEST).write_text(json.dumps({
        "set": "prosody-train", "source": {"repo": REPO, "revision": REVISION}, "seed": SEED,
        "files": {TRAIN_CLIPS: sha256_file(root / TRAIN_CLIPS, text=True)},
        "clips": {c: sha256_file(root / "audio" / f"{c}.wav") for c in clips}}, indent=2) + "\n", encoding="utf-8")
    return n


# --------------------------------------------------------------------------- #
# 88 numbers about a voice
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _smile():
    """openSMILE's eGeMAPSv02 functionals. Built once; needs requirements/affect.txt."""
    import opensmile
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.Functionals)


def feature_names() -> list[str]:
    return list(_smile().feature_names)


def egemaps(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Takes: mono float audio. Returns: the 88 eGeMAPS functionals of the whole clip, float64."""
    x = _smile().process_signal(np.asarray(audio, dtype=np.float32), sr).to_numpy(dtype=np.float64)[0]
    if x.shape != (N_FEATURES,):
        raise ValueError(f"expected {N_FEATURES} eGeMAPS functionals, got {x.shape}")
    return x


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class Tag(NamedTuple):
    label: str            # NEU | SAD | FEA | ANG
    word: str             # neutral | sad | fearful | angry
    p: float              # the model's probability for it


class Tagger(NamedTuple):
    """Standardize, multiply, softmax. Everything needed to tag, as plain lists of numbers."""

    labels: tuple[str, ...]
    features: tuple[str, ...]
    mean: np.ndarray      # (n_features,)
    scale: np.ndarray     # (n_features,)
    coef: np.ndarray      # (n_labels, n_features)
    intercept: np.ndarray  # (n_labels,)

    def proba(self, x: np.ndarray) -> np.ndarray:
        """Takes one feature vector or a matrix of them; returns probabilities, one row each."""
        z = ((np.atleast_2d(x) - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def tag(self, x: np.ndarray) -> Tag:
        p = self.proba(x)[0]
        k = int(np.argmax(p))
        return Tag(self.labels[k], WORDS.get(self.labels[k], self.labels[k]), float(p[k]))

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"labels": list(self.labels), "features": list(self.features), "l2": L2,
                                    **{k: np.round(getattr(self, k), 6).tolist()
                                       for k in ("mean", "scale", "coef", "intercept")}}, indent=1) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> "Tagger":
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["labels"]), tuple(d["features"]), *(np.array(d[k], dtype=np.float64)
                                                                for k in ("mean", "scale", "coef", "intercept")))


def _objective(theta: np.ndarray, Z: np.ndarray, onehot: np.ndarray, weight: np.ndarray, l2: float):
    """Class-weighted cross-entropy plus the L2 penalty, and its gradient.

        loss = -(1/n) * sum_i weight_i * log p(y_i | z_i)  +  l2 / (2n) * |W|^2

    theta holds W (labels x features) and then b (labels). The intercepts are not penalized.
    """
    n, d, k = len(Z), Z.shape[1], onehot.shape[1]
    W, b = theta[: k * d].reshape(k, d), theta[k * d:]
    z = Z @ W.T + b
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    g = (np.exp(logp) - onehot) * weight[:, None] / n
    value = -(weight * (logp * onehot).sum(axis=1)).sum() / n + 0.5 * l2 * (W ** 2).sum() / n
    return value, np.concatenate([(g.T @ Z + l2 * W / n).ravel(), g.sum(axis=0)])


def fit(X: np.ndarray, y: list[str], features, labels=LABELS, l2: float = L2) -> Tagger:
    """Softmax regression on standardized features, every class weighing the same.

    Takes:   X (n clips, n features); y, one label per clip; the feature names.
    Calls:   scipy.optimize.minimize (L-BFGS) on _objective(). Convex, started
             from zero: the same data always gives the same model.
    Returns: a Tagger.
    Raises:  ValueError if a label has no clip.
    """
    from scipy.optimize import minimize

    X, labels = np.asarray(X, dtype=np.float64), tuple(labels)
    idx = np.array([labels.index(v) for v in y])
    counts = np.bincount(idx, minlength=len(labels))
    if (counts == 0).any():
        raise ValueError(f"no training clip for {[l for l, c in zip(labels, counts) if c == 0]}")
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0                                  # a feature that never varies carries nothing
    Z = (X - mean) / scale
    k, d = len(labels), Z.shape[1]
    weight = (len(Z) / (k * counts))[idx]                    # balanced: each class sums to n / k
    best = minimize(_objective, np.zeros(k * d + k), args=(Z, np.eye(k)[idx], weight, l2), jac=True,
                    method="L-BFGS-B", options={"maxiter": 2000})
    return Tagger(labels, tuple(features), mean, scale, best.x[: k * d].reshape(k, d), best.x[k * d:])

# --------------------------------------------------------------------------- #
# Training it, and a first look at what it learned
# --------------------------------------------------------------------------- #


def features_for(clips: list[str], audio_dir: pathlib.Path, cache: pathlib.Path | None = None,
                 extract=None) -> np.ndarray:
    """eGeMAPS for many clips, (n clips, 88), kept in an .npz cache by clip name.

    Takes:  clip names (no .wav); the folder they are in; where the cache lives
            (results/ is ignored by git); the extractor (default: egemaps()).
    A clip is extracted once: about 0.1 s each, so 398 take under a minute.
    """
    extract = extract or egemaps
    known: dict[str, np.ndarray] = {}
    if cache is not None and pathlib.Path(cache).exists():
        with np.load(cache) as z:
            known = {name: z[name] for name in z.files}
    missing = [c for c in clips if c not in known]
    for clip in missing:
        known[clip] = extract(load_wav(pathlib.Path(audio_dir) / f"{clip}.wav"))
    if missing and cache is not None:
        pathlib.Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **known)
    return np.vstack([known[c] for c in clips])


def verify_training(root: pathlib.Path = TRAIN_ROOT) -> list[str]:
    """One line per training clip that is missing or differs from the manifest."""
    root = pathlib.Path(root)
    frozen = json.loads((root / TRAIN_MANIFEST).read_text(encoding="utf-8"))
    problems = [] if sha256_file(root / TRAIN_CLIPS, text=True) == frozen["files"][TRAIN_CLIPS] else [f"{TRAIN_CLIPS}: changed"]
    for clip, h in frozen["clips"].items():
        path = root / "audio" / f"{clip}.wav"
        if not path.exists():
            problems.append(f"audio/{clip}.wav: missing")
        elif sha256_file(path) != h:
            problems.append(f"audio/{clip}.wav: changed")
    return problems


def leave_one_actor_out(X: np.ndarray, y: list[str], actors: list[str], names) -> dict[str, tuple[int, int]]:
    """How well the recipe does on a voice it has never heard, from the training
    clips alone: fit on all actors but one, tag that one's clips, for every actor.
    Returns {label: (clips tagged right, clips)}. The tagger that is SAVED is
    fitted on all of them; this only says what to expect of it."""
    right = {label: [0, 0] for label in LABELS}
    actors_arr, y_arr = np.array(actors), np.array(y)
    for actor in sorted(set(actors)):
        held = actors_arr == actor
        model = fit(X[~held], list(y_arr[~held]), names)
        for truth, p in zip(y_arr[held], model.proba(X[held])):
            right[truth][0] += int(model.labels[int(np.argmax(p))] == truth)
            right[truth][1] += 1
    return {label: (a, b) for label, (a, b) in right.items()}


def tag_pairs(tagger: Tagger, pairs, X_neutral: np.ndarray, X_loaded: np.ndarray) -> str:
    """The tagger on pairs, one row per emotion.

        both right   the neutral clip tagged neutral AND the loaded clip tagged its own emotion
        told apart   the neutral clip tagged neutral AND the loaded clip tagged ANYTHING else --
                     all a yes/no "does the caller sound upset?" needs
    then every pair that was not told apart, with the two tags and their probabilities.
    """
    lines = ["| emotion | pairs | both right | told apart | neutral clip tagged neutral | loaded clip tagged its emotion |",
             "|---|---:|---:|---:|---:|---:|"]
    missed = []
    for emotion in [e for e in LABELS if e != "NEU"]:
        rows = [(p, tagger.tag(xn), tagger.tag(xl)) for p, xn, xl in zip(pairs, X_neutral, X_loaded)
                if p.emotion == emotion]
        if not rows:
            continue
        n_ok = sum(tn.label == "NEU" for _, tn, _ in rows)
        l_ok = sum(tl.label == emotion for _, _, tl in rows)
        both = sum(tn.label == "NEU" and tl.label == emotion for _, tn, tl in rows)
        apart = sum(tn.label == "NEU" and tl.label != "NEU" for _, tn, tl in rows)
        lines.append(f"| {emotion} | {len(rows)} | {both} | {apart} | {n_ok} | {l_ok} |")
        missed += [f"  {p.id}: neutral clip tagged {tn.word} ({tn.p:.2f}), loaded clip tagged {tl.word} ({tl.p:.2f})"
                   for p, tn, tl in rows if not (tn.label == "NEU" and tl.label != "NEU")]
    return "\n".join(lines + ["", "pairs not told apart:"] + (missed or ["  none"]))


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) == 2 else ""
    if cmd == "clips":
        ratings = read_ratings(PROSODY_ROOT / "source" / pathlib.Path(RATINGS).name)
        clips = training_clips(ratings, read_pairs(PROSODY_ROOT / PAIRS))
        write_training_clips(clips, TRAIN_ROOT / TRAIN_CLIPS)
        print(f"{len(clips)} training clips from {len({r.actor for r in clips})} actors -> {TRAIN_ROOT / TRAIN_CLIPS}")
        for label in LABELS:
            mine = [r for r in clips if r.emotion == label]
            print(f"  {label}: {len(mine)} clips heard as intended, from {len({r.actor for r in mine})} actors")
    elif cmd == "fetch":
        n = fetch_training()
        total = len(read_training_clips(TRAIN_ROOT / TRAIN_CLIPS))
        print(f"downloaded {n} clips; {total} are in {TRAIN_ROOT / 'audio'}; manifest -> {TRAIN_ROOT / TRAIN_MANIFEST}")
    elif cmd == "train":
        import time
        problems = verify_training()
        if problems:
            sys.exit("the training clips are not the ones in the manifest:\n  " + "\n  ".join(problems[:10]))
        listed = read_training_clips(TRAIN_ROOT / TRAIN_CLIPS)
        clips, y = [c for c, _ in listed], [e for _, e in listed]
        t0 = time.perf_counter()
        X = features_for(clips, TRAIN_ROOT / "audio", CACHE / "train.npz")
        print(f"{len(clips)} clips, {X.shape[1]} features each ({time.perf_counter() - t0:.0f} s; cached in {CACHE})\n")
        held_out = leave_one_actor_out(X, y, [c.split("_")[0] for c in clips], feature_names())
        print("leave one training actor out -- clips tagged right, on a voice the model had not heard:")
        for label, (a, b) in held_out.items():
            print(f"  {label}: {a} of {b} ({100 * a / b:.0f}%)")
        fit(X, y, feature_names()).save(TAGGER)
        print(f"\nfitted on all {len(clips)} clips -> {TAGGER}")
    elif cmd == "dev":
        from .prosodyset import load
        tagger, pairs = Tagger.load(TAGGER), [lp.pair for lp in load("dev")]
        Xn = features_for([p.neutral for p in pairs], PROSODY_ROOT / "audio", CACHE / "dev.npz")
        Xl = features_for([p.loaded for p in pairs], PROSODY_ROOT / "audio", CACHE / "dev.npz")
        print(f"the tagger on the {len(pairs)} dev pairs, one clip at a time\n")
        print(tag_pairs(tagger, pairs, Xn, Xl))
    else:
        sys.exit("usage: python -m s2slab.affect clips | fetch | train | dev")