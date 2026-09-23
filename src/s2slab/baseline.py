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
    python -m s2slab.baseline fresh    draw PR-007's pairs from the leftover test-side actors -> probes/prosody2/
    python -m s2slab.baseline fetch-fresh | verify-fresh     their audio and baseline clips; freeze; check
    python -m s2slab.baseline test --confirm PR-007 --save runs/prosody/pr007.json     the registered run
    python -m s2slab.baseline explore-fresh --confirm PR-007    after the result is recorded: the baseline-size curve
"""

from __future__ import annotations

import csv
import datetime
import json
import pathlib

import numpy as np

from .affect import (CACHE, LABELS, TAGGER, TRAIN_CLIPS, TRAIN_ROOT, Tagger, features_for, fit, leave_one_actor_out,
                     read_training_clips, tag_pairs, verify_training)
from .probes import sha256_file
from .prosodyset import (CLIP_SR, EMOTIONS, MANIFEST, MEDIA, PAIRS, PROSODY_ROOT, REPO, REVISION, SEED, THRESHOLDS,
                         Rating, _fingerprints, candidates, draw, fetch, read_pairs, split_actors)

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


# --------------------------------------------------------------------------- #
# The fresh pairs -- PR-007's test set, drawn from actors nothing has used
# --------------------------------------------------------------------------- #

FRESH_ROOT = pathlib.Path("probes/prosody2")
FRESH_PER_EMOTION = 13


def fresh_sides(ratings: dict[str, Rating], pairs, seed: int = SEED) -> dict[str, str]:
    """{actor: "test" | "none"}: the TEST-side actors of the original split that
    have no pair in the original set are the only ones the fresh draw may use.
    Dev-side actors trained the tagger or hold the dev pairs; the 29 test-side
    actors with a pair have been scored by everything."""
    sides = split_actors({r.actor for r in ratings.values()}, seed)
    used = {p.actor for p in pairs}
    return {a: ("test" if s == "test" and a not in used else "none") for a, s in sides.items()}


def select_fresh(ratings: dict[str, Rating], pairs, seed: int = SEED):
    """The original rule (prosodyset.select) applied to the leftover actors, test side only.

    Per emotion, in EMOTIONS order: the highest threshold in THRESHOLDS at which
    the seeded draw fills FRESH_PER_EMOTION pairs; if none does, the LOWEST
    threshold, with as many pairs as it gives -- recorded, not topped up.
    Returns: ({emotion: threshold}, the pairs with ids like sad-f01 and split
             "test", {emotion: {threshold tried: pairs drawn}}).
    """
    sides = fresh_sides(ratings, pairs, seed)
    chosen, out, counts = {}, [], {}
    for emotion in EMOTIONS:
        counts[emotion], best = {}, None
        for threshold in THRESHOLDS:
            mine = draw(candidates(ratings, threshold), sides, emotion, threshold,
                        frozenset(p.neutral for p in out), seed)
            counts[emotion][threshold] = len(mine)
            best = (threshold, mine)
            if len(mine) == FRESH_PER_EMOTION:
                break
        chosen[emotion] = best[0]
        out += [p._replace(id=p.id.replace("-t", "-f")) for p in best[1]]
    return chosen, out, counts


def format_fresh_counts(counts: dict, chosen: dict[str, float]) -> str:
    lines = ["| raters agreeing | " + " | ".join(f"{e} (of {FRESH_PER_EMOTION})" for e in counts) + " |",
             "|---|" + "---:|" * len(counts)]
    for threshold in THRESHOLDS:
        if not any(threshold in counts[e] for e in counts):
            break
        row = [("·" if threshold not in counts[e] else
                f"**{counts[e][threshold]}**" if chosen.get(e) == threshold else str(counts[e][threshold]))
               for e in counts]
        lines.append(f"| at least {threshold:.0%} | " + " | ".join(row) + " |")
    lines.append("chosen (bold): " + ", ".join(f"{e} at least {t:.0%}" for e, t in chosen.items()))
    return "\n".join(lines)


def freeze_fresh(root: pathlib.Path = FRESH_ROOT) -> pathlib.Path:
    """manifest.json for the fresh set: the SHA-256 of pairs.tsv and every clip;
    refuses a second freeze, a missing clip, a file that is not 16 kHz mono audio."""
    import soundfile as sf
    root = pathlib.Path(root)
    out = root / MANIFEST
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen set is never re-frozen; make a new version")
    pairs = read_pairs(root / PAIRS)
    for clip in sorted({c for p in pairs for c in (p.neutral, p.loaded)}):
        path = root / "audio" / f"{clip}.wav"
        if not path.exists():
            raise FileNotFoundError(f"not fetched yet: {clip}.wav")
        try:
            info = sf.info(str(path))
        except Exception as e:                                     # noqa: BLE001 -- whatever soundfile raises
            raise ValueError(f"{clip}.wav is not audio ({e}): a git-lfs pointer?") from None
        if (info.samplerate, info.channels) != (CLIP_SR, 1) or info.duration < 0.5:
            raise ValueError(f"{clip}.wav: {info.samplerate} Hz, {info.channels} channel(s), "
                             f"{info.duration:.2f} s -- expected {CLIP_SR} Hz mono, at least 0.5 s")
    ratings_sha = {name: sha256_file(PROSODY_ROOT / "source" / name) for name in ("tabulatedVotes.csv",)
                   if (PROSODY_ROOT / "source" / name).exists()}
    out.write_text(json.dumps({"set": "prosody-fresh", "version": 2, "frozen": datetime.date.today().isoformat(),
                               "drawn_from": "test-side actors of the original split with no pair in probes/prosody",
                               "source": {"repo": REPO, "revision": REVISION, "ratings_sha256": ratings_sha},
                               "seed": SEED, "pairs": len(pairs), **_fingerprints(root)}, indent=2) + "\n",
                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# PR-007's run: both taggers on the fresh pairs, the criteria, the listing.
# Reporting only, added after the registration; nothing above changes.
# --------------------------------------------------------------------------- #

P_MAX = 0.05                  # S1
GUARD = 2                     # S2: loaded clips tagged non-neutral may fall by at most this many
FRESH_CACHE = ("fresh-pairs.npz", "fresh-baseline.npz")


def sign_test_one_sided(fixed: int, broken: int) -> float:
    """P(at most `broken` of the disagreeing pairs go the wrong way | a coin flip each):
    the registered one-sided test. 12 fixed and 4 broken -> 0.038; 7 and 2 -> 0.090."""
    from math import comb
    n = fixed + broken
    return sum(comb(n, i) for i in range(broken + 1)) / 2 ** n if n else 1.0


def compare_taggers(absolute: Tagger, relative: Tagger, pairs, Xn, Xl, Xn_rel, Xl_rel) -> dict:
    """Everything PR-007 asks, as numbers.

    Takes:   the two taggers; the pairs; the pairs' features absolute (Xn, Xl)
             and relative to each speaker's baseline (Xn_rel, Xl_rel).
    Returns: {"rows": [per pair: id, emotion, the four Tags, abs_apart, rel_apart],
              "apart": {"abs": n, "rel": n}, "loaded_non_neutral": {"abs": n, "rel": n},
              "by_emotion": {emotion: {"abs": (apart, n), "rel": (apart, n)}},
              "fixed": [ids], "broken": [ids], "p": one-sided sign test, "S1": bool, "S2": bool}
    """
    rows = []
    for p, xn, xl, rn, rl in zip(pairs, Xn, Xl, Xn_rel, Xl_rel):
        an, al, bn, bl = absolute.tag(xn), absolute.tag(xl), relative.tag(rn), relative.tag(rl)
        rows.append({"id": p.id, "emotion": p.emotion, "abs": (an, al), "rel": (bn, bl),
                     "abs_apart": an.label == "NEU" and al.label != "NEU",
                     "rel_apart": bn.label == "NEU" and bl.label != "NEU"})
    apart = {k: sum(r[f"{k}_apart"] for r in rows) for k in ("abs", "rel")}
    loaded = {k: sum(r[k][1].label != "NEU" for r in rows) for k in ("abs", "rel")}
    by_emotion = {e: {k: (sum(r[f"{k}_apart"] for r in rows if r["emotion"] == e), sum(r["emotion"] == e for r in rows))
                      for k in ("abs", "rel")} for e in EMOTIONS if any(r["emotion"] == e for r in rows)}
    fixed = sorted(r["id"] for r in rows if r["rel_apart"] and not r["abs_apart"])
    broken = sorted(r["id"] for r in rows if r["abs_apart"] and not r["rel_apart"])
    p = sign_test_one_sided(len(fixed), len(broken))
    return {"rows": rows, "apart": apart, "loaded_non_neutral": loaded, "by_emotion": by_emotion,
            "fixed": fixed, "broken": broken, "p": p,
            "S1": apart["rel"] > apart["abs"] and p < P_MAX, "S2": loaded["rel"] >= loaded["abs"] - GUARD}


def format_criteria(result: dict, confirm: str = "") -> str:
    """S1 and S2 as a table, and the verdict (only for a registered run)."""
    n = len(result["rows"])
    a, l = result["apart"], result["loaded_non_neutral"]
    fmt_p = f"p = {result['p']:.3f}" if result["p"] >= 0.001 else "p < 0.001"
    lines = ["| criterion | absolute | relative | bar | |", "|---|---:|---:|---|---|",
             f"| S1 told apart, paired | {a['abs']} of {n} | {a['rel']} of {n} | more, and one-sided sign test p < {P_MAX} | "
             f"{'PASS' if result['S1'] else 'FAIL'} |",
             f"| S2 loaded clips tagged non-neutral | {l['abs']} of {n} | {l['rel']} of {n} | at least {l['abs'] - GUARD} | "
             f"{'PASS' if result['S2'] else 'FAIL'} |",
             f"  fixed {len(result['fixed'])}: {', '.join(result['fixed']) or '-'}",
             f"  broken {len(result['broken'])}: {', '.join(result['broken']) or '-'}",
             f"  exact one-sided sign test on the {len(result['fixed']) + len(result['broken'])} pairs that differ: {fmt_p}",
             "  told apart per emotion (absolute -> relative): " + ", ".join(
                 f"{e} {v['abs'][0]} -> {v['rel'][0]} of {v['abs'][1]}" for e, v in result["by_emotion"].items())]
    if confirm:
        ok = result["S1"] and result["S2"]
        lines += ["", f"{confirm}: " + ("SUPPORTED: the relative tagger tells more pairs apart, the split is not a coin flip, "
                                          "and the upset clips are still heard" if ok else
                                          "NOT SUPPORTED: failed " + ", ".join(s for s in ("S1", "S2") if not result[s]))]
    return "\n".join(lines)


def per_pair(result: dict, sizes: dict[str, int], pairs) -> str:
    """Every pair: both taggers' tags with their probabilities, told apart or not, the speaker's baseline size."""
    lines = ["| pair | absolute: calm -> upset | relative: calm -> upset | absolute | relative | baseline clips |",
             "|---|---|---|---|---|---:|"]
    for r, p in zip(result["rows"], pairs):
        cell = lambda t: f"{t[0].word} ({t[0].p:.2f}) -> {t[1].word} ({t[1].p:.2f})"
        lines.append(f"| {r['id']} | {cell(r['abs'])} | {cell(r['rel'])} | "
                     f"{'apart' if r['abs_apart'] else '**not**'} | {'apart' if r['rel_apart'] else '**not**'} | {sizes[p.actor]} |")
    return "\n".join(lines)


def save(path: pathlib.Path, head: str, pairs, result: dict, sizes: dict[str, int]) -> None:
    """Everything needed to re-count. Never overwrites."""
    from .sidechannel import refuse_to_overwrite
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**r, "abs": [t._asdict() for t in r["abs"]], "rel": [t._asdict() for t in r["rel"]]} for r in result["rows"]]
    path.write_text(json.dumps({"run": head, "pairs": [p._asdict() for p in pairs], "baseline_clips": sizes,
                                "rows": rows, **{k: v for k, v in result.items() if k != "rows"}}, indent=1) + "\n",
                    encoding="utf-8")


def fresh_features(pairs, root: pathlib.Path = FRESH_ROOT, cache: pathlib.Path = CACHE, extract=None):
    """(Xn, Xl, Xn_rel, Xl_rel, {actor: baseline size}) for the fresh pairs: features
    of the pair clips, and the same relative to each speaker's full baseline."""
    root = pathlib.Path(root)
    Xn = features_for([p.neutral for p in pairs], root / "audio", cache / FRESH_CACHE[0], extract=extract)
    Xl = features_for([p.loaded for p in pairs], root / "audio", cache / FRESH_CACHE[0], extract=extract)
    listed = read_baseline_clips(root / "baseline" / BASELINE_CLIPS)
    bc, ba = [c for c, _ in listed], [a for _, a in listed]
    Xb = features_for(bc, root / "baseline" / "audio", cache / FRESH_CACHE[1], extract=extract)
    b = make_baseline(bc, ba, Xb)
    actors = [p.actor for p in pairs]
    rel = lambda X: relative_features(X, ["-"] * len(X), actors, b)
    return Xn, Xl, rel(Xn), rel(Xl), {a: b.size(a) for a in sorted(set(actors))}


def run_test(confirm: str, save_to: str = "", root: pathlib.Path = FRESH_ROOT, cache: pathlib.Path = CACHE,
             absolute: Tagger | None = None, relative: Tagger | None = None, extract=None) -> str:
    """PR-007's run. Refuses an unverified set; the lock is prosodyset.load's.
    Returns the report (also printed by the CLI)."""
    from .prosodyset import load, verify
    import subprocess
    root = pathlib.Path(root)
    if save_to:
        from .sidechannel import refuse_to_overwrite
        refuse_to_overwrite(pathlib.Path(save_to))
    problems = verify(root) + verify_baseline(root / "baseline")
    if problems:
        raise RuntimeError("the fresh set is not the frozen one:\n  " + "\n  ".join(problems[:10]))
    pairs = [lp.pair for lp in load("test", root, confirm)]
    absolute = absolute or Tagger.load(TAGGER)
    relative = relative or Tagger.load(RELATIVE_TAGGER)
    Xn, Xl, Xn_rel, Xl_rel, sizes = fresh_features(pairs, root, cache, extract=extract)
    result = compare_taggers(absolute, relative, pairs, Xn, Xl, Xn_rel, Xl_rel)
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    head = (f"{len(pairs)} fresh pairs · confirms {confirm} · absolute tagger against the per-caller baseline · "
            f"{sum(sizes.values())} baseline clips for {len(sizes)} speakers · commit {commit}")
    report = "\n".join([head, "", "absolute tagger (tagger.json):", tag_pairs(absolute, pairs, Xn, Xl), "",
                        "relative tagger (tagger-relative.json), baseline = all of each speaker's baseline clips:",
                        tag_pairs(relative, pairs, Xn_rel, Xl_rel), "", format_criteria(result, confirm), "",
                        per_pair(result, sizes, pairs)])
    if save_to:
        save(pathlib.Path(save_to), head, pairs, result, sizes)
        report += f"\n\nsaved to {save_to}"
    return report


def explore_curve(confirm: str, root: pathlib.Path = FRESH_ROOT, cache: pathlib.Path = CACHE,
                  relative: Tagger | None = None) -> str:
    """AFTER the registered result is recorded: told apart against baseline size on
    the fresh pairs (from the run's feature caches; extracts nothing new)."""
    from .prosodyset import load
    root = pathlib.Path(root)
    pairs = [lp.pair for lp in load("test", root, confirm)]
    relative = relative or Tagger.load(RELATIVE_TAGGER)
    Xn = features_for([p.neutral for p in pairs], root / "audio", cache / FRESH_CACHE[0])
    Xl = features_for([p.loaded for p in pairs], root / "audio", cache / FRESH_CACHE[0])
    listed = read_baseline_clips(root / "baseline" / BASELINE_CLIPS)
    bc, ba = [c for c, _ in listed], [a for _, a in listed]
    Xb = features_for(bc, root / "baseline" / "audio", cache / FRESH_CACHE[1])
    actors = [p.actor for p in pairs]
    lines = ["exploratory, after the registered analysis -- told apart against baseline size:",
             "| baseline clips per speaker | " + " | ".join(EMOTIONS) + " | all |", "|---:|" + "---:|" * (len(EMOTIONS) + 1)]
    for k, (cl, ac) in [(k, first_k(bc, ba, k)) for k in CURVE] + [("all", (bc, ba))]:
        b = make_baseline(cl, ac, Xb[[bc.index(c) for c in cl]])
        r = lambda X: relative_features(X, ["-"] * len(X), actors, b)
        t = told_apart(relative, pairs, r(Xn), r(Xl))
        lines.append(f"| {k} | " + " | ".join(f"{t[e][0]} of {t[e][1]}" if e in t else "-" for e in EMOTIONS)
                     + f" | {sum(a for a, _ in t.values())} of {sum(n for _, n in t.values())} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    from .prosodyset import RATINGS, SUMMARY, cross_check, fetch_audio, read_ratings, read_summary, verify, write_pairs

    cmd = sys.argv[1] if len(sys.argv) >= 2 else ""
    if cmd in ("test", "explore-fresh"):
        import argparse
        ap = argparse.ArgumentParser(prog=f"python -m s2slab.baseline {cmd}")
        ap.add_argument("--confirm", default="", help="the registration this run confirms, e.g. PR-007")
        if cmd == "test":
            ap.add_argument("--save", default="", help="write the run to this JSON file (never overwrites)")
        args = ap.parse_args(sys.argv[2:])
        print(run_test(args.confirm, args.save) if cmd == "test" else explore_curve(args.confirm))
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit("usage: python -m s2slab.baseline clips | fetch | train | dev | own | fresh | fetch-fresh | verify-fresh"
                 " | test --confirm PR-007 [--save PATH] | explore-fresh --confirm PR-007")
    if cmd == "fresh":
        ratings = read_ratings(PROSODY_ROOT / "source" / pathlib.Path(RATINGS).name)
        v1 = read_pairs(PROSODY_ROOT / PAIRS)
        chosen, pairs, counts = select_fresh(ratings, v1)
        cross_check(pairs, read_summary(PROSODY_ROOT / "source" / pathlib.Path(SUMMARY).name))
        write_pairs(pairs, FRESH_ROOT / PAIRS)
        base = baseline_clips(ratings, v1 + pairs, {p.actor for p in pairs})
        write_baseline_clips(base, FRESH_ROOT / "baseline" / BASELINE_CLIPS)
        fresh = [a for a, s in fresh_sides(ratings, v1).items() if s == "test"]
        print(f"{len(fresh)} test-side actors with no pair in {PROSODY_ROOT}; seed {SEED}\n")
        print(format_fresh_counts(counts, chosen))
        print(f"\n{len(pairs)} fresh pairs from {len({p.actor for p in pairs})} actors -> {FRESH_ROOT / PAIRS}; "
              f"every chosen clip agrees with summaryTable.csv")
        for e in EMOTIONS:
            mine = [p for p in pairs if p.emotion == e]
            print(f"  {e}: {len(mine)} pairs; loaded clips heard as intended by {min(p.agree_loaded for p in mine):.0%} "
                  f"to {max(p.agree_loaded for p in mine):.0%} of raters; {len({p.sentence for p in mine})} of 12 sentences")
        per = sorted(sum(r.actor == a for r in base) for a in {p.actor for p in pairs})
        print(f"{len(base)} baseline clips for those actors -> {FRESH_ROOT / 'baseline' / BASELINE_CLIPS}; "
              f"per actor {per[0]} to {per[-1]}, median {per[len(per) // 2]}")
    elif cmd == "fetch-fresh":
        pairs = read_pairs(FRESH_ROOT / PAIRS)
        n = fetch_audio(pairs, FRESH_ROOT)
        m = fetch_baseline(FRESH_ROOT / "baseline")
        print(f"downloaded {n} pair clips and {m} baseline clips; frozen -> {freeze_fresh(FRESH_ROOT)}")
    elif cmd == "verify-fresh":
        problems = verify(FRESH_ROOT) + verify_baseline(FRESH_ROOT / "baseline")
        print("\n".join(problems) if problems else "intact")
        sys.exit(1 if problems else 0)
    elif cmd == "clips":
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
        sys.exit("usage: python -m s2slab.baseline clips | fetch | train | dev | own | fresh | fetch-fresh | verify-fresh"
                 " | test --confirm PR-007 [--save PATH] | explore-fresh --confirm PR-007")
