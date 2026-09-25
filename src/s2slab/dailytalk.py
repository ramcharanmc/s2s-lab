"""
DailyTalk for FT-1: the folds, one dialogue as two channels at Moshi's rate, the human reference,
and the comparison of a model's turn-taking with it.

The data: kyutai/DailyTalkContiguous (CC-BY-SA 4.0), revision pinned below -- 2,541 dialogues
read aloud by two speakers, one speaker per channel, 44.1 kHz, about 20 hours. moshi-finetune's
convention, followed here: LEFT is the speaker Moshi learns to be ("SPEAKER_MAIN", the only one
with word timings in the JSON), RIGHT is the user. Downloaded to data/dailytalk/ (ignored).

    left, right = load(n)                            24 kHz float32, one per channel
    reference(left, right)                           the human on the left answering the right
    report(model_rows, reference_rows)               a model against it, dialogue by dialogue

FOLDS, by dialogue number: n % 10 == 0 is TEST (PR-010's; never read before its registration,
never trained on), n % 10 == 1 is DEV, the rest is TRAIN. load() refuses a test dialogue without
the registration's confirmation.

WHAT DEV SHOWED (the first 50 dev dialogues, 1 to 491, 2026-09-24): DailyTalk has NO overlap --
every turn change in both directions is a gap (394 of 394), and no speaker ever talks inside the
other's turn. The left answering the right: 197 turn changes, gap median 420 ms (10th percentile
200, 90th 920, range 20-2120). Checked against the left channel's word timings: 196 turn changes,
median 415 ms; paired, the energy measure starts 20 ms earlier (median), and the two
distributions sit 54 ms apart -- the measure's own noise, against which any effect is read.
So "closer to the reference" can only mean "overlaps less" for overlap; the turn gaps carry FT-1.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .turngaps import distance, ipus, summary

REPO = "kyutai/DailyTalkContiguous"
REVISION = "33e1b501f725a6f4ed4ded95e16cd7f66b9d4bdc"      # main on 2026-09-24
ROOT = pathlib.Path("data/dailytalk")
SOURCE_SR = 44_100
SR = 24_000                                                # Moshi's rate
REGISTRATION = "PR-010"


def fold(n: int) -> str:
    return "test" if n % 10 == 0 else "dev" if n % 10 == 1 else "train"


def dialogue_ids(which: str, root: pathlib.Path = ROOT) -> list[int]:
    """The dialogue numbers of one fold that are on this machine, in order."""
    found = [int(p.stem) for p in (pathlib.Path(root) / "data_stereo").glob("*.wav") if p.stem.isdigit()]
    return sorted(n for n in found if fold(n) == which)


def load(n: int, root: pathlib.Path = ROOT, confirm: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(left, right) of dialogue n at SR, float32. Raises PermissionError for a test dialogue
    without confirm == REGISTRATION."""
    import soundfile as sf
    from scipy.signal import resample_poly
    if fold(n) == "test" and confirm != REGISTRATION:
        raise PermissionError(f"dialogue {n} is in the test fold: loading it is {REGISTRATION}'s registered run")
    x, sr = sf.read(str(pathlib.Path(root) / "data_stereo" / f"{n}.wav"), dtype="float32")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"dialogue {n}: expected stereo, got {x.shape}")
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return x[:, 0], x[:, 1]


def reference(left: np.ndarray, right: np.ndarray, sr: int = SR) -> dict:
    """The human on the left answering the right: turngaps.summary(user = right, answerer = left)."""
    return summary(ipus(right, sr), ipus(left, sr))


def pooled(rows: list[dict]) -> list[float]:
    return [f for r in rows for f in r["fto_ms"]]


def report(model: dict[int, dict], ref: dict[int, dict], name: str = "model") -> str:
    """A model's turn-taking against the reference, over the dialogues both cover: turn changes,
    gaps, overlaps, the distance between the two gap distributions (pooled), and the overlap ratio."""
    ids = sorted(set(model) & set(ref))
    m, r = [model[i] for i in ids], [ref[i] for i in ids]
    pm, pr = pooled(m), pooled(r)
    pct = lambda v: " / ".join(f"{x:.0f}" for x in np.percentile(v, [10, 50, 90])) if len(v) else "--"
    lines = [f"{len(ids)} dialogues; the answerer's floor-transfer offsets, ms (negative = overlap)", "",
             f"| | {name} | reference (the human on the left) |", "|---|---:|---:|",
             f"| turn changes | {len(pm)} | {len(pr)} |",
             f"| offset p10 / p50 / p90 | {pct(pm)} | {pct(pr)} |",
             f"| overlapping turn changes | {sum(f < 0 for f in pm)} | {sum(f < 0 for f in pr)} |",
             f"| speech inside the other's turn (within) | {sum(x['within'] for x in m)} | {sum(x['within'] for x in r)} |",
             f"| overlap ratio, median over dialogues | {np.median([x['overlap_ratio'] for x in m]):.3f} "
             f"| {np.median([x['overlap_ratio'] for x in r]):.3f} |", "",
             f"distance between the two offset distributions (Wasserstein-1, pooled): {distance(pm, pr):.0f} ms"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PR-010 (docs/preregistration.md): does FT-1's LoRA move Moshi toward the reference?
# --------------------------------------------------------------------------- #

TEST_IDS = tuple(range(0, 1000, 10))      # 100 test-fold dialogues: 0, 10, ..., 990
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0


def gain(ids, base: dict, lora: dict, ref: dict) -> tuple[float, float, float]:
    """(distance base-reference, distance LoRA-reference, their difference) over `ids`, pooled."""
    pr = pooled([ref[i] for i in ids])
    db, dl = distance(pooled([base[i] for i in ids]), pr), distance(pooled([lora[i] for i in ids]), pr)
    return db, dl, db - dl


def pr010(base: dict, lora: dict, ref: dict, ids=TEST_IDS, n: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """The registered criterion: the LoRA's distance to the reference is smaller than the base's,
    and the 95% bootstrap interval of the difference -- dialogues resampled with replacement, both
    models on the same resample -- lies above zero."""
    ids = [i for i in ids if i in base and i in lora and i in ref]
    db, dl, d = gain(ids, base, lora, ref)
    rng = np.random.default_rng(seed)
    draws = [gain(list(rng.choice(ids, len(ids))), base, lora, ref)[2] for _ in range(n)]
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    return {"n_dialogues": len(ids), "distance_base_ms": round(db, 1), "distance_lora_ms": round(dl, 1),
            "gain_ms": round(d, 1), "ci95_ms": [round(lo, 1), round(hi, 1)], "supported": bool(d > 0 and lo > 0)}


def pr010_validity(results: dict) -> list[str]:
    """What would make a run not the registered one."""
    problems = []
    got = sorted(int(i) for i in results.get("reference", {}))
    if got != list(TEST_IDS):
        problems.append(f"{len(got)} dialogues, not the registered 100 (0, 10, ..., 990)")
    commit = str(results.get("commit", ""))
    if not commit or commit.endswith("-dirty") or commit == "unknown":
        problems.append(f"commit {commit!r}: not a clean committed tree")
    if not results.get("adapter_sha256"):
        problems.append("the adapter's SHA-256 is missing")
    return problems


def format_pr010(results: dict) -> str:
    c = results["criteria"]
    lines = [f"{REGISTRATION}, the registered criterion (docs/preregistration.md), {c['n_dialogues']} test dialogues:",
             f"  distance to the reference: base {c['distance_base_ms']:.0f} ms, LoRA {c['distance_lora_ms']:.0f} ms",
             f"  gain (base - LoRA): {c['gain_ms']:+.0f} ms, 95% bootstrap interval {c['ci95_ms'][0]:+.0f} to {c['ci95_ms'][1]:+.0f}"]
    problems = pr010_validity(results)
    if problems:
        lines.append(f"{REGISTRATION}: NOT THE REGISTERED RUN -- " + "; ".join(problems))
    else:
        lines.append(f"{REGISTRATION}: " + ("SUPPORTED" if c["supported"] else
                     "NOT SUPPORTED -- the LoRA is not closer to the reference beyond the interval"))
    return "\n".join(lines)
