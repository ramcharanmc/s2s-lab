"""
Tests for s2slab.baseline -- which clips make a speaker's baseline, the arithmetic of
features relative to it, and the point of the whole thing: a voice with a big
offset of its own defeats the absolute tagger and not the relative one.

No openSMILE: made-up feature vectors and made-up ratings throughout.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from s2slab import affect, prosodyset as ps
from s2slab.affect import LABELS, Tagger, fit
from s2slab.baseline import (Baseline, baseline_clips, dev_report, fetch_baseline, first_k, own_report,
                             read_baseline_clips, relative_features, told_apart, train_relative, verify_baseline,
                             write_baseline_clips)
from s2slab.prosodyset import Rating, select

ACTORS = [str(1001 + k) for k in range(91)]
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]


def fake_ratings(vote=lambda actor, sentence, emotion: ps.VOTE[emotion]) -> dict[str, Rating]:
    """As in test_affect: every actor, every sentence, the four emotions, voted as asked."""
    out = {}
    for a in ACTORS:
        for s in SENTENCES:
            for e in ("NEU", "SAD", "FEA", "ANG"):
                clip = f"{a}_{s}_{e}_XX"
                out[clip] = Rating(clip, a, s, e, vote(a, s, e), 0.95, 10, 60.0 if e != "NEU" else -1.0)
    return out


# --------------------------------------------------------------------------- #
# Which clips make a baseline
# --------------------------------------------------------------------------- #


def test_a_baseline_is_the_actors_neutral_clips_heard_as_neutral_and_never_a_pairs_clip():
    # every actor's IEO neutral clip was heard as sad
    ratings = fake_ratings(lambda a, s, e: "S" if (e == "NEU" and s == "IEO") else ps.VOTE[e])
    pairs = select(ratings)[1]
    dev_actors = {p.actor for p in pairs if p.split == "dev"}
    clips = baseline_clips(ratings, pairs, dev_actors)
    assert clips == sorted(clips) and {r.actor for r in clips} == dev_actors
    assert all(r.emotion == "NEU" and r.vote == "N" and r.sentence != "IEO" for r in clips)
    in_pairs = {p.neutral for p in pairs}
    assert not {r.clip for r in clips} & in_pairs
    per_actor = {a: sum(r.actor == a for r in clips) for a in dev_actors}
    assert all(11 - n == sum(p.actor == a for p in pairs) for a, n in per_actor.items())   # 12 sentences, minus IEO, minus its pairs


def test_the_baseline_list_is_written_read_fetched_and_verified(tmp_path, monkeypatch):
    ratings = fake_ratings()
    pairs = select(ratings)[1]
    clips = baseline_clips(ratings, pairs, {pairs[0].actor})[:3]
    write_baseline_clips(clips, tmp_path / "baseline.tsv")
    assert read_baseline_clips(tmp_path / "baseline.tsv") == [(r.clip, r.actor) for r in clips]

    class Response:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.url.encode()

    monkeypatch.setattr(ps.urllib.request, "urlopen", lambda url, timeout: Response(url))
    assert fetch_baseline(tmp_path) == 3 and fetch_baseline(tmp_path) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["set"] == "prosody-baseline" and set(manifest["clips"]) == {r.clip for r in clips}
    assert verify_baseline(tmp_path) == []
    (tmp_path / "audio" / f"{clips[1].clip}.wav").write_bytes(b"other")
    assert verify_baseline(tmp_path) == [f"audio/{clips[1].clip}.wav: changed"]


# --------------------------------------------------------------------------- #
# Features relative to the baseline
# --------------------------------------------------------------------------- #


def test_a_members_own_vector_is_left_out_of_its_baseline_mean():
    b = Baseline(["a1", "a2", "a3", "b1"], ["A", "A", "A", "B"], np.array([[2.0], [4.0], [6.0], [10.0]]))
    assert b.mean_for("A") == pytest.approx([4.0]) and b.mean_for("A", exclude="a1") == pytest.approx([5.0])
    assert b.mean_for("A", exclude="zz") == pytest.approx([4.0])            # not a member: nothing left out
    assert b.size("A") == 3 and b.size("C") == 0
    with pytest.raises(ValueError, match="no baseline clips for actor C"):
        b.mean_for("C")
    with pytest.raises(ValueError, match="other than b1"):
        b.mean_for("B", exclude="b1")
    X = np.array([[2.0], [9.0], [10.0]])                                        # a1 itself, a new clip of A, b1 itself
    with pytest.raises(ValueError):
        relative_features(X, ["a1", "x", "b1"], ["A", "A", "B"], b)             # b1 against nothing
    rel = relative_features(X[:2], ["a1", "x"], ["A", "A"], b)
    assert rel.tolist() == [[2.0 - 5.0], [9.0 - 4.0]]


def test_first_k_takes_each_actors_first_clips_by_name():
    clips, actors = ["b2", "a3", "a1", "b1", "a2"], ["B", "A", "A", "B", "A"]
    assert first_k(clips, actors, 2) == (["a1", "a2", "b1", "b2"], ["A", "A", "B", "B"])
    assert first_k(clips, actors, 9) == (["a1", "a2", "a3", "b1", "b2"], ["A", "A", "A", "B", "B"])


# --------------------------------------------------------------------------- #
# The point: a voice's own offset
# --------------------------------------------------------------------------- #


def voices(n_actors=8, offset=3.0, seed=7):
    """Four classes 3 apart in 3 dimensions; every actor shifted by their own
    offset of the SAME size, so absolute features confuse voice with emotion."""
    rng = np.random.default_rng(seed)
    X, y, clips, actors = [], [], [], []
    for a in range(n_actors):
        voice = offset * rng.standard_normal(3)
        for k, label in enumerate(LABELS):
            centre = np.zeros(3) if k == 0 else 3.0 * np.eye(3)[k - 1]
            for j in range(6):
                X.append(centre + voice + 0.3 * rng.standard_normal(3))
                y.append(label)
                clips.append(f"10{a:02d}_S{j}_{label}_XX")
                actors.append(f"10{a:02d}")
    return np.array(X), y, clips, actors


def test_a_voice_with_its_own_offset_defeats_the_absolute_tagger_and_not_the_relative_one(tmp_path):
    import soundfile as sf
    X, y, clips, actors = voices()
    held_abs = affect.leave_one_actor_out(X, y, actors, ["a", "b", "c"])
    right_abs = sum(r for r, _ in held_abs.values())
    # train_relative reads a train.tsv and extracts features: give it the made-up ones through `extract`
    (tmp_path / "audio").mkdir()
    with open(tmp_path / "train.tsv", "w", encoding="utf-8") as f:
        f.write("clip\tactor\tsentence\temotion\tagreement\traters\n")
        for c, a, e in zip(clips, actors, y):
            f.write(f"{c}\t{a}\tS\t{e}\t0.9\t10\n")
            sf.write(str(tmp_path / "audio" / f"{c}.wav"), np.zeros(1600, np.float32), 16_000, subtype="PCM_16")
    table = {c: x for c, x in zip(clips, X)}
    order = iter(clips)
    tagger, held_rel = train_relative(tmp_path, tmp_path / "cache", extract=lambda audio: table[next(order)])
    right_rel = sum(r for r, _ in held_rel.values())
    assert right_rel >= 0.95 * len(y) and right_abs <= 0.8 * len(y)
    assert tagger.features == ("f0", "f1", "f2") and (tmp_path / "cache" / "train.npz").exists()


def test_told_apart_counts_what_the_pair_table_counts():
    X, y = np.array([[0.0], [3.0], [6.0], [9.0]]), list(LABELS)
    model = fit(np.repeat(X, 10, axis=0), [l for l in y for _ in range(10)], ["f"])
    pairs = select(fake_ratings())[1]
    ang = [p for p in pairs if p.emotion == "ANG"][:3]
    Xn, Xl = np.array([[0.0], [0.0], [6.0]]), np.array([[9.0], [3.0], [9.0]])
    assert told_apart(model, ang, Xn, Xl) == {"ANG": (2, 3)}
    assert "| ANG | 3 | 1 | 2 | 2 | 2 |" in affect.tag_pairs(model, ang, Xn, Xl)


def test_the_dev_report_has_both_taggers_and_the_baseline_size_curve():
    X, y, clips, actors = voices(n_actors=3)
    absolute = fit(X, y, ["a", "b", "c"])
    neutral = [k for k, e in enumerate(y) if e == "NEU"]
    b = Baseline([clips[k] for k in neutral], [actors[k] for k in neutral], X[neutral])
    relative = fit(relative_features(X, clips, actors, b), y, ["a", "b", "c"])
    pairs = [p._replace(actor=a) for p, a in zip([q for q in select(fake_ratings())[1] if q.emotion == "ANG"][:3],
                                                  ["1000", "1001", "1002"])]
    Xn = np.vstack([X[[k for k, (e, a) in enumerate(zip(y, actors)) if e == "NEU" and a == p.actor][0]] for p in pairs])
    Xl = np.vstack([X[[k for k, (e, a) in enumerate(zip(y, actors)) if e == "ANG" and a == p.actor][0]] for p in pairs])
    text = dev_report(absolute, relative, pairs, Xn, Xl, [clips[k] for k in neutral], [actors[k] for k in neutral], X[neutral])
    assert text.startswith("absolute tagger (tagger.json):") and "relative tagger (tagger-relative.json)" in text
    assert "| baseline clips per speaker | SAD | FEA | ANG | all |" in text
    assert "| all | - | - | 3 of 3 | 3 of 3 |" in text and "| 1 | - | - |" in text


# --------------------------------------------------------------------------- #
# The fresh pairs
# --------------------------------------------------------------------------- #


def test_the_fresh_draw_uses_only_leftover_test_actors_and_takes_what_the_lowest_threshold_gives():
    from s2slab.baseline import FRESH_PER_EMOTION, format_fresh_counts, fresh_sides, select_fresh
    ratings = fake_ratings()
    v1 = select(ratings)[1]
    sides = fresh_sides(ratings, v1)
    original = ps.split_actors(ACTORS)
    assert {a for a, s in sides.items() if s == "test"} == {a for a in ACTORS if original[a] == "test"} - {p.actor for p in v1}
    chosen, pairs, counts = select_fresh(ratings, v1)
    assert chosen == {e: 0.9 for e in ps.EMOTIONS} and len(pairs) == 3 * FRESH_PER_EMOTION
    assert all(p.split == "test" and sides[p.actor] == "test" and p.id.split("-")[1].startswith("f") for p in pairs)
    assert not {p.actor for p in pairs} & {p.actor for p in v1}
    assert len({p.neutral for p in pairs}) == len(pairs)                          # no neutral clip in two pairs
    assert pairs[0].id == "sad-f01" and pairs[13].id == "fea-f01"
    # sadness heard as intended by only two of the leftover actors: the draw takes those two, at 50%
    leftover = sorted(a for a, s in sides.items() if s == "test")
    ratings2 = fake_ratings(lambda a, s, e: "N" if (e == "SAD" and a not in leftover[:2]) else ps.VOTE[e])
    chosen2, pairs2, counts2 = select_fresh(ratings2, v1)
    assert chosen2["SAD"] == 0.5 and sum(p.emotion == "SAD" for p in pairs2) == 2
    assert counts2["SAD"] == {t: 2 for t in ps.THRESHOLDS} and chosen2["ANG"] == 0.9
    text = format_fresh_counts(counts2, chosen2)
    assert "| at least 50% | **2** | · | · |" in text and "chosen (bold): SAD at least 50%" in text


def test_the_fresh_set_freezes_verifies_and_stays_locked_without_a_registration(tmp_path):
    import soundfile as sf
    from s2slab.baseline import freeze_fresh
    from s2slab.prosodyset import Pair, load, verify, write_pairs
    pairs = [Pair("ang-f01", "test", "ANG", "1090", "DFA", "1090_DFA_NEU_XX", "1090_DFA_ANG_XX", 0.9, 0.9, 10, 10, 60.0, 0.8)]
    write_pairs(pairs, tmp_path / "pairs.tsv")
    (tmp_path / "audio").mkdir()
    for c in (pairs[0].neutral, pairs[0].loaded):
        sf.write(str(tmp_path / "audio" / f"{c}.wav"), np.zeros(16_000, np.float32), 16_000, subtype="PCM_16")
    manifest = json.loads(freeze_fresh(tmp_path).read_text(encoding="utf-8"))
    assert manifest["set"] == "prosody-fresh" and manifest["version"] == 2 and len(manifest["clips"]) == 2
    assert verify(tmp_path) == []
    with pytest.raises(FileExistsError, match="never re-frozen"):
        freeze_fresh(tmp_path)
    with pytest.raises(PermissionError, match="preregistration"):
        load("test", tmp_path)
    assert [lp.pair for lp in load("test", tmp_path, confirm="PR-007")] == pairs
    (tmp_path / "audio" / f"{pairs[0].loaded}.wav").write_bytes(b"not audio")
    assert verify(tmp_path) == [f"audio/{pairs[0].loaded}.wav: changed"]


# --------------------------------------------------------------------------- #
# PR-007's run, on made-up features
# --------------------------------------------------------------------------- #


def test_the_one_sided_sign_test():
    from s2slab.baseline import sign_test_one_sided
    assert sign_test_one_sided(12, 4) == pytest.approx(0.0384, abs=0.0001)
    assert sign_test_one_sided(7, 2) == pytest.approx(0.0898, abs=0.0001)
    assert sign_test_one_sided(5, 0) == pytest.approx(1 / 32) and sign_test_one_sided(0, 0) == 1.0
    assert sign_test_one_sided(2, 7) > 0.98                                  # the wrong way round is no evidence


def one_feature_tagger():
    """x below 0 is neutral, above 0 angry; sad and fear never."""
    return Tagger(LABELS, ("f",), np.zeros(1), np.ones(1), np.array([[-10.0], [0.0], [0.0], [10.0]]),
                  np.array([0.0, -100.0, -100.0, 0.0]))


def fresh_root(tmp_path):
    """Six pairs, one made-up feature carried in the audio itself (a constant of
    value/10), two baseline clips per speaker at the speaker's baseline value.

        pair     baseline  calm  upset   absolute   relative
        sad-f01  2         1     3       not apart  apart      fixed
        sad-f02  2         1     3       not apart  apart      fixed
        fea-f01  0         -1    3       apart      apart
        fea-f02  5         -1    3       apart      not apart  broken (upset tagged neutral)
        ang-f01  0         -1    -2      not        not        (upset neutral both ways)
        ang-f02  2         1     3       not apart  apart      fixed
    """
    import soundfile as sf
    from s2slab.baseline import Rating, fetch_baseline, freeze_fresh, write_baseline_clips
    from s2slab.prosodyset import Pair, write_pairs
    spec = [("sad-f01", "SAD", "1101", 2, 1, 3), ("sad-f02", "SAD", "1102", 2, 1, 3), ("fea-f01", "FEA", "1103", 0, -1, 3),
            ("fea-f02", "FEA", "1104", 5, -1, 3), ("ang-f01", "ANG", "1105", 0, -1, -2), ("ang-f02", "ANG", "1106", 2, 1, 3)]
    (tmp_path / "audio").mkdir()
    (tmp_path / "baseline" / "audio").mkdir(parents=True)
    pairs, base = [], []
    for pid, e, a, b, c, l in spec:
        n, ld = f"{a}_DFA_NEU_XX", f"{a}_DFA_{e}_XX"
        pairs.append(Pair(pid, "test", e, a, "DFA", n, ld, 0.9, 0.7, 10, 10, 60.0, 0.6))
        for clip, v in ((n, c), (ld, l)):
            sf.write(str(tmp_path / "audio" / f"{clip}.wav"), np.full(16_000, v / 10, np.float32), 16_000, subtype="PCM_16")
        for s in ("IEO", "TIE"):
            base.append(Rating(f"{a}_{s}_NEU_XX", a, s, "NEU", "N", 0.8, 10, -1.0))
            sf.write(str(tmp_path / "baseline" / "audio" / f"{a}_{s}_NEU_XX.wav"), np.full(16_000, b / 10, np.float32),
                     16_000, subtype="PCM_16")
    write_pairs(pairs, tmp_path / "pairs.tsv")
    write_baseline_clips(base, tmp_path / "baseline" / "baseline.tsv")
    freeze_fresh(tmp_path)
    fetch_baseline(tmp_path / "baseline")                                     # files exist: writes the manifest only
    return pairs


def carried(audio):
    return np.array([round(float(audio[0]) * 10)])


def test_the_registered_run_counts_fixed_and_broken_and_gives_the_verdict(tmp_path):
    from s2slab.baseline import run_test
    fresh_root(tmp_path)
    t = one_feature_tagger()
    with pytest.raises(PermissionError, match="preregistration"):
        run_test("", root=tmp_path, cache=tmp_path / "cache", absolute=t, relative=t, extract=carried)
    report = run_test("PR-007", str(tmp_path / "out" / "pr007.json"), root=tmp_path, cache=tmp_path / "cache",
                      absolute=t, relative=t, extract=carried)
    assert report.startswith("6 fresh pairs · confirms PR-007") and "12 baseline clips for 6 speakers" in report
    assert "| S1 told apart, paired | 2 of 6 | 4 of 6 | more, and one-sided sign test p < 0.05 | FAIL |" in report
    assert "| S2 loaded clips tagged non-neutral | 5 of 6 | 4 of 6 | at least 3 | PASS |" in report
    assert "fixed 3: ang-f02, sad-f01, sad-f02" in report and "broken 1: fea-f02" in report
    assert "on the 4 pairs that differ: p = 0.312" in report
    assert "told apart per emotion (absolute -> relative): SAD 0 -> 2 of 2, FEA 2 -> 1 of 2, ANG 0 -> 1 of 2" in report
    assert "PR-007: NOT SUPPORTED: failed S1" in report
    assert "| fea-f02 | neutral (1.00) -> angry (1.00) | neutral (1.00) -> neutral (1.00) | apart | **not** | 2 |" in report
    kept = json.loads((tmp_path / "out" / "pr007.json").read_text(encoding="utf-8"))
    assert kept["apart"] == {"abs": 2, "rel": 4} and kept["S1"] is False and kept["S2"] is True
    assert kept["rows"][3]["rel"][1]["label"] == "NEU" and kept["baseline_clips"] == {f"110{k}": 2 for k in range(1, 7)}
    with pytest.raises(FileExistsError):
        run_test("PR-007", str(tmp_path / "out" / "pr007.json"), root=tmp_path, cache=tmp_path / "cache",
                 absolute=t, relative=t, extract=carried)
    # a changed clip stops the run before anything is tagged
    import soundfile as sf
    sf.write(str(tmp_path / "audio" / "1101_DFA_SAD_XX.wav"), np.full(16_000, 0.9, np.float32), 16_000, subtype="PCM_16")
    with pytest.raises(RuntimeError, match="changed"):
        run_test("PR-007", root=tmp_path, cache=tmp_path / "cache", absolute=t, relative=t, extract=carried)


def test_a_clear_win_is_supported_and_the_guardrail_can_fail_alone():
    from s2slab.baseline import compare_taggers, format_criteria
    from s2slab.prosodyset import Pair
    t = one_feature_tagger()
    pairs = [Pair(f"ang-f{k:02d}", "test", "ANG", f"12{k:02d}", "DFA", "n", "l", 0.9, 0.9, 10, 10, 60.0, 0.8) for k in range(1, 15)]
    # absolute tells 2 of 14 apart; relative 14 of 14: 12 fixed, 0 broken
    Xn = np.array([[-1.0]] * 2 + [[1.0]] * 12)
    Xl = np.full((14, 1), 3.0)
    r = compare_taggers(t, t, pairs, Xn, Xl, np.full((14, 1), -1.0), Xl)
    assert (r["apart"], len(r["fixed"]), len(r["broken"])) == ({"abs": 2, "rel": 14}, 12, 0)
    assert r["p"] == pytest.approx(1 / 4096) and r["S1"] and r["S2"]
    assert "PR-007: SUPPORTED" in format_criteria(r, "PR-007") and "PR-007" not in format_criteria(r)
    # the same, but the relative tagger hears three of the upset clips as neutral: S2 fails
    Xl_rel = np.array([[-1.0]] * 3 + [[3.0]] * 11)
    r2 = compare_taggers(t, t, pairs, Xn, Xl, np.full((14, 1), -1.0), Xl_rel)
    assert r2["loaded_non_neutral"] == {"abs": 14, "rel": 11} and r2["S2"] is False and r2["S1"] is True
    assert "NOT SUPPORTED: failed S2" in format_criteria(r2, "PR-007")


def test_the_exploratory_curve_reads_the_caches_the_run_left(tmp_path):
    from s2slab.baseline import explore_curve, run_test
    fresh_root(tmp_path)
    t = one_feature_tagger()
    run_test("PR-007", root=tmp_path, cache=tmp_path / "cache", absolute=t, relative=t, extract=carried)
    text = explore_curve("PR-007", root=tmp_path, cache=tmp_path / "cache", relative=t)
    assert text.startswith("exploratory, after the registered analysis")
    assert "| all | 2 of 2 | 1 of 2 | 1 of 2 | 4 of 6 |" in text and "| 1 | 2 of 2 | 1 of 2 | 1 of 2 | 4 of 6 |" in text


def test_the_own_report_uses_the_other_calm_takes_as_each_pairs_baseline():
    seen = []

    class Spy(Tagger):
        def tag(self, x):
            seen.append(np.asarray(x).tolist())
            return affect.Tag("NEU", "neutral", 1.0)

    spy = Spy(LABELS, ("f",), np.zeros(1), np.ones(1), np.zeros((4, 1)), np.zeros(4))
    pairs = [p for p in select(fake_ratings())[1] if p.emotion == "ANG"][:3]
    Xn, Xl = np.array([[1.0], [2.0], [6.0]]), np.array([[10.0], [10.0], [10.0]])
    own_report(spy, spy, pairs, Xn, Xl)
    absolute, relative = seen[:6], seen[6:]
    assert relative == [[1.0 - 4.0], [10.0 - 4.0], [2.0 - 3.5], [10.0 - 3.5], [6.0 - 1.5], [10.0 - 1.5]]
