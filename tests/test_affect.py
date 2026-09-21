"""
Tests for s2slab.affect -- who the tagger learns from, and the model's arithmetic.

No openSMILE is needed for the fast tests: the model is tested on made-up
feature vectors, and the selection on made-up ratings. One test extracts real
eGeMAPS features from a tone and is skipped where openSMILE is not installed.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
from scipy.optimize import check_grad

from s2slab import affect, prosodyset as ps
from s2slab.affect import (LABELS, Tagger, _objective, fetch_training, fit, read_training_clips,
                           training_clips, write_training_clips)
from s2slab.prosodyset import Rating, select, split_actors

ACTORS = [str(1001 + k) for k in range(91)]
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]


def fake_ratings(vote=lambda actor, sentence, emotion: ps.VOTE[emotion]) -> dict[str, Rating]:
    out = {}
    for a in ACTORS:
        for s in SENTENCES:
            for e in ("NEU", "SAD", "FEA", "ANG"):
                clip = f"{a}_{s}_{e}_XX"
                out[clip] = Rating(clip, a, s, e, vote(a, s, e), 0.95, 10, 60.0 if e != "NEU" else -1.0)
    return out


# --------------------------------------------------------------------------- #
# Who the tagger learns from
# --------------------------------------------------------------------------- #


def test_training_clips_come_only_from_dev_actors_that_have_no_pair():
    ratings = fake_ratings()
    _, pairs, _ = select(ratings)
    clips = training_clips(ratings, pairs)
    sides = split_actors(ACTORS)
    actors = {r.actor for r in clips}
    assert actors and all(sides[a] == "dev" for a in actors)
    assert not actors & {p.actor for p in pairs}                          # not the dev pairs' voices, nor the test pairs'
    assert actors == {a for a in ACTORS if sides[a] == "dev"} - {p.actor for p in pairs}
    assert {r.emotion for r in clips} == set(LABELS) and clips == sorted(clips)


def test_a_clip_the_raters_did_not_hear_as_intended_teaches_nothing():
    """Every "sad" clip is heard as neutral: no sad clip is a training clip."""
    ratings = fake_ratings(lambda a, s, e: "N" if e == "SAD" else ps.VOTE[e])
    pairs = select(fake_ratings())[1]
    assert {r.emotion for r in training_clips(ratings, pairs)} == {"NEU", "FEA", "ANG"}


def test_the_training_list_is_written_read_and_fetched_with_a_manifest(tmp_path, monkeypatch):
    ratings = fake_ratings()
    clips = training_clips(ratings, select(ratings)[1])[:3]
    write_training_clips(clips, tmp_path / "train.tsv")
    assert read_training_clips(tmp_path / "train.tsv") == [(r.clip, r.emotion) for r in clips]

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
    assert fetch_training(tmp_path) == 3 and fetch_training(tmp_path) == 0          # never twice
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == ps.REVISION and set(manifest["clips"]) == {r.clip for r in clips}
    assert (tmp_path / "audio" / f"{clips[0].clip}.wav").read_bytes().endswith(f"AudioWAV/{clips[0].clip}.wav".encode())


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


def blobs(n_per_class=(60, 60, 60, 60), spread=0.4, seed=0):
    """Four clouds in three dimensions, one per label, plus a feature that never varies."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=float)
    X = np.vstack([c + spread * rng.standard_normal((n, 3)) for c, n in zip(centres, n_per_class)])
    X = np.hstack([100 * X, np.full((len(X), 1), 7.0)])                   # big units, and a constant column
    y = [label for label, n in zip(LABELS, n_per_class) for _ in range(n)]
    return X, y


def test_the_gradient_is_the_gradient_of_the_loss():
    rng = np.random.default_rng(1)
    Z, idx = rng.standard_normal((30, 5)), rng.integers(0, 4, 30)
    onehot, weight = np.eye(4)[idx], rng.uniform(0.5, 2.0, 30)
    theta = rng.standard_normal(4 * 5 + 4)
    err = check_grad(lambda t: _objective(t, Z, onehot, weight, 1.0)[0],
                     lambda t: _objective(t, Z, onehot, weight, 1.0)[1], theta)
    assert err < 1e-5


def test_separable_classes_are_learned_and_probabilities_add_up():
    X, y = blobs()
    model = fit(X, y, ["a", "b", "c", "const"])
    p = model.proba(X)
    assert p.shape == (240, 4) and np.allclose(p.sum(axis=1), 1.0)
    predicted = [LABELS[k] for k in p.argmax(axis=1)]
    assert sum(a == b for a, b in zip(predicted, y)) >= 238        # one cloud point in 240 lies on the midline
    tag = model.tag(X[0])
    assert (tag.label, tag.word) == ("NEU", "neutral") and tag.p > 0.9
    assert model.scale[3] == 1.0                                          # the constant column did not divide by zero


def test_a_rare_class_is_not_drowned_by_a_common_one():
    """6 sad clips against 200 neutral ones, overlapping: balanced weights keep sad findable."""
    X, y = blobs(n_per_class=(200, 6, 40, 40), spread=1.2, seed=3)
    model = fit(X, y, ["a", "b", "c", "const"])
    predicted = [LABELS[k] for k in model.proba(X).argmax(axis=1)]
    sad = [p for p, t in zip(predicted, y) if t == "SAD"]
    assert sad.count("SAD") >= 5


def test_the_same_data_gives_the_same_model_and_it_survives_a_round_trip(tmp_path):
    X, y = blobs()
    a, b = fit(X, y, ["a", "b", "c", "const"]), fit(X, y, ["a", "b", "c", "const"])
    assert np.array_equal(a.coef, b.coef)
    a.save(tmp_path / "tagger.json")
    back = Tagger.load(tmp_path / "tagger.json")
    assert back.labels == LABELS and back.features == ("a", "b", "c", "const")
    assert np.allclose(back.proba(X), a.proba(X), atol=1e-4)
    assert json.loads((tmp_path / "tagger.json").read_text(encoding="utf-8"))["l2"] == affect.L2


def test_a_label_with_no_clip_is_refused():
    X, y = blobs()
    with pytest.raises(ValueError, match="SAD"):
        fit(X[[k for k, v in enumerate(y) if v != "SAD"]], [v for v in y if v != "SAD"], ["a", "b", "c", "const"])

# --------------------------------------------------------------------------- #
# Training it, and the first look
# --------------------------------------------------------------------------- #


def test_features_are_extracted_once_and_then_read_from_the_cache(tmp_path):
    import soundfile as sf
    calls = []

    def fake_extract(audio):
        calls.append(len(audio))
        return np.full(4, float(len(audio)))

    for name, n in (("a", 1600), ("b", 3200)):
        sf.write(str(tmp_path / f"{name}.wav"), np.zeros(n, np.float32), 16_000, subtype="PCM_16")
    X = affect.features_for(["a", "b"], tmp_path, tmp_path / "cache" / "f.npz", extract=fake_extract)
    assert X.shape == (2, 4) and list(X[:, 0]) == [1600.0, 3200.0] and calls == [1600, 3200]
    again = affect.features_for(["b", "a"], tmp_path, tmp_path / "cache" / "f.npz", extract=fake_extract)
    assert list(again[:, 0]) == [3200.0, 1600.0] and calls == [1600, 3200]          # nothing extracted twice


def test_training_refuses_clips_that_are_not_the_ones_in_the_manifest(tmp_path, monkeypatch):
    ratings = fake_ratings()
    clips = training_clips(ratings, select(ratings)[1])[:2]
    write_training_clips(clips, tmp_path / "train.tsv")
    monkeypatch.setattr(ps.urllib.request, "urlopen", lambda url, timeout: _Bytes(url.encode()))
    fetch_training(tmp_path)
    assert affect.verify_training(tmp_path) == []
    (tmp_path / "audio" / f"{clips[0].clip}.wav").write_bytes(b"something else")
    (tmp_path / "audio" / f"{clips[1].clip}.wav").unlink()
    assert affect.verify_training(tmp_path) == [f"audio/{clips[0].clip}.wav: changed",
                                                f"audio/{clips[1].clip}.wav: missing"]


class _Bytes:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.data


def test_leaving_an_actor_out_tests_the_recipe_on_a_voice_it_has_not_heard():
    """Every actor has their own offset -- a voice. The classes still separate, so
    held-out accuracy is high; and every clip is tagged exactly once."""
    rng = np.random.default_rng(5)
    X, y, actors = [], [], []
    for a in range(6):
        voice = rng.standard_normal(3) * 0.3
        for k, label in enumerate(LABELS):
            centre = np.zeros(3) if k == 0 else 3.0 * np.eye(3)[k - 1]
            for _ in range(5):
                X.append(centre + voice + 0.3 * rng.standard_normal(3))
                y.append(label)
                actors.append(f"10{a:02d}")
    got = affect.leave_one_actor_out(np.array(X), y, actors, ["a", "b", "c"])
    assert {label: n for label, (_, n) in got.items()} == {label: 30 for label in LABELS}
    assert all(right >= 28 for right, _ in got.values())


def test_pairs_are_scored_both_right_and_told_apart():
    X, y = blobs()
    model = fit(X, y, ["a", "b", "c", "const"])
    centre = {label: X[[k for k, v in enumerate(y) if v == label]].mean(axis=0) for label in LABELS}
    pairs = select(fake_ratings())[1]
    ang = [p for p in pairs if p.emotion == "ANG"][:3]
    Xn = np.vstack([centre["NEU"], centre["NEU"], centre["FEA"]])          # the third neutral clip sounds fearful
    Xl = np.vstack([centre["ANG"], centre["SAD"], centre["ANG"]])          # the second angry clip sounds sad
    text = affect.tag_pairs(model, ang, Xn, Xl)
    assert "| ANG | 3 | 1 | 2 | 2 | 2 |" in text
    assert f"  {ang[2].id}: neutral clip tagged fearful" in text and ang[0].id not in text.split("not told apart:")[1]
    assert ang[1].id not in text.split("not told apart:")[1]                # tagged sad, not angry -- but told apart


# --------------------------------------------------------------------------- #
# Real eGeMAPS features: skipped where openSMILE is not installed
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(importlib.util.find_spec("opensmile") is None,
                    reason="openSMILE is not installed (requirements/affect.txt)")
def test_a_higher_tone_has_a_higher_egemaps_pitch():
    t = np.arange(2 * 16_000) / 16_000

    def harmonics(hz):
        return sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)).astype(np.float32) * 0.1

    names = affect.feature_names()
    low, high = affect.egemaps(harmonics(120)), affect.egemaps(harmonics(240))
    assert len(names) == affect.N_FEATURES == len(low) and np.isfinite(low).all()
    pitch = names.index("F0semitoneFrom27.5Hz_sma3nz_amean")
    assert high[pitch] - low[pitch] == pytest.approx(12.0, abs=1.0)       # an octave is twelve semitones