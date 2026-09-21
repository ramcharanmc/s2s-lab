"""
Tests for s2slab.prosodyset -- the selection rule, on made-up ratings only.

The rule is committed before any real rating is read, so nothing here knows
what CREMA-D's raters said. fake_ratings() invents a corpus of the same shape:
91 actors, 12 sentences, neutral + three emotions, voice-only ids 1xxxxx.
"""

from __future__ import annotations

import csv

import pytest

from s2slab import prosodyset as ps
from s2slab.prosodyset import (DEV_ACTORS, EMOTIONS, PER_EMOTION, PER_SENTENCE, Rating, candidates,
                               cross_check, draw, fetch, fetch_audio, filled, format_counts, read_pairs,
                               read_ratings, read_summary, select, split_actors, write_pairs)

ACTORS = [str(1001 + k) for k in range(91)]
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]


def rating(clip: str, vote: str | None = None, agreement: float = 0.95, raters: int = 10) -> Rating:
    actor, sentence, emotion, _ = clip.split("_")
    return Rating(clip, actor, sentence, emotion, ps.VOTE[emotion] if vote is None else vote,
                  agreement, raters, 60.0 if emotion != "NEU" else -1.0)


def fake_ratings(agreement=lambda actor, sentence, emotion: 0.95) -> dict[str, Rating]:
    out = {}
    for a in ACTORS:
        for s in SENTENCES:
            for e in ("NEU", *EMOTIONS):
                clip = f"{a}_{s}_{e}_XX"
                out[clip] = rating(clip, agreement=agreement(a, s, e))
    return out


# --------------------------------------------------------------------------- #
# Reading CREMA-D's files
# --------------------------------------------------------------------------- #

HEADER = ["", "A", "D", "F", "H", "N", "S", "fileName", "numResponses", "agreement", "emoVote", "meanEmoResp",
          "meanAngerResp", "meanDisgustResp", "meanFearResp", "meanHappyResp", "meanNeutralResp", "meanSadResp"]


def write_votes(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(HEADER)
        for row_id, clip, n, agreement, vote, fear in rows:
            w.writerow([str(row_id), 0, 0, 0, 0, 0, 0, clip, n, agreement, vote, 50, -1, -1, fear, -1, 70, -1])


def test_only_the_voice_only_ratings_of_the_four_emotions_are_read(tmp_path):
    path = tmp_path / "tabulatedVotes.csv"
    write_votes(path, [(100001, "1001_IEO_NEU_XX", 11, 0.909, "N", -1),
                       (100002, "1001_IEO_FEA_HI", 10, 0.8, "F", 77.5),
                       (100003, "1001_IEO_HAP_HI", 10, 0.9, "H", -1),           # not an emotion this set uses
                       (200002, "1001_IEO_FEA_HI", 10, 1.0, "F", 90.0),         # face only
                       (300002, "1001_IEO_FEA_HI", 10, 1.0, "F", 95.0)])        # voice and face
    got = read_ratings(path)
    assert set(got) == {"1001_IEO_NEU_XX", "1001_IEO_FEA_HI"}
    assert got["1001_IEO_FEA_HI"] == Rating("1001_IEO_FEA_HI", "1001", "IEO", "FEA", "F", 0.8, 10, 77.5)
    assert got["1001_IEO_NEU_XX"].intensity == -1.0


def test_a_ratings_file_without_the_columns_the_rule_reads_is_refused(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text('"","fileName","agreement"\n"100001","1001_IEO_NEU_XX",0.9\n', encoding="utf-8")
    with pytest.raises(ValueError, match="emoVote"):
        read_ratings(path)


# --------------------------------------------------------------------------- #
# Which pairs qualify
# --------------------------------------------------------------------------- #


def test_a_pair_needs_both_clips_heard_as_intended_by_enough_of_enough_raters():
    ok = {c: rating(c) for c in ("1001_DFA_NEU_XX", "1001_DFA_SAD_XX")}
    assert [(n.clip, l.clip) for n, l in candidates(ok, 0.8)] == [("1001_DFA_NEU_XX", "1001_DFA_SAD_XX")]
    for broken in (rating("1001_DFA_SAD_XX", vote="N"),                   # sad, but heard as neutral
                   rating("1001_DFA_SAD_XX", vote="N:S"),                 # a tie is not a majority
                   rating("1001_DFA_SAD_XX", agreement=0.79),
                   rating("1001_DFA_SAD_XX", raters=7),
                   rating("1001_DFA_NEU_XX", vote="S")):                  # the neutral clip sounds sad
        assert candidates({**ok, broken.clip: broken}, 0.8) == []


def test_without_a_neutral_clip_of_the_same_sentence_there_is_no_pair():
    assert candidates({c: rating(c) for c in ("1001_DFA_NEU_XX", "1001_TIE_SAD_XX")}, 0.8) == []


def test_of_several_intensities_the_one_the_raters_agreed_on_most_is_the_candidate():
    clips = {"1001_IEO_NEU_XX": rating("1001_IEO_NEU_XX"), "1001_IEO_ANG_LO": rating("1001_IEO_ANG_LO", agreement=0.82),
             "1001_IEO_ANG_MD": rating("1001_IEO_ANG_MD", agreement=0.91),
             "1001_IEO_ANG_HI": rating("1001_IEO_ANG_HI", agreement=0.91)}
    (pair,) = candidates(clips, 0.8)
    assert pair[1].clip == "1001_IEO_ANG_HI"                              # 0.91 twice: the first by name


# --------------------------------------------------------------------------- #
# Dev and test never share a voice
# --------------------------------------------------------------------------- #


def test_actors_are_split_by_the_seed_alone():
    sides = split_actors(ACTORS)
    assert sorted(sides) == ACTORS and list(sides.values()).count("dev") == DEV_ACTORS
    assert sides == split_actors(reversed(ACTORS)) and sides != split_actors(ACTORS, seed=1)


def test_no_actor_is_on_both_sides_and_none_twice_in_one_emotion():
    _, pairs, _ = select(fake_ratings())
    dev, test = ({p.actor for p in pairs if p.split == s} for s in ("dev", "test"))
    assert dev and test and not dev & test
    for emotion in EMOTIONS:
        actors = [p.actor for p in pairs if p.emotion == emotion]
        assert len(actors) == len(set(actors))


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #


def test_the_draw_fills_every_cell_and_spreads_over_sentences():
    threshold, pairs, counts = select(fake_ratings())
    assert threshold == 0.9 and len(pairs) == 3 * (PER_EMOTION["dev"] + PER_EMOTION["test"]) == 60
    assert filled(pairs) == {(e, s): PER_EMOTION[s] for e in EMOTIONS for s in ("dev", "test")}
    for emotion in EMOTIONS:
        for split in ("dev", "test"):
            sentences = [p.sentence for p in pairs if (p.emotion, p.split) == (emotion, split)]
            assert max(sentences.count(s) for s in sentences) <= PER_SENTENCE[split]
    neutrals = [p.neutral for p in pairs]
    assert len(neutrals) == len(set(neutrals))                            # no neutral clip in two pairs
    assert [p.id for p in pairs][:2] == ["sad-d01", "sad-d02"] and pairs[-1].id == "ang-t13"


def test_the_same_seed_gives_the_same_set_and_another_seed_another():
    ratings = fake_ratings()
    assert select(ratings)[1] == select(dict(reversed(list(ratings.items()))))[1]
    assert select(ratings)[1] != select(ratings, seed=7)[1]


def test_the_threshold_is_the_highest_that_fills_every_cell_and_the_counting_is_kept():
    """Fear is heard less reliably: 0.75 agreement. Everything else 0.95."""
    threshold, pairs, counts = select(fake_ratings(lambda a, s, e: 0.75 if e == "FEA" else 0.95))
    assert threshold == 0.7 and list(counts) == [0.9, 0.8, 0.7]
    assert counts[0.9][("FEA", "test")] == 0 and counts[0.9][("SAD", "test")] == PER_EMOTION["test"]
    assert all(p.agree_loaded == 0.75 for p in pairs if p.emotion == "FEA")
    text = format_counts(counts, threshold)
    assert "| at least 70% | 7 | 13 | 7 | 13 | 7 | 13 | **chosen** |" in text
    assert "| at least 90% | 7 | 13 | 0 | 0 | 7 | 13 | |" in text


def test_a_corpus_that_cannot_fill_the_set_is_an_error_with_the_counts():
    with pytest.raises(ValueError, match="at least 50%"):
        select(fake_ratings(lambda a, s, e: 0.4 if e == "ANG" else 0.95))


def test_the_draw_takes_what_the_rule_allows_and_no_more():
    """Two actors only: one pair per actor per emotion, whatever PER_EMOTION asks for."""
    ratings = {c: r for c, r in fake_ratings().items() if r.actor in ("1001", "1002")}
    pairs = draw(candidates(ratings, 0.9), {"1001": "test", "1002": "test"})
    assert filled(pairs) == {(e, "dev"): 0 for e in EMOTIONS} | {(e, "test"): 2 for e in EMOTIONS}


# --------------------------------------------------------------------------- #
# Cross-check, files, downloads
# --------------------------------------------------------------------------- #


def test_the_summary_table_must_agree_about_every_chosen_clip(tmp_path):
    _, pairs, _ = select(fake_ratings())
    summary = {c: ps.VOTE[c.split("_")[2]] for p in pairs for c in (p.neutral, p.loaded)}
    cross_check(pairs, summary)
    summary[pairs[0].loaded] = "N"
    with pytest.raises(ValueError, match=pairs[0].loaded):
        cross_check(pairs, summary)
    path = tmp_path / "summaryTable.csv"
    path.write_text('"","FileName","VoiceVote","VoiceLevel"\n"1","1001_IEO_NEU_XX","N","69.1"\n', encoding="utf-8")
    assert read_summary(path) == {"1001_IEO_NEU_XX": "N"}


def test_the_selection_is_written_once_and_reads_back_exactly(tmp_path):
    _, pairs, _ = select(fake_ratings())
    write_pairs(pairs, tmp_path / "prosody" / "pairs.tsv")
    assert read_pairs(tmp_path / "prosody" / "pairs.tsv") == pairs
    with pytest.raises(FileExistsError, match="made once"):
        write_pairs(pairs, tmp_path / "prosody" / "pairs.tsv")


def test_only_the_chosen_clips_are_fetched_from_the_pinned_revision_and_never_twice(tmp_path, monkeypatch):
    asked = []

    class Response:
        def __init__(self, url):
            asked.append(url)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"RIFF"

    monkeypatch.setattr(ps.urllib.request, "urlopen", lambda url, timeout: Response(url))
    _, pairs, _ = select(fake_ratings())
    assert fetch_audio(pairs[:2], tmp_path) == 4 and len(list((tmp_path / "audio").glob("*.wav"))) == 4
    assert all(u.startswith(f"https://media.githubusercontent.com/media/{ps.REPO}/{ps.REVISION}/AudioWAV/")
               for u in asked)
    assert fetch_audio(pairs[:2], tmp_path) == 0 and len(asked) == 4      # already there: nothing asked again
    assert not fetch("https://example.invalid/x", tmp_path / "audio" / f"{pairs[0].neutral}.wav")