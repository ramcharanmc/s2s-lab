"""
Tests for s2slab.sidechannel -- with a made-up LLM that believes whatever tag it is shown.

fake_scorer() answers Yes (0.9) to "sad", "fearful" and "angry", No (0.1) to
"neutral", and sits on the fence (0.5) with no tag: a perfectly obedient LLM,
so every number below is the TAGS' doing and can be counted by hand. The slow
test asks the real Qwen2.5-0.5B:

    python -m pytest -m slow tests/test_sidechannel.py
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from s2slab.prosodyset import Pair
from s2slab.affect import WORDS
from s2slab.sidechannel import (QUESTION, YesNoScorer, deal_counts, evaluate, format_criteria, format_report,
                                per_pair, permutation_p, run, save, score_all, user_text)

def pair(emotion: str, k: int) -> Pair:
    actor = f"10{k:02d}"
    return Pair(f"{emotion.lower()}-d{k:02d}", "dev", emotion, actor, "DFA", f"{actor}_DFA_NEU_XX",
                f"{actor}_DFA_{emotion}_XX", 0.9, 0.9, 10, 10, 60.0, 0.9)


PAIRS = [pair("SAD", 1), pair("FEA", 2), pair("ANG", 3), pair("ANG", 4)]
CLIPS = {c: "Don't forget a jacket." for p in PAIRS for c in (p.neutral, p.loaded)}


def fake_scorer(transcript, word):
    return {None: 0.5, "neutral": 0.1}.get(word, 0.9), 1.0


# The tagger is right about everything except: it hears the sad clip as neutral,
# and it hears actor 1004's NEUTRAL clip as sad.
TAGS = {"1001_DFA_NEU_XX": "neutral", "1001_DFA_SAD_XX": "neutral",
        "1002_DFA_NEU_XX": "neutral", "1002_DFA_FEA_XX": "fearful",
        "1003_DFA_NEU_XX": "neutral", "1003_DFA_ANG_XX": "angry",
        "1004_DFA_NEU_XX": "sad", "1004_DFA_ANG_XX": "angry"}


def test_the_llm_is_shown_the_words_and_then_the_tone():
    assert user_text("I'm fine.", None) == f'A caller said: "I\'m fine."\n{QUESTION}'
    assert user_text("I'm fine.", "angry") == f'A caller said: "I\'m fine." Their tone of voice was angry.\n{QUESTION}'


def test_every_clip_is_scored_under_every_word_once(tmp_path):
    asked = []

    def counting(transcript, word):
        asked.append(word)
        return fake_scorer(transcript, word)

    p = score_all(CLIPS, counting, tmp_path / "pyes.json")
    assert len(asked) == 8 * 5 and set(p) == set(CLIPS)
    assert p["1003_DFA_ANG_XX"] == {"transcript": "Don't forget a jacket.", "none": 0.5, "mass": 1.0,
                                     "neutral": 0.1, "sad": 0.9, "fearful": 0.9, "angry": 0.9}
    score_all(CLIPS, counting, tmp_path / "pyes.json")
    assert len(asked) == 40                                               # all from the cache
    score_all({**CLIPS, "1003_DFA_ANG_XX": "Don't forget the jacket."}, counting, tmp_path / "pyes.json")
    assert len(asked) == 45                                               # a changed transcript is scored again
    assert json.loads((tmp_path / "pyes.json").read_text(encoding="utf-8"))["1003_DFA_ANG_XX"]["transcript"].endswith("the jacket.")


def test_without_a_tag_both_clips_get_the_same_answer_so_no_pair_is_right():
    result = evaluate(PAIRS, score_all(CLIPS, fake_scorer), TAGS, n_shuffles=50)
    assert result["none"] == {"SAD": (0.0, 0.0, 1), "FEA": (0.0, 0.0, 1), "ANG": (0.0, 0.0, 2)}


def test_the_side_channel_is_as_good_as_its_tags_and_the_oracle_shows_what_is_the_taggers_fault():
    result = evaluate(PAIRS, score_all(CLIPS, fake_scorer), TAGS, n_shuffles=50)
    assert result["tagger"] == {"SAD": (0.0, 0.0, 1),                     # tagged neutral: missed
                                "FEA": (1.0, 1.0, 1),
                                "ANG": (1.0, 1.0, 2)}                     # 1004's neutral clip was tagged sad
    assert result["oracle"] == {"SAD": (1.0, 1.0, 1), "FEA": (1.0, 1.0, 1), "ANG": (2.0, 2.0, 2)}


def test_shuffled_tags_score_by_luck_and_the_same_seed_gives_the_same_luck():
    p = score_all(CLIPS, fake_scorer)
    a, b = evaluate(PAIRS, p, TAGS, n_shuffles=2000), evaluate(PAIRS, p, TAGS, n_shuffles=2000)
    assert a["shuffled"] == b["shuffled"]
    # 4 of the 8 dealt words are "neutral": a pair is right when its neutral clip draws one of them
    # and its loaded clip draws one of the other four -- 4/8 * 4/7 = 0.286 of the time.
    assert a["shuffled"]["FEA"][0] == pytest.approx(0.286, abs=0.04)
    assert a["shuffled"]["ANG"][0] == pytest.approx(2 * 0.286, abs=0.06)
    assert a["shuffled"]["ANG"][0] < a["tagger"]["ANG"][0]


def test_the_report_has_a_row_per_condition_and_says_how_the_llm_leans():
    p = score_all(CLIPS, fake_scorer)
    text = format_report(evaluate(PAIRS, p, TAGS, n_shuffles=50), p)
    assert "| condition | SAD: answered right · ranked right (of 1) |" in text
    assert "| none | 0 · 0 | 0 · 0 | 0 · 0 |" in text and "| oracle | 1 · 1 | 1 · 1 | 2 · 2 |" in text
    assert "| tagger | 0 · 0 | 1 · 1 | 1 · 1 |" in text
    assert "with no tag the LLM says Yes to 0 of 8 clips (median p_yes 0.50)" in text
    assert "median p_yes by the word shown: neutral 0.10, sad 0.90, fearful 0.90, angry 0.90" in text


def test_the_test_pairs_are_refused_without_a_registration():
    with pytest.raises(PermissionError, match="preregistration"):
        run("test")
# --------------------------------------------------------------------------- #
# PR-005's criteria, the listing, the evidence file
# --------------------------------------------------------------------------- #


def test_the_deals_counted_one_by_one_are_the_deals_evaluate_averages():
    p = score_all(CLIPS, fake_scorer)
    null = deal_counts(PAIRS, p, TAGS, n_shuffles=300)
    result = evaluate(PAIRS, p, TAGS, n_shuffles=300)
    for e in ("SAD", "FEA", "ANG"):
        assert len(null[e]) == 300 and sum(null[e]) / 300 == pytest.approx(result["shuffled"][e][0])


def test_the_permutation_p_is_the_share_of_deals_at_or_above_the_count():
    assert permutation_p(3, [0, 1, 2, 3, 4, 5, 3, 3, 1, 0]) == 0.5
    assert permutation_p(6, [0, 1, 2, 3, 4, 5, 3, 3, 1, 0]) == 0.0
    assert permutation_p(0, [0, 1]) == 1.0


def test_criteria_need_both_the_count_and_the_p_and_sadness_does_not_decide():
    thirteen = {e: [pair(e, 10 * k + j) for j in range(13)] for k, e in enumerate(("SAD", "FEA", "ANG"), 1)}
    prs = thirteen["SAD"] + thirteen["FEA"] + thirteen["ANG"]
    clips = {c: "x" for pr in prs for c in (pr.neutral, pr.loaded)}
    p = score_all(clips, fake_scorer)
    # a tagger that tells fear apart 10 of 13, anger 7 of 13, sadness 2 of 13
    right = {"FEA": 10, "ANG": 7, "SAD": 2}
    tags = {}
    for e, mine in thirteen.items():
        for j, pr in enumerate(mine):
            tags[pr.neutral] = "neutral"
            tags[pr.loaded] = WORDS[e] if j < right[e] else "neutral"
    result, null = evaluate(prs, p, tags, n_shuffles=400), deal_counts(prs, p, tags, n_shuffles=400)
    text = format_criteria(result, null, "PR-005")
    assert "| S1 fear | 10 of 13 |" in text and "| PASS |" in text.split("S1 fear")[1].split("\n")[0]
    assert "| S2 anger | 7 of 13 |" in text
    assert "| S3 sadness | 2 of 13 |" in text and "FAIL (reported, not deciding)" in text
    assert text.endswith("PR-005: SUPPORTED: fear and anger both hold") or "NOT SUPPORTED: failed angry" in text
    assert "sad" not in text.split("PR-005:")[1]                          # sadness never decides
    # without a registration there is no verdict line
    assert "PR-005:" not in format_criteria(result, null)


def test_a_count_over_the_bar_with_a_lucky_control_still_fails():
    null = {"FEA": [8] * 100, "ANG": [0] * 100, "SAD": [0] * 100}         # every deal matches fear's 8
    result = {"tagger": {"FEA": (8.0, 8.0, 13), "ANG": (7.0, 7.0, 13), "SAD": (7.0, 7.0, 13)}}
    text = format_criteria(result, null, "PR-005")
    assert "| S1 fear | 8 of 13 | 100 of 100 | p = 1.000 |" in text and "NOT SUPPORTED: failed fearful" in text


def test_the_per_pair_listing_shows_both_tags_and_who_was_wrong():
    p = score_all(CLIPS, fake_scorer)
    transcripts = {**CLIPS, "1002_DFA_FEA_XX": "Don't forget the jacket."}
    text = per_pair(PAIRS, p, TAGS, transcripts)
    assert "| sad-d01 | neutral | 0.10 | neutral | 0.10 | **wrong** | wrong | yes |" in text
    assert "| fea-d02 | neutral | 0.10 | fearful | 0.90 | right | right | no |" in text
    assert "| ang-d04 | sad | 0.90 | angry | 0.90 | **wrong** | wrong | yes |" in text
    assert text.endswith("SAD 0 of 1, FEA 1 of 1, ANG 1 of 2")


def test_a_saved_run_holds_everything_and_is_never_overwritten(tmp_path):
    p = score_all(CLIPS, fake_scorer)
    result, null = evaluate(PAIRS, p, TAGS, n_shuffles=20), deal_counts(PAIRS, p, TAGS, n_shuffles=20)
    path = tmp_path / "prosody" / "run.json"
    save(path, "4 dev pairs", PAIRS, p, TAGS, CLIPS, result, null)
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["run"] == "4 dev pairs" and kept["question"] == QUESTION and len(kept["pairs"]) == 4
    assert kept["tags"] == TAGS and kept["shuffled_deal_counts"]["ANG"] == null["ANG"]
    assert kept["result"]["tagger"]["FEA"] == [1.0, 1.0, 1]
    with pytest.raises(FileExistsError, match="never overwritten"):
        save(path, "again", PAIRS, p, TAGS, CLIPS, result, null)
    with pytest.raises(FileExistsError):
        run("dev", save_to=path)                                          # refused before any model loads



@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is not installed")
def test_the_real_llm_answers_yes_more_readily_to_an_angry_tone_than_to_a_neutral_one():
    scorer = YesNoScorer().load()
    angry, mass = scorer.p_yes("Don't forget a jacket.", "angry")
    neutral, _ = scorer.p_yes("Don't forget a jacket.", "neutral")
    assert 0.0 <= neutral < angry <= 1.0 and mass > 0.9
    assert scorer.p_yes("Don't forget a jacket.", "angry") == (angry, mass)          # one forward pass: deterministic