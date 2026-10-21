"""
Tests for s2slab.flicker (M45): the partial flicker rate, on made-up partials in the committed files' shape
[at_s, text, final, event_s, no_speech].
"""

from __future__ import annotations

import pytest

from s2slab import flicker


def test_an_update_that_only_adds_words_is_not_a_revision_and_one_that_changes_them_is():
    assert flicker.compare(["oh"], ["oh", "no"]) == {"revision": False, "deep": False, "withdrawn": 0}
    assert flicker.compare(["oh", "no"], ["oh", "no"]) == {"revision": False, "deep": False, "withdrawn": 0}
    assert flicker.compare(["nobody", "wi"], ["nobody", "will", "be"]) == {"revision": True, "deep": False, "withdrawn": 1}
    assert flicker.compare(["the", "first", "day"], ["doesnt", "it"]) == {"revision": True, "deep": True, "withdrawn": 3}
    assert flicker.compare(["oh", "no"], ["oh"]) == {"revision": True, "deep": False, "withdrawn": 1}
    assert flicker.compare([], ["okay"])["revision"] is False


def test_punctuation_and_case_are_not_flicker_and_events_are_counted_apart():
    partials = [[1.0, "Oh.", False, 0.9, 0.1], [1.2, "Oh no!", False, 0.9, 0.1], [1.4, "oh no", True, 0.9, 0.1],
                [3.0, "Okay.", False, 2.8, 0.2], [3.2, "Doesn't it?", False, 2.8, 0.2],
                [3.4, "Doesn't it does not work?", True, 2.8, 0.2]]
    c = flicker.count(partials)
    assert c["events"] == 2 and c["partials"] == 6 and c["updates"] == 4
    assert c["revisions"] == 1 and c["tail"] == 1 and c["deep"] == 0     # only "okay" -> "doesn't it"
    assert c["withdrawn"] == 1 and c["shown"] == 1 + 2 + 1 + 2
    assert c["settled"] == 2 and c["settled_revised"] == 0


def test_the_rates_per_run_and_per_kind_of_item(tmp_path):
    runs = {"r": {"partials": {
        "sl01": [[1.0, "the first day", False, 0.5, 0], [1.2, "doesnt it", True, 0.5, 0]],
        "cl01": [[1.0, "uh", False, 0.5, 0], [1.2, "uh huh", True, 0.5, 0]]}}}
    out = flicker.measure(runs, {"sl01": "stop-long", "cl01": "continue-long"})
    assert out["r"]["all"]["updates"] == 2 and out["r"]["all"]["flicker"] == 0.5
    assert out["r"]["stop"]["flicker_deep"] == 1.0 and out["r"]["keep going"]["flicker"] == 0.0
    assert out["r"]["all"]["withdrawn_per_word"] == pytest.approx(3 / 4)
    assert out["r"]["stop"]["settled_revised"] == 1
    text = flicker.format_report(out)
    assert "keep going" in text and "50.0%" in text and "1 of 1" in text
    script = tmp_path / "script.tsv"
    script.write_text("id\tcell\tkind\nsl01\tstop-long\tspeech\n", encoding="utf-8")
    assert flicker.read_cells(script) == {"sl01": "stop-long"}


def test_the_committed_runs_are_read_and_it_is_recorded_once(monkeypatch, tmp_path):
    cells = flicker.read_cells()
    assert len(cells) == 20 and set(cells.values()) == set(flicker.GROUPS)
    out = tmp_path / "flicker.json"
    out.write_text("{}")
    monkeypatch.setattr(flicker, "OUT", out)
    with pytest.raises(SystemExit, match="recorded once"):
        flicker.main([])
