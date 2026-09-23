"""
Tests for s2slab.evalrun -- offline, with a test-double rung.

EchoRung "hears" whatever the test tells it to, instantly, so the wiring, the
provenance, the percentile rules and the scoring can all be checked without a
model. A tiny eval set of silent clips stands in for the real 200.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from s2slab.evalrun import latency_table, report, run, save, stage_table, wer_tables
from s2slab.rungs import Reply, Rung
from s2slab.timing import stage


class EchoRung(Rung):
    """Says it heard whatever the test queued, and replies with silence."""

    rung_id, description = "echo", "a test double"

    def __init__(self, heard):
        self.heard = list(heard)

    def _respond(self, audio, sr, sink):
        with stage("asr", sink):
            text = self.heard.pop(0)
        with stage("tts", sink):
            reply = np.zeros(12_000, dtype=np.float32)      # 0.5 s of silence at 24 kHz
        return Reply(reply, 24_000, text, "ok")


def tiny_evalset(tmp_path, items=None):
    """A manifest and silent WAVs -- everything evalset.load() needs."""
    items = items or [
        {"id": "clean-00", "cohort": "clean", "text": "the cat sat on the mat"},
        {"id": "clean-01", "cohort": "clean", "text": "a dog barked"},
        {"id": "disfluent-00", "cohort": "disfluent", "text": "well uh i went home",
         "text_fluent": "i went home"},
    ]
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    for item in items:
        sf.write(str(tmp_path / "audio" / f"{item['id']}.wav"),
                 np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"built": "2026-09-15", "seed": 1, "items": items}), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_a_run_records_every_clip_with_its_cohort_and_transcript(tmp_path):
    root = tiny_evalset(tmp_path)
    results = run(EchoRung(["the cat sat on the mat", "a cat barked", "well i went home"]), root)
    assert list(results["turns"]) == ["clean-00", "clean-01", "disfluent-00"]
    assert results["turns"]["clean-01"]["cohort"] == "clean"
    assert results["turns"]["clean-01"]["transcript"] == "a cat barked"
    assert list(results["turns"]["clean-01"]["stages_ms"]) == ["asr", "tts"]
    assert results["turns"]["clean-01"]["total_ms"] >= 0


def test_a_run_carries_its_own_provenance(tmp_path):
    root = tiny_evalset(tmp_path)
    results = run(EchoRung(["a", "b", "c"]), root, machine="laptop CPU")
    assert results["rung"] == "echo" and results["machine"] == "laptop CPU"
    assert results["commit"] and results["ran"]
    assert results["evalset"] == {"built": "2026-09-15", "seed": 1, "turns": 3}


def test_a_run_can_take_one_cohort_or_a_limit(tmp_path):
    root = tiny_evalset(tmp_path)
    one = run(EchoRung(["x"]), root, cohorts={"disfluent"})
    assert list(one["turns"]) == ["disfluent-00"]
    two = run(EchoRung(["x", "y"]), root, limit=2)
    assert list(two["turns"]) == ["clean-00", "clean-01"]


def test_a_silent_reply_has_no_leading_silence_and_the_json_stays_valid(tmp_path):
    root = tiny_evalset(tmp_path)
    results = run(EchoRung(["a", "b", "c"]), root)
    assert results["turns"]["clean-00"]["leading_silence_ms"] is None
    text = save(results, tmp_path / "run.json").read_text(encoding="utf-8")
    assert "NaN" not in text                       # NaN is not JSON; None is
    assert json.loads(text)["turns"]["clean-00"]["leading_silence_ms"] is None


# --------------------------------------------------------------------------- #
# Reporting, from the saved file alone
# --------------------------------------------------------------------------- #


def test_the_percentile_rules_still_apply(tmp_path):
    """25 turns support a median and nothing above it."""
    items = [{"id": f"clean-{k:02d}", "cohort": "clean", "text": "the cat sat on the mat"}
             for k in range(25)]
    root = tiny_evalset(tmp_path, items)
    results = run(EchoRung(["the cat sat on the mat"] * 25), root)
    row = [ln for ln in latency_table(results).splitlines() if ln.startswith("| all ")][0]
    p50, p90, p99 = [c.strip() for c in row.strip("| ").split("|")[2:5]]
    assert p50 != "—" and p90 == "—" and p99 == "—"


def test_the_stage_table_names_every_stage(tmp_path):
    results = run(EchoRung(["a", "b", "c"]), tiny_evalset(tmp_path))
    text = stage_table(results)
    assert "| asr |" in text and "| tts |" in text and "| total |" in text and "| (untimed) |" in text


def test_wer_tables_score_the_transcripts_by_cohort(tmp_path):
    root = tiny_evalset(tmp_path)
    results = run(EchoRung(["the cat sat on the mat",      # perfect
                            "a cat barked",                 # one word wrong of three
                            "well i went home"]), root)     # dropped the filler
    text = wer_tables(results, root)
    assert "Word error rate" in text and "| clean |" in text
    assert "tidied reference" in text                        # the disfluent second table
    clean = [ln for ln in text.splitlines() if ln.startswith("| clean ")][0]
    assert "11.1%" in clean                                  # 1 error in 9 reference words


def test_a_saved_run_can_be_reported_without_models_or_audio(tmp_path):
    root = tiny_evalset(tmp_path)
    path = save(run(EchoRung(["a", "b", "c"]), root), tmp_path / "run.json")
    again = json.loads(path.read_text(encoding="utf-8"))
    text = report(again, root)
    assert "echo: a test double" in text
    assert "### Processing time" in text and "Word error rate" in text
    assert "no reply ever spoke" in text                      # silent replies, so no silence stat

def test_a_run_records_first_audio_and_first_sound(tmp_path):
    """EchoRung is batch and silent: first audio is the total, and there is no
    first sound -- stored as None, like the leading silence."""
    results = run(EchoRung(["a", "b", "c"]), tiny_evalset(tmp_path))
    turn = results["turns"]["clean-00"]
    assert turn["first_audio_ms"] == pytest.approx(turn["total_ms"], abs=0.1)
    assert turn["first_sound_ms"] is None

def test_a_run_records_each_piece_and_the_playback_stall(tmp_path):
    """The pieces are saved, not just the summary, so a stall can be re-derived
    from the file -- and a different playback model tried -- without a GPU."""
    results = run(EchoRung(["a", "b", "c"]), tiny_evalset(tmp_path))
    turn = results["turns"]["clean-00"]
    assert turn["stall_ms"] == 0.0                     # batch: one piece
    assert turn["chunks"] == [[turn["first_audio_ms"], 12_000]]
    assert "playback stall:" in report(results, tiny_evalset(tmp_path))


def test_report_shows_first_audio_tables_when_the_run_has_them(tmp_path):
    root = tiny_evalset(tmp_path)
    text = report(run(EchoRung(["a", "b", "c"]), root), root)
    assert "### First audio" in text and "### First sound" in text
    assert "| first audio |" in text                 # a row in the stage table too


def test_a_run_saved_before_first_audio_existed_still_reports(tmp_path):
    """runs/r0 and runs/r0-gpu were saved without these fields; they must
    re-score exactly as before, just without the new tables."""
    root = tiny_evalset(tmp_path)
    old = run(EchoRung(["a", "b", "c"]), root)
    for turn in old["turns"].values():
        del turn["first_audio_ms"], turn["first_sound_ms"], turn["stall_ms"], turn["chunks"]
    text = report(old, root)
    assert "### Processing time" in text and "Word error rate" in text
    assert "First audio" not in text and "| first audio |" not in text
    assert "playback stall" not in text


def test_report_states_the_machine_and_the_eval_set_it_used(tmp_path):
    root = tiny_evalset(tmp_path)
    text = report(run(EchoRung(["a", "b", "c"]), root, machine="Modal T4"), root)
    assert "Modal T4" in text and "eval set built 2026-09-15, seed 1" in text


# --------------------------------------------------------------------------- #
# A rung with no transcript in the loop (R3), and what it adds to the file
# --------------------------------------------------------------------------- #


class HearingRung(Rung):
    """Never has a transcript; can be asked afterwards what it heard."""

    rung_id, description = "hearing", "a test double that hears"

    def __init__(self, heard):
        self.heard, self.hear_calls = list(heard), 0

    def _respond(self, audio, sr, sink):
        with stage("llm", sink):
            pass
        with stage("tts", sink):
            reply = np.zeros(12_000, dtype=np.float32)
        return Reply(reply, 24_000, None, "ok")

    def hear(self, audio, sr):
        self.hear_calls += 1
        return self.heard.pop(0)


def test_a_rung_without_a_transcript_gets_a_sentence_not_a_table_of_blanks_and_heard_is_scored_apart(tmp_path):
    root = tiny_evalset(tmp_path)
    rung = HearingRung(["the cat sat on the mat", "a cat barked", "well i went home"])
    results = run(rung, root)
    assert rung.hear_calls == 3 and all(t["transcript"] is None for t in results["turns"].values())
    assert results["turns"]["clean-01"]["heard"] == "a cat barked"
    assert results["peak_vram_mb"] is None or isinstance(results["peak_vram_mb"], int)
    text = wer_tables(results, root)
    assert "no transcript in the loop" in text and "| clean |" in text
    assert "asked to transcribe, outside the loop" in text and "tidied reference" not in text
    clean = [ln for ln in text.splitlines() if ln.startswith("| clean ")][0]
    assert "11.1%" in clean                                  # the heard words score as any transcript would
    assert "### Word error rate" in report(results, root)


def test_a_run_file_without_heard_or_memory_prints_as_before(tmp_path):
    root = tiny_evalset(tmp_path)
    results = run(EchoRung(["a", "b", "c"]), root)
    del results["peak_vram_mb"]                              # a file saved before the field existed
    for t in results["turns"].values():
        assert "heard" not in t
    text = report(results, root)
    assert "peak GPU memory" not in text and "outside the loop" not in text and "Word error rate" in text


def test_heard_is_scored_through_the_wrapper_rule_as_it_stands_now(tmp_path):
    """A run saved with a wrapper its code did not strip is re-scored, not edited."""
    root = tiny_evalset(tmp_path)
    rung = HearingRung(["The speech transcribed from the audio is: 'the cat sat on the mat'", "a cat barked", "well i went home"])
    results = run(rung, root)
    assert results["turns"]["clean-00"]["heard"].startswith("The speech transcribed")   # the file keeps what was said
    clean = [ln for ln in wer_tables(results, root).splitlines() if ln.startswith("| clean ")][0]
    assert "11.1%" in clean                                                 # scored on the words alone
