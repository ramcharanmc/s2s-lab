"""
Tests for PR-014's VAP side: TV (ProjectionThreshold), the export and read-back, the DEV rule, the
criteria, and the runner's pure parts. VAP itself runs in its own environment; here its probabilities
are made up.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from s2slab import vapturns
from s2slab.policies import POLICIES, ProjectionThreshold, TurnContext, get
from s2slab.simulate import make_all

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ctx(speech: list[float], vap: list[float]) -> TurnContext:
    return TurnContext(speech_prob=speech, vap_p=vap)


def test_tv_ends_the_turn_in_a_silence_of_100_ms_where_vap_reaches_theta():
    speech = [0.9] * 10 + [0.1] * 100                         # 200 ms of speech, then silence
    vap = [0.0] * 30 + [0.9] * 80                             # VAP says "system" from frame 30
    d = ProjectionThreshold(0.5).decide(_ctx(speech, vap))
    assert d.fired and d.endpoint_s == pytest.approx(31 * 0.02) and d.reason.endswith("+vap")


def test_tv_waits_at_least_100_ms_of_silence_even_when_vap_is_sure():
    speech = [0.9] * 10 + [0.1] * 100
    d = ProjectionThreshold(0.5).decide(_ctx(speech, [1.0] * 110))
    assert d.endpoint_s == pytest.approx((10 + 5) * 0.02)


def test_tv_falls_back_at_1500_ms_when_vap_never_reaches_theta():
    speech = [0.9] * 10 + [0.1] * 100
    d = ProjectionThreshold(0.5).decide(_ctx(speech, [0.4] * 110))
    assert d.endpoint_s == pytest.approx((10 + 75) * 0.02) and d.reason.endswith("+fallback")


def test_tv_reads_the_frame_that_ends_at_the_decision_and_none_after():
    speech = [0.9] * 10 + [0.1] * 100
    vap = [0.0] * 110
    vap[40] = 0.9                                             # one frame: it ends at 41 x 20 ms
    d = ProjectionThreshold(0.5).decide(_ctx(speech, vap))
    assert d.endpoint_s == pytest.approx(41 * 0.02)


def test_tv_refuses_a_context_without_vap_and_is_not_in_the_default_sweep():
    with pytest.raises(ValueError, match="vap_p"):
        ProjectionThreshold(0.5).decide(TurnContext(speech_prob=[0.9, 0.1, 0.1]))
    assert not any(isinstance(p, ProjectionThreshold) for p in POLICIES.values())


def _items(n=4):
    us = make_all(n, 7, split="dev", turns="sentence")["clean"]
    return [(f"t{i:02d}", u) for i, u in enumerate(us)]


def _vap_json(d, items, value=0.9, commit=None, sha=None):
    (d / "vap.json").write_text(json.dumps({
        "vap_commit": commit or vapturns.VAP_COMMIT, "state_dict_sha256": sha or vapturns.STATE_DICT_SHA256,
        "p_system": {tid: [value] * (len(u.audio) // 320) for tid, u in items}}), encoding="utf-8")


def test_an_export_reads_back_exactly_with_vaps_track_in_each_context(tmp_path):
    items = _items()
    d = vapturns.export(items, tmp_path / "x")
    _vap_json(d, items)
    back = vapturns.load(d)
    for (tid, u), (tid2, v) in zip(items, back):
        assert tid == tid2 and np.array_equal(u.audio, v.audio) and v.true_end_s == u.true_end_s
        assert v.words == u.words and len(v.context().vap_p) == len(u.audio) // 320
    t1 = get("T1_semantic_500ms")
    assert [t1.decide(u.context()).endpoint_s for _, u in items] == [t1.decide(v.context()).endpoint_s for _, v in back]
    with pytest.raises(FileExistsError):
        vapturns.export(items, d)


def test_a_changed_wav_or_another_vap_is_refused(tmp_path):
    items = _items(2)
    d = vapturns.export(items, tmp_path / "x")
    _vap_json(d, items, commit="0000000")
    with pytest.raises(RuntimeError, match="pins"):
        vapturns.load(d)
    _vap_json(d, items)
    raw = bytearray((d / "t00.wav").read_bytes())
    raw[len(raw) // 2] ^= 0xFF                                # one byte in the middle of the audio
    (d / "t00.wav").write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="not the file"):
        vapturns.load(d)


def test_the_rule_picks_the_fewest_cut_offs_within_a_median_wait_of_1000_ms(tmp_path, monkeypatch):
    items = _items(6)
    d = vapturns.export(items, tmp_path / "dev")
    _vap_json(d, items, value=0.55)                          # sure enough for theta <= 0.5, never above
    blob = vapturns.tune(d)
    by = {r["theta"]: r for r in blob["rows"]}
    assert by[0.6]["fallback"] + by[0.6]["no_fire"] == 6 and by[0.5]["fallback"] == 0   # VAP never decides above 0.55
    ok = [r for r in blob["rows"] if r["p50"] <= 1000]
    assert blob["chosen"] == min(ok, key=lambda r: (r["cut"], r["theta"]))["theta"]


def test_pr014s_criteria_are_read_from_t1b_against_tv():
    from s2slab.policies import CompletionThreshold
    from s2slab.sweep import evaluate, run_comparisons
    t1b, tv = CompletionThreshold(lambda text: 1.0, 0.5), ProjectionThreshold(0.5)
    results = []
    for cohort, us in sorted(make_all(20, 9, split="dev", turns="sentence").items()):
        for u in us:
            u.context().vap_p = [1.0] * (len(u.audio) // 320)  # VAP sure at every frame: TV ends at 100 ms of silence
        results += [evaluate(t1b, us), evaluate(tv, us)]
    comps = run_comparisons(results, [(t1b.name, tv.name)], seed=0)
    crit = vapturns.pr014_criteria(results, comps, tv.name)
    assert len(comps) == 3 and crit["disfluent_cut_tv"] >= crit["disfluent_cut_t1b"]
    assert crit["S1"] is False and crit["supported"] is False


def test_the_command_line_refuses_the_eval_turns_without_the_registration():
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, "-m", "s2slab.vapturns", "export", "eval"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "--confirm PR-014" in proc.stderr


def _vaprun():
    spec = importlib.util.spec_from_file_location("vaprun", ROOT / "src" / "s2slab" / "vaprun.py")
    vaprun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vaprun)                          # imports nothing VAP needs until it runs
    return vaprun


def test_an_export_names_its_prompt_and_vap_must_have_heard_the_same(tmp_path):
    import soundfile as sf
    items = _items(2)
    pr = tmp_path / "prompt.wav"
    sf.write(str(pr), np.full(8000, 0.1, dtype="float32"), 16000, subtype="FLOAT")
    d = vapturns.export(items, tmp_path / "x", prompt=pr)
    meta = json.loads((d / "turns.json").read_text(encoding="utf-8"))
    assert meta["prompt"]["sha256"] == vapturns._sha256(pr) and (d / meta["prompt"]["file"]).resolve() == pr.resolve()
    _vap_json(d, items)                                      # a vap.json run without the prompt
    with pytest.raises(RuntimeError, match="another prompt"):
        vapturns.load(d)


def test_the_runner_puts_the_prompt_before_the_turn_on_whole_frames():
    vaprun = _vaprun()
    x = np.random.default_rng(0).normal(0, 0.01, 16000).astype("float32")
    lead = vaprun.lead(np.ones(1000, dtype="float32"), 0.3)
    assert len(lead) % 320 == 0 and len(lead) >= 1000 + 4800
    s = vaprun.stereo(x, lead, "k")
    assert s.shape == (2, len(lead) + len(x)) and np.array_equal(s[0, len(lead):], x)
    assert np.array_equal(s[1, :1000], np.ones(1000)) and not s[1, len(lead):].any()   # the system: prompt, then silence
    cut = vaprun.stereo(x[:3200], lead, "k", floor_from=x)
    assert np.array_equal(cut[0, :len(lead)], s[0, :len(lead)])                      # a cut keeps the same lead-in
    assert vaprun.lead(None, 0.3) is None and vaprun.stereo(x, None, "k")[1].any() == False


def test_the_runner_counts_read_ahead_from_the_end_of_a_cut():
    vaprun = _vaprun()
    whole = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert vaprun.read_ahead_frames(whole, [0.1, 0.2, 0.3, 0.4]) == 0
    assert vaprun.read_ahead_frames(whole, [0.1, 0.2, 0.31, 0.45]) == 2
    assert vaprun.VAP_COMMIT == vapturns.VAP_COMMIT
