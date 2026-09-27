"""The quantization ladder's comparison, noise floor and table (s2slab.quantladder), on a synthetic voice
and fake speakers: no Kokoro, no ONNX file, no recognizer."""
import hashlib
import json
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab import quantladder as q
from s2slab import severity
from test_severity import voice

SR = q.SR
LOWER = severity.depression(0)._replace(f0_offset_hz=-6.0)          # 6 Hz lower, nothing else


class FakeTorch:
    """The 'package': the synthetic voice, a little different on every render (its swing moves)."""

    def __init__(self):
        self.calls = 0

    def speak(self, text, who):
        self.calls += 1
        return voice(swing=30.0 + (self.calls % 5)), [[5, 6, 7], [8, 9]]


class FakeOnnx:
    def __init__(self, audio):
        self.audio, self.size_mb, self.load_s, self.fed = audio, 1.0, 0.1, []

    def speak(self, chunks, who):
        self.fed.append(chunks)
        return self.audio


def ladder():
    ref = voice()
    lower = severity.render(ref, SR, LOWER).audio     # 6 Hz lower, nothing else
    return {"onnx-fp32": FakeOnnx(ref), "onnx-fp16": FakeOnnx(np.full(len(ref), np.nan, dtype=np.float32)),
            "onnx-q8": FakeOnnx(lower)}


def test_a_copy_that_changes_nothing_measures_as_identity_and_a_lower_voice_as_lower():
    ref = voice()
    same = q.compare(ref, ref.copy())
    assert same["finite"] and same["measures"] == pytest.approx(q.IDENTITY)
    lower = severity.render(ref, SR, LOWER).audio
    got = q.compare(ref, lower)["measures"]
    assert got["f0_offset_hz"] == pytest.approx(-6.0, abs=1.0) and got["rate"] == pytest.approx(1.0, abs=0.02)
    assert q.deviation("f0_offset_hz", -6.0) == 6.0 and q.deviation("rate", 0.9) == pytest.approx(0.1)


def test_a_render_that_is_not_a_number_is_counted_and_never_measured():
    ref = voice()
    bad = ref.copy()
    bad[100:350] = np.nan
    got = q.compare(ref, bad)
    assert got == {"finite": False, "nonfinite_samples": 250, "n_samples": len(ref), "measures": None}
    assert q.compare(ref, np.zeros(0, np.float32))["finite"] is False
    silent = q.compare(ref, np.zeros(SR, np.float32))
    assert silent["finite"] and silent["measures"] is None                     # audio, but no speech in it


def test_a_rungs_file_is_held_to_its_sha256_before_anything_loads(tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "m.onnx").write_bytes(b"a model")
    rungs = {"r": ("onnx/m.onnx", hashlib.sha256(b"a model").hexdigest())}
    assert q.rung_file("r", tmp_path, rungs) == tmp_path / "onnx" / "m.onnx"
    (tmp_path / "onnx" / "m.onnx").write_bytes(b"another model")
    with pytest.raises(ValueError, match="the pinned file has"):
        q.rung_file("r", tmp_path, rungs)
    with pytest.raises(FileNotFoundError, match="download commands"):
        q.rung_file("r", tmp_path / "nowhere", rungs)
    assert set(q.RUNGS) == {"onnx-fp32", "onnx-fp16", "onnx-q8"} and q.REFERENCE in q.RUNGS
    assert all(len(d) == 64 and f.startswith("onnx/") for f, d in q.RUNGS.values()) and len(q.REVISION) == 40


def test_the_onnx_speaker_feeds_padded_ids_and_the_voice_row_of_their_length(tmp_path):
    (tmp_path / "voices").mkdir()
    pack = np.arange(510 * 256, dtype=np.float32).reshape(510, 1, 256)
    pack.tofile(tmp_path / "voices" / "v.bin")

    class Session:
        def __init__(self):
            self.feeds = []

        def run(self, names, feed):
            self.feeds.append(feed)
            return [np.ones((1, 10), np.float32)]

    s = q.OnnxKokoro("onnx-fp32", tmp_path)
    s.session = Session()
    audio = s.speak([[5, 6, 7], [8, 9]], "v")
    assert audio.shape == (20,) and audio.dtype == np.float32
    first, second = s.session.feeds
    assert first["input_ids"].tolist() == [[0, 5, 6, 7, 0]] and first["input_ids"].dtype == np.int64
    assert np.array_equal(first["style"], pack[3]) and np.array_equal(second["style"], pack[2])
    assert first["speed"].tolist() == [1.0]
    with pytest.raises(ValueError, match="1 to 510"):
        s.speak([list(range(511))], "v")


def test_the_run_measures_every_rung_against_the_reference_and_torch_against_itself(tmp_path):
    onnx, torch_tts = ladder(), FakeTorch()
    heard = lambda audio: "the shop was closed"
    results = q.run({"s1": "The shop was closed.", "s2": "The shop was open."}, ["v1"], torch_tts, onnx, hear=heard,
                    listen=tmp_path, torch_repeats=3)
    rows, floor_rows = results["rows"], results["floor_rows"]
    assert len(rows) == 2 * (3 + 3) and len(floor_rows) == 2 * 2 and torch_tts.calls == 6
    assert all(r["against"] == "onnx-fp32" for r in rows) and all(r["against"] == r["rung"] == "torch" for r in floor_rows)
    assert all(s.fed == [[[5, 6, 7], [8, 9]]] * 2 for s in onnx.values())                # every rung, the package's ids
    by = {(r["rung"], r["sentence"], r["repeat"]): r for r in rows}
    assert by[("onnx-fp32", "s1", 0)]["measures"] == pytest.approx(q.IDENTITY)
    assert by[("onnx-q8", "s1", 0)]["measures"]["f0_offset_hz"] == pytest.approx(-6.0, abs=1.0)
    nan = by[("onnx-fp16", "s1", 0)]
    assert not nan["finite"] and nan["measures"] is None and "heard" not in nan and "wer" not in nan
    assert by[("onnx-q8", "s1", 0)]["wer"] == {"raw": {"errors": 0, "words": 4}, "normalized": {"errors": 0, "words": 4}}
    assert by[("onnx-q8", "s2", 0)]["wer"]["raw"] == {"errors": 1, "words": 4}
    assert "heard" in by[("torch", "s1", 0)] and "heard" not in by[("torch", "s1", 1)]    # the recognizer hears one torch render
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"synthetic-{r}-v1-{s}.wav" for r in ("onnx-fp32", "onnx-q8", "torch") for s in ("s1", "s2")]   # no file for NaN
    with sf.SoundFile(str(tmp_path / "synthetic-onnx-q8-v1-s1.wav")) as f:
        assert f.samplerate == SR and "SYNTHETIC SPEECH" in f.comment and "Rung onnx-q8" in f.comment
    json.dumps(results)                                                                    # the run file can be written
    with pytest.raises(ValueError, match="reference rung"):
        q.run({"s1": "x"}, ["v1"], FakeTorch(), {"onnx-q8": onnx["onnx-q8"]})
    with pytest.raises(RuntimeError, match="not audio"):
        q.run({"s1": "x"}, ["v1"], FakeTorch(), {"onnx-fp32": onnx["onnx-fp16"]})


def test_the_summary_counts_the_floor_the_tolerance_and_the_renders_that_are_not_audio():
    results = q.run({"s1": "The shop was closed.", "s2": "The shop was open."}, ["v1"], FakeTorch(), ladder(),
                    hear=lambda audio: "the shop was closed", torch_repeats=3)
    results["rungs"] = {r: {"size_mb": 1.0} for r in q.RUNGS}
    s, f = q.summarize(results), q.floor(results)
    assert s["onnx-fp32"]["finite"] == 2 and s["onnx-fp32"]["measures"]["f0_offset_hz"]["within"] == 2
    assert s["onnx-fp16"]["finite"] == 0 and s["onnx-fp16"]["renders"] == 2 and s["onnx-fp16"]["measures"]["rate"]["n"] == 0
    assert s["onnx-fp16"]["wer_raw"] == {"errors": 0, "words": 0, "renders": 0, "rate": None}
    assert s["onnx-q8"]["measures"]["f0_offset_hz"]["within"] == 0                          # 6 Hz is outside 3 Hz
    assert s["onnx-q8"]["measures"]["f0_offset_hz"]["mean_signed"] == pytest.approx(-6.0, abs=1.0)
    assert s["onnx-q8"]["measures"]["rate"]["within"] == 2
    assert s["onnx-q8"]["wer_raw"] == {"errors": 1, "words": 8, "renders": 2, "rate": 0.125}
    assert s["torch"]["renders"] == 2                                                       # the first render of each sentence
    assert f["f0_dev_scale"]["n"] == 4 and f["f0_dev_scale"]["max"] > 0.01                  # the fake package's swing moves
    assert f["rate"]["max"] == pytest.approx(0.0, abs=0.02)
    text = q.report(results)
    assert "| onnx-fp16 | 1.0 | 0 of 2 |" in text and "renders that are not audio: 2" in text
    assert "onnx-fp16 v1 s1:" in text and "noise floor, torch against torch (2 repeats a sentence)" in text
    assert "12.5% / 12.5%" in text


def test_the_dev_set_is_pr009s_twelve_sentences_and_its_two_dev_voices():
    assert len(q.DEV_SENTENCES) == 12 and set(q.DEV_SENTENCES) == set(severity.DEV_SENTENCES) | set(severity.TEST_SENTENCES)
    assert tuple(q.DEV_VOICES) == ("af_heart", "am_michael") and q.TORCH_REPEATS == 5


# --------------------------------------------------------------------------- #
# PR-013: the held-out set, the criteria, the lock -- on made-up runs and fake speakers only
# --------------------------------------------------------------------------- #
ROOT = pathlib.Path(__file__).resolve().parents[1]


def made_up(q8_offset=-4.0, q8_errors=3, fp16_nan=1, floor_hz=0.5, voices=q.TEST_VOICES, sentences=q.TEST_SENTENCES):
    """A run in run()'s shape: fp32 and torch change nothing, the 8-bit copy is `q8_offset` Hz off and
    makes one more error on its first `q8_errors` renders, fp16 returns no audio `fp16_nan` times."""
    rows, floor_rows, n = [], [], 0
    same = lambda **m: {**q.IDENTITY, **m}
    for voice in voices:
        for sid in sentences:
            base = {"voice": voice, "sentence": sid, "against": "onnx-fp32", "rtf": 0.3, "finite": True, "nonfinite_samples": 0, "n_samples": 100}
            heard = lambda errors: {"wer": {k: {"errors": errors, "words": 10} for k in ("raw", "normalized")}, "heard": "x"}
            rows.append({**base, "rung": "onnx-fp32", "repeat": 0, "measures": same(), **heard(0)})
            if n < fp16_nan:
                rows.append({**base, "rung": "onnx-fp16", "repeat": 0, "finite": False, "nonfinite_samples": 100, "measures": None})
            else:
                rows.append({**base, "rung": "onnx-fp16", "repeat": 0, "measures": same(f0_offset_hz=-0.3), **heard(0)})
            rows.append({**base, "rung": "onnx-q8", "repeat": 0, "measures": same(f0_offset_hz=q8_offset, f0_dev_scale=0.93),
                         **heard(1 if n < q8_errors else 0)})
            for k in range(q.TORCH_REPEATS):
                rows.append({**base, "rung": "torch", "repeat": k, "measures": same(), **(heard(0) if k == 0 else {})})
                if k:
                    floor_rows.append({**base, "rung": "torch", "repeat": k, "against": "torch", "measures": same(f0_offset_hz=floor_hz)})
            n += 1
    return {"sr": q.SR, "reference": "onnx-fp32", "torch_repeats": q.TORCH_REPEATS, "voices": list(voices), "sentences": dict(sentences),
            "rungs": {r: {"file": f, "sha256": d, "size_mb": 1.0} for r, (f, d) in q.RUNGS.items()},
            "rows": rows, "floor_rows": floor_rows, "commit": "abc1234", "registration": "PR-013"}


def test_pr013s_sentences_are_new_and_its_voices_are_not_the_dev_voices():
    spent = set(severity.DEV_SENTENCES.values()) | set(severity.TEST_SENTENCES.values())
    other = (ROOT / "ft" / "src" / "ft3" / "pr012.py").read_text(encoding="utf-8")
    assert len(q.TEST_SENTENCES) == 24 and len(set(q.TEST_SENTENCES.values())) == 24
    for sid, text in q.TEST_SENTENCES.items():
        assert text not in spent and text not in other and sid not in q.DEV_SENTENCES
        assert text.count(". ") == 1 and text.count(",") == 1 and text.endswith(".")               # a sentence break, a comma
        assert 12 <= len(text.split()) <= 20
    assert q.TEST_VOICES == ("af_sarah", "am_adam") and not set(q.TEST_VOICES) & set(q.DEV_VOICES)
    assert (q.A1_MAX_SHARE, q.A3_MAX_WER_POINTS, q.B2_OFFSET_SHARE, q.B2_RANGE_SHARE) == (0.90, 2.0, 0.90, 0.80)
    assert (q.A1_MAX_SHARE, q.B2_RANGE_SHARE) == (severity.S4_OFFSET_SHARE, severity.S5_RANGE_SHARE)   # PR-009's bars


def test_claim_a_needs_the_pitch_outside_its_guardrail_and_the_words_inside_theirs():
    c = q.criteria(made_up())
    assert c["A1"] == {"within": 0, "of": 48, "share": 0.0, "pass": True}
    assert c["A2"]["mean"] == pytest.approx(-4.0) and c["A2"]["floor_p95"] == pytest.approx(0.5) and c["A2"]["pass"]
    assert c["A3"]["points"] == pytest.approx(100 * 3 / 480) and c["A3"]["hi"] <= 2.0 and c["A3"]["pass"]
    assert c["A3"]["quantized"] == {"errors": 3, "words": 480} and c["A3"]["reference"] == {"errors": 0, "words": 480}
    assert c["claim_a"] is True
    near = q.criteria(made_up(q8_offset=-1.0))                                 # a real shift, inside the guardrail
    assert near["A1"]["within"] == 48 and not near["A1"]["pass"] and near["A2"]["pass"] and not near["claim_a"]
    noise = q.criteria(made_up(q8_offset=-4.0, floor_hz=5.0))                  # a shift no bigger than the package's own
    assert noise["A1"]["pass"] and not noise["A2"]["pass"] and not noise["claim_a"]
    garbled = q.criteria(made_up(q8_errors=30))                                # the words went too
    assert garbled["A3"]["points"] == pytest.approx(6.25) and not garbled["A3"]["pass"] and not garbled["claim_a"]
    gone = made_up()
    for r in gone["rows"]:
        if r["rung"] == "onnx-q8":
            r.update(finite=False, measures=None)
            r.pop("wer"), r.pop("heard")
    c = q.criteria(gone)                                                       # the 8-bit copy returned no audio at all
    assert not c["A1"]["pass"] and not c["A2"]["pass"] and not c["A3"]["pass"] and not c["claim_a"]
    assert "cannot be computed" in q.format_criteria(gone)
    up = q.criteria(made_up(q8_offset=+4.0))                                   # the direction is not part of the claim
    assert up["A2"]["lo"] > 0.5 and up["claim_a"]


def test_claim_b_needs_audio_from_every_render():
    c = q.criteria(made_up())
    assert c["B1"] == {"audio": 47, "of": 48, "pass": False} and c["B2"] == {"offset_within": 47, "range_within": 47, "of": 47, "pass": True}
    assert c["claim_b"] is False
    whole = q.criteria(made_up(fp16_nan=0))
    assert whole["B1"]["pass"] and whole["claim_b"] is True
    none = q.criteria(made_up(fp16_nan=48))
    assert none["B1"]["audio"] == 0 and not none["B2"]["pass"] and not none["claim_b"]


def test_the_verdicts_are_worded_as_registered_and_a_run_on_another_set_is_not_the_run():
    text = q.format_criteria(made_up())
    assert "PR-013, Claim A: SUPPORTED" in text and "PR-013, Claim B: NOT SUPPORTED -- failed B1" in text
    assert "A1  8-bit pitch offset within 3 Hz on fewer than 90% of renders: 0 of 48 (0.0%)   PASS" in text
    assert "B1  fp16 returns audio for every render: 47 of 48   FAIL" in text
    assert q.validity(made_up()) == []
    dev = made_up(voices=q.DEV_VOICES, sentences=q.DEV_SENTENCES)
    assert "NOT THE REGISTERED RUN" in q.format_criteria(dev) and "Claim A:" not in q.format_criteria(dev).splitlines()[-1]
    for change, word in (({"commit": "abc-dirty"}, "clean committed tree"), ({"rungs": {}}, "pinned model files"),
                         ({"rows": made_up()["rows"][:-1]}, "383 rows")):
        problems = q.validity({**made_up(), **change})
        assert len(problems) == 1 and word in problems[0]


def test_the_registered_run_is_locked_clean_and_never_overwritten(tmp_path):
    run_path, audio, loaded = tmp_path / "runs" / "pr013.json", tmp_path / "audio", []

    def speakers():
        loaded.append(1)
        return FakeTorch(), ladder()

    kw = dict(run_path=run_path, audio_dir=audio, speakers=speakers, hear=lambda a: "the shop was closed")
    with pytest.raises(PermissionError, match="--confirm PR-013"):
        q.registered_run("PR-009", commit_fn=lambda: "abc1234", **kw)
    with pytest.raises(PermissionError, match="clean committed tree"):
        q.registered_run("PR-013", commit_fn=lambda: "abc1234-dirty", **kw)
    assert not loaded and not run_path.exists()                                   # every refusal before a model loads
    results = q.registered_run("PR-013", commit_fn=lambda: "abc1234", **kw)
    saved = json.loads(run_path.read_text())
    assert saved["registration"] == "PR-013" and saved["commit"] == "abc1234" and saved["set"] == "test"
    assert saved["voices"] == list(q.TEST_VOICES) and saved["sentences"] == q.TEST_SENTENCES
    assert len(saved["rows"]) == 48 * 8 and len(saved["floor_rows"]) == 48 * 4
    assert saved["criteria"]["B1"] == {"audio": 0, "of": 48, "pass": False}       # the fake fp16 never returns audio
    assert saved["criteria"]["A1"]["of"] == 48 and set(saved["summary"]) == {"onnx-fp32", "onnx-fp16", "onnx-q8", "torch"}
    assert len(list(audio.iterdir())) == 48 * 3                                   # fp32, 8-bit, torch; no file for what is not audio
    assert results["criteria"] == saved["criteria"]
    with pytest.raises(FileExistsError, match="never overwritten"):
        q.registered_run("PR-013", commit_fn=lambda: "abc1234", **kw)
