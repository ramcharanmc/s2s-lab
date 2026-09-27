"""The quantization ladder's comparison, noise floor and table (s2slab.quantladder), on a synthetic voice
and fake speakers: no Kokoro, no ONNX file, no recognizer."""
import hashlib
import json

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
