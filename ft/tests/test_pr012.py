"""PR-012's lock, pooling and verdict (src.ft3.pr012), on a fake voice and made-up measures: no NeMo,
no audio, and none of the eight sentences is spoken here."""
import hashlib
import json
import types

import pytest

from s2slab import severity
from src.ft3 import pr012

FILES = {v: {"fastpitch": f"{v}.nemo", "hifigan": f"{v}-voc.nemo"} for v in pr012.VOICES}


def fake_load(fastpitch, hifigan):
    return types.SimpleNamespace(name=fastpitch.removesuffix(".nemo"), sr=22050)


def fake_sweep(miss=()):
    """sweep.run's shape with every target met, except the (voice, field) pairs in `miss`."""
    def run(v, sentences, listen=None):
        rows = []
        for sid in sentences:
            for condition, levels in severity.SCORES.items():
                steps = []
                for score in levels:
                    t = severity.CONDITIONS[condition](score)
                    target = {f: getattr(t, f) for f in severity.FIELDS}
                    got = {f: (1.0 if f != "f0_offset_hz" else 0.0) if (v.name, f) in miss else x for f, x in target.items()}
                    steps.append({"score": score, "target": target, "achieved": got, "misses": []})
                rows.append({"voice": v.name, "sentence": sid, "condition": condition, "monotone": [], "steps": steps})
        return {"sr": v.sr, "tolerance": severity.TOLERANCE, "rows": rows, "settings": dict(pr012.SETTINGS),
                "voice_info": {"name": v.name}}
    return run


def test_pr012s_sentences_are_new_and_its_shape_is_pr009s():
    used = set(severity.DEV_SENTENCES.values()) | set(severity.TEST_SENTENCES.values())
    assert len(pr012.SENTENCES) == 8 and not used & set(pr012.SENTENCES.values())
    assert not set(pr012.SENTENCES) & (set(severity.DEV_SENTENCES) | set(severity.TEST_SENTENCES))      # ids too
    for text in pr012.SENTENCES.values():
        assert text.count(". ") == 1 and text.count(",") == 1 and text.endswith(".")               # a sentence break, a comma
    assert len(pr012.VOICES) * len(pr012.SENTENCES) * len(severity.SCORES) == 32                   # S2's bar is of 32 sweeps
    assert pr012.SETTINGS == {"centre_mode": "audio", "range_gain": 1.2, "min_pause_s": 0.15}
    from src.ft3 import control, synth
    assert (synth.CENTRE_MODE, control.RANGE_GAIN, control.MIN_PAUSE_S) == ("audio", 1.2, 0.15)      # the defaults the run uses
    for models in pr012.VOICES.values():
        assert set(models) == {"fastpitch", "hifigan"} and all(len(d) == 64 for _, d in models.values())


def test_the_model_files_are_checked_by_sha256_before_anything_loads(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "fp.nemo").write_bytes(b"fastpitch")
    (tmp_path / "voc.nemo").write_bytes(b"hifigan")
    voices = {"v": {"fastpitch": ("*/fp.nemo", hashlib.sha256(b"fastpitch").hexdigest()),
                    "hifigan": ("voc.nemo", hashlib.sha256(b"hifigan").hexdigest())}}
    files = pr012.model_files(voices, root=tmp_path)
    assert files == {"v": {"fastpitch": str(tmp_path / "a" / "fp.nemo"), "hifigan": str(tmp_path / "voc.nemo")}}
    (tmp_path / "voc.nemo").write_bytes(b"another vocoder")
    with pytest.raises(ValueError, match="the registered file has"):
        pr012.model_files(voices, root=tmp_path)
    (tmp_path / "voc.nemo").unlink()
    with pytest.raises(FileNotFoundError, match="0 files match"):
        pr012.model_files(voices, root=tmp_path)


def test_the_registered_run_is_locked_clean_and_never_overwritten(tmp_path):
    run_path, audio = tmp_path / "runs" / "pr012.json", tmp_path / "audio"
    loaded = []
    load = lambda fp, voc: loaded.append(fp) or fake_load(fp, voc)
    kw = dict(run_path=run_path, audio_dir=audio, files=FILES, load=load, sweep_run=fake_sweep())
    with pytest.raises(PermissionError, match="--confirm PR-012"):
        pr012.registered_run("PR-009", commit_fn=lambda: "abc1234", **kw)
    with pytest.raises(PermissionError, match="clean committed tree"):
        pr012.registered_run("PR-012", commit_fn=lambda: "abc1234-dirty", **kw)
    assert not loaded and not run_path.exists()                                   # every refusal before a model loads
    results = pr012.registered_run("PR-012", commit_fn=lambda: "abc1234", **kw)
    saved = json.loads(run_path.read_text())
    assert saved["registration"] == "PR-012" and len(saved["rows"]) == 32 and saved["commit"] == "abc1234"
    assert saved["voices"] == list(pr012.VOICES) and saved["sentences"] == pr012.SENTENCES
    assert saved["criteria"]["supported"] and pr012.validity(results) == []
    assert pr012.format_criteria(results).splitlines()[6] == "PR-012: SUPPORTED"
    with pytest.raises(FileExistsError, match="never overwritten"):
        pr012.registered_run("PR-012", commit_fn=lambda: "abc1234", **kw)


def test_the_verdict_pools_both_voices_and_reports_each_alone(tmp_path):
    results = pr012.registered_run("PR-012", run_path=tmp_path / "r.json", audio_dir=tmp_path / "a", files=FILES,
                                   load=fake_load, sweep_run=fake_sweep(miss={("fastpitch_9017", "f0_dev_scale")}),
                                   commit_fn=lambda: "abc1234")
    c, each = results["criteria"], results["criteria_by_voice"]
    assert (c["S5"]["of"], c["S4"]["of"], c["S3"]["of"], c["S1"]["of"], c["S2"]["of"]) == (192, 192, 384, 96, 32)
    assert each["tts_en_fastpitch_align"]["S5"] == {"met": 96, "of": 96}
    # the fine-tuned voice's range stays at 1.00: on target only where the target is 1.00 (score 0, both conditions)
    assert each["fastpitch_9017"]["S5"] == {"met": 16, "of": 96} and each["fastpitch_9017"]["S2"] == {"met": 0, "of": 16}
    assert c["S5"]["met"] == 112 and not c["S5"]["pass"] and c["S2"]["met"] == 16 and not c["supported"]
    text = pr012.format_criteria(results)
    assert "PR-012: NOT SUPPORTED -- failed S2, S5" in text and "rate within 10%: 192 of 192" in text
    assert "fastpitch_9017" in text and "S5 16 of 96" in text


def test_a_run_on_other_sentences_settings_or_files_is_not_the_registered_run(tmp_path):
    good = pr012.registered_run("PR-012", run_path=tmp_path / "r.json", audio_dir=tmp_path / "a", files=FILES,
                                load=fake_load, sweep_run=fake_sweep(), commit_fn=lambda: "abc1234")
    for change, word in (({"sentences": severity.DEV_SENTENCES}, "voices and sentences"),
                         ({"settings": {**pr012.SETTINGS, "range_gain": 1.0}}, "settings"),
                         ({"model_sha256": {}}, "model files"), ({"commit": "abc-dirty"}, "clean committed tree"),
                         ({"rows": good["rows"][:30]}, "30 sweeps")):
        problems = pr012.validity({**good, **change})
        assert len(problems) == 1 and word in problems[0]
        assert "NOT THE REGISTERED RUN" in pr012.format_criteria({**good, **change})
    wrong = lambda fp, voc: types.SimpleNamespace(name="someone_else", sr=22050)
    with pytest.raises(ValueError, match="names itself"):
        pr012.run(FILES, severity.DEV_SENTENCES, load=wrong, sweep_run=fake_sweep())
