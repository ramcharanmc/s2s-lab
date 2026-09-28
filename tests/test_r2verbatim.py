"""R2 with FT-2's recognizer (s2slab.r2verbatim): where the export is, that it is PR-011's, and the
rung around it -- on fake models and a made-up export."""
import hashlib

import numpy as np
import pytest

from s2slab import r2verbatim as rv
from s2slab.rungs import RUNGS, get


def export(tmp_path, content=b"a model"):
    folder = tmp_path / "ct2"
    folder.mkdir(exist_ok=True)
    (folder / "model.bin").write_bytes(content)
    return folder, hashlib.sha256(content).hexdigest()


def test_the_export_is_found_by_argument_then_environment_then_default(tmp_path, monkeypatch):
    monkeypatch.delenv(rv.EXPORT_ENV, raising=False)
    assert str(rv.where()).replace("\\", "/") == rv.EXPORT_DEFAULT == "data/ft2/ct2"
    monkeypatch.setenv(rv.EXPORT_ENV, str(tmp_path / "from-env"))
    assert rv.where() == tmp_path / "from-env" and rv.where(tmp_path / "given") == tmp_path / "given"
    assert not rv.available()
    folder, digest = export(tmp_path)
    assert rv.available(folder, digest) and not rv.available(folder)           # another model in its place is not the export


def test_the_export_is_held_to_pr011s_sha256_before_anything_loads(tmp_path):
    folder, digest = export(tmp_path)
    assert rv.export_dir(folder, digest) == folder
    with pytest.raises(ValueError, match="PR-011's model has"):
        rv.export_dir(folder)                                        # a model, but not the registered one
    with pytest.raises(FileNotFoundError, match="no exported model"):
        rv.export_dir(tmp_path / "nowhere")
    asr = rv.VerbatimASR(folder)
    with pytest.raises(ValueError, match="PR-011's model has"):
        asr.load()
    assert asr.model is None and len(rv.EXPORT_SHA256) == 64


def test_the_rungs_are_registered_and_differ_from_r2_in_the_recognizer_alone():
    assert RUNGS["r2-verbatim"] == "s2slab.r2verbatim:TwoTierCascadeVerbatim"
    assert RUNGS["r2-gpu-verbatim"] == "s2slab.r2verbatim:TwoTierCascadeGPUVerbatim"
    rung, r2 = get("r2-verbatim"), get("r2")
    assert rung.rung_id == "r2-verbatim" and isinstance(rung.asr, rv.VerbatimASR) and not rung._loaded
    assert (rung.asr.device, rung.asr.compute_type, rung.asr.beam_size) == (r2.asr.device, r2.asr.compute_type, r2.asr.beam_size)
    assert type(rung.llm) is type(r2.llm) and type(rung.tts) is type(r2.tts)
    assert rung.asr.model_name == rung.tier_b == rv.TIER_B_VERBATIM and rung.tier_b in rung.description
    gpu = rv.TwoTierCascadeGPUVerbatim()
    assert (gpu.rung_id, gpu.asr.device, gpu.asr.compute_type, gpu.llm.device, gpu.tts.device) == ("r2-gpu-verbatim", "cuda", "float16", "cuda", "cuda")


def test_a_turn_goes_through_the_verbatim_recognizer_and_r1s_streamed_tail():
    given = []                                   # what the LLM was handed: the warm-up's sentence, then the turn's

    class Heard:
        device, compute_type, beam_size = "cpu", "int8", 1

        def load(self):
            pass

        def transcribe(self, audio):
            return "And, uh, they have a policy."

    class Says:
        def load(self):
            pass

        def stream(self, transcript):
            given.append(transcript)
            yield from ("Which ", "policy? ", "Tell me more.")

        def reply(self, transcript):
            return "Hello."

    class Speaks:
        SR = 24_000

        def load(self):
            pass

        def speak(self, text):
            return np.full(2_400, 0.1, dtype=np.float32)

    turn = rv.TwoTierCascadeVerbatim(Heard(), Says(), Speaks()).respond(np.zeros(16_000, np.float32), 16_000)
    assert turn.transcript == given[-1] == "And, uh, they have a policy." and turn.reply_text.startswith("Which policy?")
    assert list(turn.stages_ms) == ["asr", "llm", "tts"] and len(turn.chunks) == 2 and turn.first_audio_ms < turn.total_ms
