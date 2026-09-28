"""
R2 with FT-2's recognizer: the two-tier cascade whose slow tier writes down what was said as it
was said, fillers and repetitions kept.

FT-2 (docs/ft2-log.md, PR-011) trained a LoRA on whisper-small.en from verbatim transcripts, merged
it and converted it to CTranslate2. On the eval set's disfluent cohort it took raw WER from 9.9% to
2.3% and kept 29 of 29 fillers where small.en kept 1. Until now no rung loaded it: the model sat
on the Modal volume `ft2`, and the test ran it through the recognizer class alone.

    r2-verbatim        the export in place of small.en, on the CPU (int8)
    r2-gpu-verbatim    the same in fp16 on a CUDA GPU, beside r2-gpu

THE MODEL IS HELD TO PR-011'S. The export is a folder in faster-whisper's layout (model.bin and
the files beside it); its model.bin must have the SHA-256 PR-011 registered, or the rung refuses
to load. Where the folder is: the environment variable S2SLAB_FT2_CT2, or data/ft2/ct2.

WHAT IT CANNOT CLAIM. PR-011's result is one speaker's performed disfluency, trained on and tested
on: same-speaker adaptation. A visitor's "um" is another speaker's; nothing measured here says the
model writes it down.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

from .r0 import KokoroTTS, WhisperASR
from .r1 import StreamingCascade, StreamingQwenLLM
from .r2 import TwoTierCascade

EXPORT_ENV = "S2SLAB_FT2_CT2"
EXPORT_DEFAULT = "data/ft2/ct2"                # ignored by git
EXPORT_SHA256 = "b31833fe4ed0e1cb429d33a52d81e9974fe32838e32e9a85f19e666af2a51cff"    # PR-011's ct2/model.bin
TIER_B_VERBATIM = "small.en + FT-2's verbatim LoRA"
EXPORT_NAME = "export-runs_train-f37d0df-20260926-235222_csv_version_0_adapters_step-000900-float16"   # on the volume ft2


def where(path=None) -> pathlib.Path:
    return pathlib.Path(path or os.environ.get(EXPORT_ENV) or EXPORT_DEFAULT)


def export_dir(path=None, sha256: str | None = None) -> pathlib.Path:
    """The export's folder, its model.bin the registered one. Raises before anything is loaded."""
    folder, sha256 = where(path), sha256 or EXPORT_SHA256
    model = folder / "model.bin"
    if not model.exists():
        raise FileNotFoundError(f"no exported model at {folder}: set {EXPORT_ENV}, or put PR-011's export in {EXPORT_DEFAULT}")
    h = hashlib.sha256()
    with open(model, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != sha256:
        raise ValueError(f"{model} has SHA-256 {h.hexdigest()}, PR-011's model has {sha256}")
    return folder


def available(path=None, sha256: str | None = None) -> bool:
    """Is PR-011's export here? False for no export and for another model in its place: a tier is
    offered only where it can load."""
    try:
        export_dir(path, sha256)
    except (FileNotFoundError, ValueError):
        return False
    return True


class VerbatimASR(WhisperASR):
    """The rungs' recognizer class, loading FT-2's export from its folder."""

    def __init__(self, path=None, device: str = "cpu", compute_type: str = "int8", beam_size: int = 1,
                 sha256: str | None = None):
        super().__init__(TIER_B_VERBATIM, device, compute_type, beam_size)
        self.path, self.sha256 = path, sha256

    def load(self) -> None:
        from faster_whisper import WhisperModel
        folder = export_dir(self.path, self.sha256)                 # refuses here, before the model is read
        self.model = WhisperModel(str(folder), device=self.device, compute_type=self.compute_type)


class TwoTierCascadeVerbatim(TwoTierCascade):
    """R2, its slow tier FT-2's recognizer."""

    rung_id = "r2-verbatim"
    description = "two-tier cascade: small.en + FT-2's verbatim LoRA (slow tier) -> R1's streamed Qwen2.5-0.5B -> Kokoro-82M"
    tier_b = TIER_B_VERBATIM

    def __init__(self, asr=None, llm=None, tts=None):
        super().__init__(asr or VerbatimASR(), llm, tts)


class TwoTierCascadeGPUVerbatim(TwoTierCascadeVerbatim):
    """The same beside r2-gpu: same LLM, same voice, same precision; only the recognizer differs."""

    rung_id = "r2-gpu-verbatim"
    description = "R2 on a CUDA GPU (fp16): slow tier small.en + FT-2's verbatim LoRA"

    def __init__(self):
        StreamingCascade.__init__(self, VerbatimASR(device="cuda", compute_type="float16"),
                                  StreamingQwenLLM(device="cuda"), KokoroTTS(device="cuda"))
