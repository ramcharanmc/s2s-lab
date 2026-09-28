"""
R2 -- the two-tier cascade: a fast recognizer for the moment, a slow one for the words that matter.

The v2 plan, R2: "Fast tier drives the UI and barge-in; slow tier drives the
action. Resolves the contradiction between real-time partials and big-model
accuracy -- no single tier does both."

    while the caller talks     TIER A  Whisper base.en, a transcript every 200 ms
                                       -> barge-in decisions (bargein_asr.py, PR-003)
    once the turn is finished  TIER B  a larger Whisper, once, over the whole turn
                                       -> [llm] -> clause by clause -> [tts]     exactly as R1

TIER A IS ALREADY BUILT AND TESTED. It is R0's and R1's recognizer, and the one
the barge-in policies read. Its job needs speed, not accuracy: "no, wait" and
"okay" survive a small model.

WHAT THIS FILE ADDS IS TIER B, and nothing else. A rung receives a FINISHED
turn, so inside respond() the fast tier has no work left: R2 here is R1 with
the slow tier's recognizer on the critical path. The LLM, its prompt, the
chunker and the voice are R1's own objects, so any difference from R1 -- in
what is heard, in what is said, in when the first audio exists -- is the
recognizer's.

WHAT IT SHOULD BUY, AND COST. R1 mishears 11.9% of words, 26.5% on accented
speech, and a misheard word becomes the answer (f02: "half pass too" -> "half
time"). Tier B should hear more; it also sits in front of the first audio, so
every ms it takes is a ms the caller waits. The eval-set run reports both, for
two sizes of Tier B.

    r2             small.en         2.5x base.en's parameters, English only
    r2-turbo       large-v3-turbo   large-v3's encoder with a 4-layer decoder, multilingual
    r2-gpu, r2-gpu-turbo            the same two in fp16 on a CUDA GPU, beside r1-gpu
    r2-verbatim, r2-gpu-verbatim    small.en with FT-2's LoRA, from its export: r2verbatim.py

    PYTHONPATH=src python -m s2slab.r0 probes/endpointing/audio/f02.wav r2   one turn, reply saved
    PYTHONPATH=src python -m s2slab.rungs r2                                 30 probe clips
"""

from __future__ import annotations

from .r0 import KokoroTTS, WhisperASR
from .r1 import StreamingCascade, StreamingQwenLLM

TIER_A = "base.en"              # the fast tier: R0's and R1's recognizer, and the barge-in policies'
TIER_B = "small.en"
TIER_B_TURBO = "large-v3-turbo"


class TwoTierCascade(StreamingCascade):
    """R2: R1, with the slow tier's recognizer deciding what was said."""

    rung_id = "r2"
    description = "two-tier cascade: Whisper small.en (slow tier) -> R1's streamed Qwen2.5-0.5B -> Kokoro-82M"
    tier_b = TIER_B

    def __init__(self, asr=None, llm=None, tts=None):
        super().__init__(asr or WhisperASR(self.tier_b), llm, tts)


class TwoTierCascadeTurbo(TwoTierCascade):
    rung_id = "r2-turbo"
    description = "two-tier cascade: Whisper large-v3-turbo (slow tier) -> R1's streamed Qwen2.5-0.5B -> Kokoro-82M"
    tier_b = TIER_B_TURBO


class TwoTierCascadeGPU(TwoTierCascade):
    """R2 beside r1-gpu: same LLM, same voice, same precision; only the recognizer differs."""

    rung_id = "r2-gpu"
    description = "R2 on a CUDA GPU (fp16): slow tier Whisper small.en"

    def __init__(self):
        super().__init__(WhisperASR(self.tier_b, device="cuda", compute_type="float16"),
                         StreamingQwenLLM(device="cuda"), KokoroTTS(device="cuda"))


class TwoTierCascadeGPUTurbo(TwoTierCascadeGPU):
    rung_id = "r2-gpu-turbo"
    description = "R2 on a CUDA GPU (fp16): slow tier Whisper large-v3-turbo"
    tier_b = TIER_B_TURBO