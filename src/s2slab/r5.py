"""
R5's models, as duplex.py drives them: begin(seed), step(80 ms frame) -> (audio, text), end().

MoshiDuplex -- Kyutai's Moshi (moshiko, 7B, bf16) through the `moshi` package:
Mimi encodes each 80 ms caller frame into 8 codebooks, LMGen predicts the next
step of BOTH streams -- the caller's and its own -- plus a text token (its
"inner monologue"), and Mimi decodes its own audio tokens back to 24 kHz.
The calls are the ones in the package's README and moshi/server.py:

    info = loaders.CheckpointInfo.from_hf_repo(repo)
    mimi, lm, tok = info.get_mimi(...), info.get_moshi(...), info.get_text_tokenizer()
    gen = LMGen(lm, temp=0.8, temp_text=0.7)            the package's own defaults: it SAMPLES
    with gen.streaming(1), mimi.streaming(1):
        tokens = gen.step(mimi.encode(frame))           None while its delays fill
        audio  = mimi.decode(tokens[:, 1:])             its 8 audio codebooks
        text   = tokens[0, 0, 0]                        pieces 0 and 3 are padding

Every conversation is seeded (duplex.SEED) and starts from a fresh streaming state.

GPU ONLY, and its own image: the `moshi` package pins its own torch (the
README says 2.2 or 2.4), so it does not share the R0-R4 images (torch 2.14).
PersonaPlex (NVIDIA, built on the same architecture) comes next as a second
adapter; its code installs a package also named `moshi`, so it gets an image
of its own too. The tests use a fake model.
"""

from __future__ import annotations

import contextlib

import numpy as np

MOSHI_REPO = "kyutai/moshiko-pytorch-bf16"
PAD_PIECES = (0, 3)          # text-stream ids that mean "nothing said this step" (moshi/server.py)


class MoshiDuplex:
    """Moshi behind duplex.py's three calls."""

    rung_id = "r5-moshi"
    description = "R5 full duplex: Kyutai Moshi (moshiko, 7B, bf16) -- no turns, both streams at once"

    def __init__(self, repo: str = MOSHI_REPO, device: str = "cuda", temp: float = 0.8, temp_text: float = 0.7):
        self.repo, self.device, self.temp, self.temp_text = repo, device, temp, temp_text
        self.mimi = self.lm = self.tokenizer = self.gen = None
        self._stack = None

    def load(self) -> None:
        import torch
        from moshi.models import LMGen, loaders
        info = loaders.CheckpointInfo.from_hf_repo(self.repo)
        self.mimi = info.get_mimi(device=self.device)
        self.lm = info.get_moshi(device=self.device, dtype=torch.bfloat16)
        if hasattr(self.lm, "dep_q"):                   # Moshi reads 8 of Mimi's codebooks
            self.mimi.set_num_codebooks(self.lm.dep_q)
        self.tokenizer = info.get_text_tokenizer()
        self.gen = LMGen(self.lm, temp=self.temp, temp_text=self.temp_text)

    def begin(self, seed: int) -> None:
        import torch
        torch.manual_seed(seed)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(torch.no_grad())
        self._stack.enter_context(self.gen.streaming(1))
        self._stack.enter_context(self.mimi.streaming(1))

    def step(self, frame: np.ndarray):
        import torch
        x = torch.from_numpy(np.asarray(frame, dtype=np.float32)).to(self.device)[None, None]
        tokens = self.gen.step(self.mimi.encode(x))
        if tokens is None:
            return None, None
        audio = self.mimi.decode(tokens[:, 1:]).float().cpu().numpy().reshape(-1)
        t = int(tokens[0, 0, 0].item())
        return audio, None if t in PAD_PIECES else self.tokenizer.id_to_piece(t).replace("▁", " ")

    def end(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None


MODELS = {"r5-moshi": MoshiDuplex}


def get(model_id: str):
    try:
        return MODELS[model_id]()
    except KeyError:
        raise KeyError(f"unknown full-duplex model {model_id!r}; known: {', '.join(MODELS)}") from None
