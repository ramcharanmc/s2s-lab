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

PersonaPlexDuplex -- NVIDIA's PersonaPlex (7B, bf16): Moshi fine-tuned so that a
text prompt sets its role and a voice prompt its voice. It comes with its own
package, also named `moshi` (github NVIDIA/personaplex, pinned by commit in the
image), and is driven the way that package's offline.py drives it:

    mimi, lm = loaders.get_mimi(file), loaders.get_moshi_lm(file); lm.eval()
    gen = LMGen(lm, audio_silence_frame_cnt=6, temp=0.8, temp_text=0.7,
                top_k=250, top_k_text=25, ...)                offline.py's defaults
    mimi.streaming_forever(1), gen.streaming_forever(1), four warm-up steps
    gen.load_voice_prompt_embeddings(voices/NATF2.pt)
    gen.text_prompt_tokens = tokenizer.encode("<system> role <system>")
    each call:  reset both streams; gen.step_system_prompts(mimi) -- the voice,
                0.5 s of silence, the role, 0.5 s of silence; reset Mimi; then
                tokens = gen.step(mimi.encode(frame))
                audio  = mimi.decode(tokens[:, 1:9])          the agent's 8 codebooks
                text   = tokens[0, 0, 0]                      0 and 3 are padding

The role is R0's instruction word for word (r0.SYSTEM_PROMPT), as R3 and R4 use
it; the voice is NATF2, the README's example. The prompts run in begin(),
before the stream, so they are in neither the stream time nor the RTF.
offline.py also decodes every frame through a second Mimi and throws the
result away; nothing heard depends on it, so it is left out.

Every conversation is seeded (duplex.SEED) and starts from a fresh streaming state.

GPU ONLY, and an image each: Kyutai's `moshi` pins torch below 2.10 and
NVIDIA's `moshi` below 2.5, and both take the same import name, so neither
shares the R0-R4 images (torch 2.14) or each other's. The tests use a fake model.
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


PERSONAPLEX_REPO = "nvidia/personaplex-7b-v1"
PERSONAPLEX_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"      # the weights on 2026-09-24
PERSONAPLEX_VOICE = "NATF2.pt"


def system_tags(text: str) -> str:
    """The role as PersonaPlex was trained to read it (offline.py's wrap_with_system_tags)."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class PersonaPlexDuplex:
    """PersonaPlex behind duplex.py's three calls: its voice and role are set again before every call."""

    rung_id = "r5-personaplex"
    description = ("R5 full duplex: NVIDIA PersonaPlex (7B, bf16, Moshi fine-tuned for a role and a voice) "
                   "-- R0's instruction as the role, voice NATF2")

    def __init__(self, repo: str = PERSONAPLEX_REPO, revision: str = PERSONAPLEX_REVISION,
                 voice: str = PERSONAPLEX_VOICE, role: str | None = None, device: str = "cuda",
                 temp: float = 0.8, temp_text: float = 0.7, top_k: int = 250, top_k_text: int = 25):
        from .r0 import SYSTEM_PROMPT
        self.repo, self.revision, self.voice, self.device = repo, revision, voice, device
        self.role = SYSTEM_PROMPT if role is None else role
        self.temp, self.temp_text, self.top_k, self.top_k_text = temp, temp_text, top_k, top_k_text
        self.mimi = self.lm = self.tokenizer = self.gen = None
        self._stack = None

    def _file(self, name: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, name, revision=self.revision)

    def load(self) -> None:
        import pathlib
        import tarfile

        import sentencepiece
        import torch
        from moshi.models import LMGen, loaders
        self.mimi = loaders.get_mimi(self._file(loaders.MIMI_NAME), self.device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(self._file(loaders.TEXT_TOKENIZER_NAME))
        self.lm = loaders.get_moshi_lm(self._file(loaders.MOSHI_NAME), device=self.device)
        self.lm.eval()
        self.gen = LMGen(self.lm, device=self.device, audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                         sample_rate=self.mimi.sample_rate, frame_rate=self.mimi.frame_rate, use_sampling=True,
                         temp=self.temp, temp_text=self.temp_text, top_k=self.top_k, top_k_text=self.top_k_text)
        self.mimi.streaming_forever(1)
        self.gen.streaming_forever(1)
        with torch.no_grad():                           # offline.py's warm-up: CUDA graphs, streaming state
            zeros = torch.zeros(1, 1, int(self.mimi.sample_rate / self.mimi.frame_rate), device=self.device)
            for _ in range(4):
                tokens = self.gen.step(self.mimi.encode(zeros))
                if tokens is not None:
                    self.mimi.decode(tokens[:, 1:9])
        torch.cuda.synchronize()
        tgz = pathlib.Path(self._file("voices.tgz"))
        voices = tgz.parent / "voices"
        if not voices.exists():                         # the image unpacks them at build time
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(path=tgz.parent)
        self.gen.load_voice_prompt_embeddings(str(voices / self.voice))
        self.gen.text_prompt_tokens = self.tokenizer.encode(system_tags(self.role))

    def begin(self, seed: int) -> None:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(torch.no_grad())
        try:
            self.mimi.reset_streaming()
            self.gen.reset_streaming()
            self.gen.step_system_prompts(self.mimi)
            self.mimi.reset_streaming()                 # as offline.py: the voice prompt went through Mimi
        except BaseException:
            self.end()
            raise

    def step(self, frame: np.ndarray):
        import torch
        x = torch.from_numpy(np.asarray(frame, dtype=np.float32)).to(self.device)[None, None]
        tokens = self.gen.step(self.mimi.encode(x))
        if tokens is None:
            return None, None
        audio = self.mimi.decode(tokens[:, 1:9]).float().cpu().numpy().reshape(-1)
        t = int(tokens[0, 0, 0].item())
        return audio, None if t in PAD_PIECES else self.tokenizer.id_to_piece(t).replace("▁", " ")

    def end(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None


MODELS = {"r5-moshi": MoshiDuplex, "r5-personaplex": PersonaPlexDuplex}


def get(model_id: str):
    try:
        return MODELS[model_id]()
    except KeyError:
        raise KeyError(f"unknown full-duplex model {model_id!r}; known: {', '.join(MODELS)}") from None
