"""
R4, fast and streamed (M31): the same model, prompt, greedy text, voice and seed as R4.

    r4-gpu-fast     R4 with the Talker's decoder layers compiled (the speed probe's static+layers):
                    the whole reply still decoded at once, so first audio = total
    r4-gpu-stream   the same Talker, and the reply decoded and handed out a block at a time as the
                    Talker writes it (Rung.emit), with the DiT compiled for those blocks

Three rungs from R4 to the stream, so each change is measured on its own: R4 to r4-gpu-fast is what
compiling the Talker buys; r4-gpu-fast to r4-gpu-stream is what streaming buys, as R1 is to R0.

WHY THIS WAY (docs/m31-log.md, steps 1 and 1b). R4's Talker took 75% of its time at real time, 19.8 ms
a code, and token2wav cost ~860 ms a call whatever the length. Compiled, the DiT took 219, 273 and
421 ms for 25, 50 and 100 codes; the Talker 12.7 ms a code, 0.63 of real time, once it had a static
cache, its plain 2-D mask (r4speed.keep_2d_mask) and its layers compiled without a mask on one-code
steps (r4speed.compile_layers). A static cache of the Talker's own is made once here and reset each
turn: transformers would size a new one to each prompt, and every longer prompt would recompile.

THE BLOCKS (BlockDecoder). The DiT attends within blocks of `block_size` mel frames (two frames a
code), one block back in two of its layers and one block ahead in one: what a frame depends on reaches
two blocks back and one ahead. So a block of codes [s, e) is decoded in a window [s - LEFT, e + RIGHT)
of codes, LEFT two DiT blocks and RIGHT one, the window aligned to the DiT's blocks, and only [s, e)
is kept: the attention it sees is the attention the whole reply would give it. The noise is not: each
window draws its own, so neighbouring blocks are crossfaded over one code (20 ms). A block can be
decoded once RIGHT more codes exist, or the reply has ended. The DiT runs compiled on the window
lengths a reply's steady blocks use (CUDA graphs need fixed shapes) and eager on the last, shorter one.

THE TALKER IS NOT DISTURBED. Each window is decoded inside the Talker's own generate call (the
streamer hands each code over as it is made) and inside torch.random.fork_rng, so the DiT's noise
draws do not move the Talker's sampling: r4-gpu-stream writes the same codes as r4-gpu-fast on the
same clip and seed, which the development run checks.

    modal run modal_app/r4_bench.py::streamdev      R4, r4-gpu-fast and the stream's settings on the 30 probe clips
"""

from __future__ import annotations

import numpy as np

from .r0 import ASR_SR, clean_for_speech
from .r4 import OUT_SR, OmniS2S, SpeechToSpeech
from .rungs import Reply, Rung
from .timing import stage

MAX_CACHE = 6144              # the Talker's static cache: its prompt + 4,096 codes
TALKER_MAX_NEW = 4096         # the Talker's own default cap, kept
MAX_TURN_S = 60.0             # ~25 prompt tokens a second of audio, plus ~110 of text: 60 s is ~1,610 + 4,096
EOS = (8292, 8294)            # the Talker's end codes (generate's talker_eos_token_id)
SAMPLES_PER_CODE = 480        # 24 kHz at 50 codes a second (measured in step 1: 480 on every reply)
XFADE_CODES = 1               # neighbouring blocks overlap by one code, 20 ms, and are crossfaded
# The stream's setting, frozen for PR-015: on the 30 probe clips (runs/r4-gpu/stream-dev-l40s.json), the
# lowest first-sound p50 among the settings and start buffers with no stalled turn (docs/m31-log.md, step 2).
FIRST_CODES = 24
BLOCK_CODES = 96
START_BUFFER_S = 0.5


class FastOmni:
    """R4's model, with the speed probe's Talker fixes installed for good, and the DiT compilable for
    chosen window lengths. Wraps an OmniS2S, loaded or not, so one set of weights can serve R4 first
    and the fast rungs after it (streamdev does).

    speak() is OmniS2S.speak with the Talker's static cache (reset each turn) and transformers' own
    compile turned off; extra arguments pass through (the stream adds its streamer)."""

    def __init__(self, base: OmniS2S | None = None, compile_fn=None, cache_factory=None):
        self.base = base or OmniS2S()
        self.compile_fn, self.cache_factory = compile_fn, cache_factory
        self.cache = None
        self.dit_lengths: set[int] = set()       # window lengths in codes the DiT runs compiled
        self._dit_eager = self._dit_compiled = None

    def __getattr__(self, name):                 # model, processor, speaker, device, seed, sr...
        return getattr(self.__dict__["base"], name)

    def load(self) -> None:
        from .r4speed import _torch_compile, compile_layers, keep_2d_mask
        if self.cache is not None:                  # once: the wrappers must not be installed twice
            return
        if self.base.model is None:
            self.base.load()
        talker = self.base.model.talker
        self.cache = self.make_cache(MAX_CACHE)
        keep_2d_mask(talker)
        compile_layers(talker, self.compile_fn or _torch_compile)

    def make_cache(self, places: int):
        """A static cache of the Talker's, this many places (M34 keeps a smaller one beside MAX_CACHE's)."""
        talker = self.base.model.talker
        if self.cache_factory is None:
            from transformers import StaticCache
            return StaticCache(config=talker.config, max_cache_len=places)
        return self.cache_factory(talker.config, places)

    def compile_dit(self, lengths) -> None:
        """Run the DiT compiled on windows of these lengths (in codes), eager on any other."""
        from .r4speed import _mark_step, _torch_compile
        dit = self.base.model.token2wav.code2wav_dit_model
        if self._dit_eager is None:
            self._dit_eager = dit.forward
            self._dit_compiled = (self.compile_fn or _torch_compile)(self._dit_eager, mode="reduce-overhead",
                                                                    dynamic=False)
            frames = lambda: {n * dit.repeats for n in self.dit_lengths}

            def forward(*args, **kwargs):
                h = kwargs.get("hidden_states", args[0] if args else None)
                if h is not None and h.shape[1] in frames():
                    _mark_step()
                    return self._dit_compiled(*args, **kwargs).clone()
                return self._dit_eager(*args, **kwargs)

            dit.forward = forward
        self.dit_lengths |= set(lengths)

    def speak(self, audio: np.ndarray, sr: int, **extra) -> tuple[str, np.ndarray]:
        if len(audio) / sr > MAX_TURN_S:
            raise ValueError(f"a turn over {MAX_TURN_S:g} s would leave the Talker's cache ({MAX_CACHE}) too short "
                             f"for {TALKER_MAX_NEW} codes")
        import torch
        with torch.inference_mode():                # its tensors were made inside generate's inference mode
            self.cache.reset()
        return self.base.speak(audio, sr, talker_past_key_values=self.cache, talker_disable_compile=True,
                               talker_max_new_tokens=TALKER_MAX_NEW, **extra)


class FastSpeechToSpeech(SpeechToSpeech):
    """R4 with the Talker compiled: the reply decoded whole, as R4's is."""

    rung_id = "r4-gpu-fast"
    description = ("R4 with its Talker's decoder layers compiled (static cache, CUDA graphs); the reply decoded "
                   "whole: Qwen2.5-Omni-7B, bf16, CUDA")

    def __init__(self, model=None):
        super().__init__(model or FastOmni())


class CodeStreamer:
    """generate's streamer for the Talker: the first put is the prompt; each later put is one new code,
    handed to on_code until an end code."""

    def __init__(self, on_code):
        self.on_code, self.prompt_seen, self.done, self.codes = on_code, False, False, []

    def put(self, value) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.done:
            return
        for code in np.asarray(value).reshape(-1).tolist():
            if int(code) in EOS:
                self.done = True
                return
            self.codes.append(int(code))
            self.on_code(int(code))

    def end(self) -> None:
        pass


class BlockDecoder:
    """The Talker's codes in, audio out a block at a time: the first block FIRST codes, then BLOCK.

    decode_window(codes) must return the audio of a window of codes, SAMPLES_PER_CODE a code. A block
    [s, e) is decoded in the window [s - min(left, s), min(e + right, n)) and [s - XFADE, e) of it kept
    (just [0, e) for the first); the overlap is crossfaded with the tail held back from the block
    before. push() returns the pieces now ready; finish() the rest, the held tail last.
    """

    def __init__(self, decode_window, first: int, block: int, left: int, right: int,
                 xfade: int = XFADE_CODES, spc: int = SAMPLES_PER_CODE):
        self.decode_window, self.first, self.block, self.left, self.right = decode_window, first, block, left, right
        self.xf, self.spc = xfade * spc, spc
        self.codes: list[int] = []
        self.s = 0
        self.held: np.ndarray | None = None
        self.windows: list[tuple[int, int]] = []           # (window start, window end) as decoded

    def _size(self) -> int:
        return self.first if self.s == 0 else self.block

    def _decode(self, e: int, final: bool) -> np.ndarray:
        s, n = self.s, len(self.codes)
        ws, we = s - min(self.left, s), min(e + self.right, n)
        self.windows.append((ws, we))
        audio = np.asarray(self.decode_window(self.codes[ws:we]), dtype=np.float32)
        start = (s - ws) * self.spc - (self.xf if s > 0 else 0)
        piece = audio[start:(e - ws) * self.spc]
        if self.held is not None:
            ramp = np.linspace(0.0, 1.0, self.xf, endpoint=False, dtype=np.float32)
            head = piece[:self.xf] * ramp + self.held * (1.0 - ramp)
            piece = np.concatenate([head, piece[self.xf:]])
            self.held = None
        if not final:
            piece, self.held = piece[:-self.xf], piece[-self.xf:].copy()
        self.s = e
        return piece

    def push(self, code: int) -> list[np.ndarray]:
        self.codes.append(int(code))
        out = []
        while len(self.codes) >= self.s + self._size() + self.right:
            out.append(self._decode(self.s + self._size(), final=False))
        return out

    def finish(self) -> list[np.ndarray]:
        out, n = [], len(self.codes)
        while self.s < n:
            e = min(self.s + self._size(), n)
            out.append(self._decode(e, final=(e == n)))
        if self.held is not None:
            out.append(self.held)
            self.held = None
        return [p for p in out if len(p)]


class StartBuffer:
    """Holds the first pieces back until `seconds` of audio exist (or the reply ends), then hands each
    piece on as it comes. 0 hands every piece on at once."""

    def __init__(self, emit, seconds: float, sr: int = OUT_SR):
        self.emit, self.need, self.held, self.open = emit, int(seconds * sr), [], seconds <= 0

    def add(self, pieces) -> None:
        for p in pieces:
            if self.open:
                self.emit(p)
                continue
            self.held.append(p)
            if sum(len(h) for h in self.held) >= self.need:
                self.flush()

    def flush(self) -> None:
        if self.held:
            self.emit(np.concatenate(self.held))
            self.held = []
        self.open = True


def check_blocks(model, settings) -> None:
    """Every first and later block a whole number of the DiT's blocks, or the windows would not align."""
    dit = model.token2wav.code2wav_dit_model
    per = dit.block_size // dit.repeats
    bad = [s for s in settings if s[0] % per or s[1] % per]
    if bad:
        raise ValueError(f"blocks {bad} (codes) do not align with the DiT's blocks of {per} codes")


def windows_for(first: int, block: int, left: int, right: int) -> tuple[int, ...]:
    """The window lengths, in codes, that a long reply's blocks use: the first block's, the second's
    (less left context when the first is shorter than LEFT), and every later one's."""
    return tuple(sorted({first + right, min(left, first) + block + right, left + block + right}))


class StreamingSpeechToSpeech(Rung):
    """R4, fast, and streamed: each block of the reply handed out as soon as it is decoded."""

    rung_id = "r4-gpu-stream"
    description = ("R4 streamed: Qwen2.5-Omni-7B's Talker compiled, its codes decoded a block at a time by the "
                   "compiled DiT and handed out as they are made; bf16, CUDA")
    streams, out_sr = True, OUT_SR          # what the demo reads: its pieces can be played as they come (demo.stream_turn)

    def __init__(self, model=None, first: int = FIRST_CODES, block: int = BLOCK_CODES,
                 start_buffer_s: float = START_BUFFER_S):
        self.model = model or FastOmni()
        self.first, self.block, self.start_buffer_s = first, block, start_buffer_s
        self.last_codes: list[int] = []
        self.last_windows: list[tuple[int, int]] = []

    def _context(self) -> tuple[int, int]:
        """LEFT and RIGHT in codes, from the DiT's own configuration: two blocks back, one ahead."""
        dit = self.model.model.token2wav.code2wav_dit_model
        per_block = dit.block_size // dit.repeats
        cfg = dit.config
        return len(cfg.look_backward_layers) * per_block, len(cfg.look_ahead_layers) * per_block

    def _decode_window(self, codes: list[int]) -> np.ndarray:
        import torch
        m = self.model.model
        sp = m.speaker_map[self.model.speaker]
        dev = torch.device(self.model.device)
        ids = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=ids), torch.inference_mode():
            x = torch.tensor([codes], device=dev)
            mel = m.token2wav.code2wav_dit_model.sample(sp["cond"].to(dev).float().expand(1, -1),
                                                        sp["ref_mel"].to(dev).float().expand(1, -1, -1), x)
            wav = m.token2wav.code2wav_bigvgan_model(mel)
        return wav.reshape(-1).float().cpu().numpy()

    def _load(self) -> None:
        self.model.load()
        check_blocks(self.model.model, ((self.first, self.block),))
        left, right = self._context()
        lengths = windows_for(self.first, self.block, left, right)
        self.model.compile_dit(lengths)
        self.model.speak(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)      # the Talker's graph, as R4's throwaway turn
        for n in lengths:                                                  # each DiT window: compile, then capture
            for _ in range(2):
                self._decode_window([0] * n)

    def _respond(self, audio, sr, sink):
        if sr != ASR_SR:
            raise ValueError(f"R4 expects {ASR_SR} Hz input, got {sr}: resample first (probes.load_wav does)")
        left, right = self._context()
        dec = BlockDecoder(self._decode_window, self.first, self.block, left, right)
        out = StartBuffer(self.emit, self.start_buffer_s)
        streamer = CodeStreamer(lambda code: out.add(dec.push(code)))
        t2w = self.model.model.token2wav
        with stage("s2s", sink):
            t2w.forward = lambda *a, **k: _empty()               # the whole-reply decode is the stream's now
            try:
                text, _ = self.model.speak(audio, sr, talker_streamer=streamer)
            finally:
                del t2w.forward
            out.add(dec.finish())
            out.flush()
        self.last_codes, self.last_windows = streamer.codes, dec.windows
        return Reply(None, OUT_SR, None, clean_for_speech(text))


def _empty():
    import torch
    return torch.zeros(0)


def pair(model=None):
    """The demo's L40S: r4-gpu-fast and r4-gpu-stream on one FastOmni, one set of R4's weights (two sets,
    22.5 GB each with their activations, would not fit the card's 48). Returns get(rung_id), as
    demo.Live takes it; each rung is made when it is asked for, both on the same model."""
    fast = model or FastOmni()
    made = {"r4-gpu-fast": lambda: FastSpeechToSpeech(fast), "r4-gpu-stream": lambda: StreamingSpeechToSpeech(fast)}

    def get(rung_id: str):
        if rung_id not in made:
            raise KeyError(f"{rung_id!r} is not one of the pair: {', '.join(made)}")
        return made[rung_id]()

    return get
