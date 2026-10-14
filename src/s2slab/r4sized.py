"""
M34: R4 streamed on a cache sized to the turn.

M33 (runs/r4-gpu/step-profile-l40s.json) found half of a Talker step to be attention over the static
cache's empty places: our captured loop wrote a code in 10.26 ms on today's 6144-place cache and in 6.62 ms
on a 2048-place one, the same codes on 10 of 10 probe clips. So:

    SizedTalker                     talker.generate on the smallest cache that leaves the turn ROOM_CODES of
                                    room after its prompt (of SIZES: 2048 places, else today's 6144), through
                                    our loop with its step captured (s2slab.talkerloop), one loop and one
                                    captured step a cache. A reply that reaches its cache's end is cut, and
                                    the turn says so (last["cut"]): counted in every run, never hidden.
    r4-gpu-stream-sized             R4 streamed (r4stream.StreamingSpeechToSpeech: the same blocks, DiT and
                                    start buffer) with its codes from a SizedTalker. Its load writes the
                                    caches of PREPARED once (our loop, eager) and captures their steps, so no
                                    turn pays it; any other cache's step is captured on the first turn that
                                    needs it (M35: the demo's turns, 15 s at most, never need 6144 places).

ROOM_CODES is 1,024 codes, about 20 s of speech: twice the longest reply on the probe clips (503 codes,
M33). A prompt of 1,024 places takes about 36 s of audio; a longer turn takes today's cache.

    modal run modal_app/r4_bench.py::sizeddev       both streams on the 30 probe clips (development)
"""

from __future__ import annotations

import numpy as np

from .r0 import ASR_SR
from .r4stream import MAX_CACHE, StreamingSpeechToSpeech

SIZES = (2048, MAX_CACHE)           # a cache's places: the small one first
PREPARED = (2048,)                  # the caches whose steps are captured at load (M35); the rest on first use
ROOM_CODES = 1024                   # the codes a turn's cache must leave room for after its prompt


class SizedTalker:
    """talker.generate on a cache sized to each turn (see the module's docstring). `caches` maps places
    to a static cache of that many; `layers` runs the decoder inside each captured step."""

    def __init__(self, talker, caches: dict, layers=None, room: int = ROOM_CODES, graph: bool = True):
        from .talkerloop import TalkerLoop
        self.talker, self.room = talker, room
        self.loops = {n: TalkerLoop(talker, caches[n], graph=graph, layers=layers) for n in sorted(caches)}
        self.last: dict | None = None
        self._force = None                               # (places, "eager" | "loop"): prepare() only

    def places_for(self, prompt: int) -> int:
        """The smallest cache that leaves `room` codes after the prompt; the largest when none does."""
        return next((n for n in self.loops if n - prompt >= self.room), max(self.loops))

    def generate(self, *args, **kw):
        from .talkerloop import TalkerLoop
        prompt = int(kw["inputs_embeds"].shape[1])
        n = self._force[0] if self._force else self.places_for(prompt)
        loop = self.loops[n]
        unwritten = loop.graph and loop._g is None and not getattr(loop.cache.layers[0], "is_initialized", True)
        if (self._force and self._force[1] == "eager") or unwritten:   # writes the cache once: its tensors made;
            loop = TalkerLoop(self.talker, loop.cache)                 # its step captured on the next turn
        room = n - prompt
        if room < 1:
            raise ValueError(f"a prompt of {prompt} places does not fit a cache of {n}")
        asked = int(kw.get("max_new_tokens", 4096))
        loop.cache.reset()
        kw.update(past_key_values=loop.cache, max_new_tokens=min(asked, room))
        out = loop.generate(*args, **kw)
        self.last = {"places": n, "prompt": prompt, "codes": loop.last_codes,
                     "cut": loop.last_codes >= room and room < asked}
        return out

    def install(self):
        """talker.generate becomes this; returns the undo."""
        had = "generate" in vars(self.talker)
        inner = self.talker.generate
        self.talker.generate = self.generate
        return (lambda: setattr(self.talker, "generate", inner)) if had else (lambda: delattr(self.talker, "generate"))

    def prepare(self, speak, sizes=None) -> None:
        """Each cache of `sizes` (all, if None) written once (our loop, eager), then its step captured, by
        throwaway turns through `speak` (R4's call): so that no real turn pays either."""
        undo = self.install()
        try:
            for n in (self.loops if sizes is None else sizes):
                for mode in ("eager", "loop"):
                    self._force = (n, mode)
                    speak()
        finally:
            self._force = None
            undo()


class SizedStreamingSpeechToSpeech(StreamingSpeechToSpeech):
    """R4 streamed, its codes written by our captured loop on a cache sized to the turn."""

    rung_id = "r4-gpu-stream-sized"
    description = ("R4 streamed on a cache sized to the turn: Qwen2.5-Omni-7B's Talker through our own decode loop "
                   "with its step captured as a CUDA graph, on 2048 places when the turn leaves room, else 6144; "
                   "the reply decoded a block at a time by the compiled DiT; bf16, CUDA")

    def __init__(self, model=None, sizes=SIZES, room: int = ROOM_CODES, prepared=PREPARED, **stream):
        super().__init__(model, **stream)
        self.sizes, self.room, self.prepared = tuple(sizes), room, tuple(prepared)
        self.sized: SizedTalker | None = None
        self.last_turn: dict | None = None

    def _load(self) -> None:
        super()._load()                                  # the model, the DiT's windows, R4's throwaway turn
        from .r4speed import _torch_compile
        talker = self.model.model.talker
        caches = {n: self.model.cache if n == MAX_CACHE else self.model.make_cache(n) for n in self.sizes}
        m = talker.model
        layers = (self.model.compile_fn or _torch_compile)(lambda **kw: type(m).forward(m, **kw), mode="default",
                                                           dynamic=False)
        self.sized = SizedTalker(talker, caches, layers=layers, room=self.room)
        speak = lambda: self.model.speak(np.zeros(ASR_SR, dtype=np.float32), ASR_SR)        # noqa: E731
        for n in self.prepared:
            self._timed(f"{n}-place cache prepared", lambda n=n: self.sized.prepare(speak, sizes=(n,)))

    def _respond(self, audio, sr, sink):
        undo = self.sized.install()
        try:
            reply = super()._respond(audio, sr, sink)
        finally:
            undo()
        self.last_turn = dict(self.sized.last or {})
        return reply
