"""
M34: R4 streamed on a right-sized cache (s2slab.r4sized) and its development run (s2slab.r4sizeddev), on
the tiny Talker of tests/test_talkerloop.py and the fake DiT of tests/test_r4stream.py, on the CPU: the
cache each turn takes, a reply cut by a cache too small and saying so, the same codes and audio as today's
stream, and the start rule. Skipped without transformers.
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_r4stream import FakeASR, Token2Wav  # noqa: E402
from test_talkerloop import NEW, PROMPT, SAMPLING, fast_route, generate_codes, static_cache, tiny_talker, turn  # noqa: E402

from s2slab import r4sized, r4sizeddev  # noqa: E402
from s2slab.r4stream import EOS, FastOmni, StreamingSpeechToSpeech  # noqa: E402

same = lambda fn, **kw: fn                                     # noqa: E731  (no compiling on the CPU)


class TinyBase:
    """OmniS2S's surface around the tiny Talker: speak() hands talker.generate a made-up turn (one per clip,
    by its first sample) with R4's extra arguments, and token2wav the codes, as R4's call does."""

    def __init__(self):
        self.model, self.speaker, self.device, self.seed = None, "Chelsie", "cpu", 0

    def load(self):
        self.model = SimpleNamespace(talker=tiny_talker(), token2wav=Token2Wav(),
                                     speaker_map={"Chelsie": {"cond": torch.zeros(4), "ref_mel": torch.zeros(3, 4)}})

    def speak(self, audio, sr, **extra):
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            out = self.model.talker.generate(**turn(self.model.talker, int(audio[0]) + 1), streamer=extra.get("talker_streamer"),
                                             past_key_values=extra.get("talker_past_key_values"), disable_compile=True,
                                             max_new_tokens=min(NEW, extra.get("talker_max_new_tokens", NEW)), **SAMPLING)
            codes = [c for c in out[0, PROMPT:].tolist() if c not in EOS]
            wav = self.model.token2wav(torch.tensor([codes]), conditioning=torch.zeros(1, 4), reference_mel=torch.zeros(1, 3, 4))
        return "Sure, here it is.", np.asarray(wav.reshape(-1), dtype=np.float32)


def sized_codes(sized, inputs, seed=7):
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = sized.generate(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                             max_new_tokens=NEW, past_key_values=None, disable_compile=True, **SAMPLING)
    return out[0, PROMPT:].tolist()


def test_a_turn_takes_the_smallest_cache_that_leaves_it_room_and_writes_generates_codes():
    from transformers import StaticCache
    talker = tiny_talker()
    fast_route(talker)
    inputs = turn(talker)
    want, _ = generate_codes(talker, inputs, static_cache(talker))
    caches = {n: StaticCache(config=talker.config, max_cache_len=n) for n in (PROMPT + 90, PROMPT + 200)}
    sized = r4sized.SizedTalker(talker, caches, room=85)
    assert sized.places_for(PROMPT) == PROMPT + 90 and sized.places_for(PROMPT + 10) == PROMPT + 200
    assert sized.places_for(10_000) == PROMPT + 200                      # none leaves room: the largest
    assert sized_codes(sized, inputs) == want
    assert sized.last == {"places": PROMPT + 90, "prompt": PROMPT, "codes": NEW, "cut": False}
    sized.room = 95                                                      # now the small one is too small
    assert sized_codes(sized, inputs) == want and sized.last["places"] == PROMPT + 200


def test_a_cache_too_small_cuts_the_reply_and_says_so():
    from transformers import StaticCache
    talker = tiny_talker()
    fast_route(talker)
    sized = r4sized.SizedTalker(talker, {PROMPT + 30: StaticCache(config=talker.config, max_cache_len=PROMPT + 30)})
    codes = sized_codes(sized, turn(talker))
    assert len(codes) == 30 and sized.last["cut"] and sized.last["places"] == PROMPT + 30


def test_the_sized_stream_writes_todays_streams_codes_and_audio_on_the_small_cache():
    fast = FastOmni(TinyBase(), compile_fn=same)
    stream = StreamingSpeechToSpeech(fast, start_buffer_s=0.0)
    sized = r4sized.SizedStreamingSpeechToSpeech(fast, start_buffer_s=0.0)
    stream.load()
    sized.load()
    assert sized.sized.loops[2048]._g is not None and sized.sized.loops[6144]._g is None   # only 2048's step at load (M35)
    assert list(sized.load_phases) == ["model", "throwaway turn", "DiT windows", "2048-place cache prepared"]
    assert "generate" not in vars(fast.model.talker)                            # installed only while it answers
    clip = np.full(16_000, 2.0, dtype=np.float32)
    a, b = stream.respond(clip, 16_000), sized.respond(clip, 16_000)
    assert sized.last_codes == stream.last_codes and len(sized.last_codes) > 24
    assert np.array_equal(a.audio, b.audio) and len(b.chunks) == len(a.chunks) > 1
    assert (sized.last_turn["places"], sized.last_turn["prompt"], sized.last_turn["cut"]) == (2048, PROMPT, False)
    assert stream.respond(clip, 16_000).audio.tolist() == a.audio.tolist()      # today's stream untouched after


def test_the_development_run_compares_both_streams_and_applies_the_start_rule():
    clips = {f"p{i}": np.full(16_000, float(i), dtype=np.float32) for i in range(3)}
    blob = r4sizeddev.run(clips, 16_000, asr=FakeASR(), base=TinyBase(), make_fast=lambda b: FastOmni(b, compile_fn=same),
                          listen=1)
    assert list(blob["rungs"]) == ["r4-gpu-fast", "r4-gpu-stream", "r4-gpu-stream-sized"]
    for name in r4sizeddev.STREAMS:
        assert all(t["codes_as_fast"] for t in blob["rungs"][name]["turns"].values())
        assert set(blob["rungs"][name]["buffers"]) == {"0", "0.25", "0.5", "1"}
    assert all(t["places"] == 2048 and not t["cut"] for t in blob["rungs"]["r4-gpu-stream-sized"]["turns"].values())
    chosen = r4sizeddev.choose(blob)
    assert set(chosen) == set(r4sizeddev.STREAMS)
    text = r4sizeddev.format_sized_dev(blob)
    assert "2048 places on 3 turns; replies cut: 0" in text and "The start rule" in text


def test_the_start_rule_takes_the_lowest_first_sound_with_no_stall_and_the_smaller_buffer_on_a_tie():
    def turns(fs, stall):
        return {"a": {"first_sound_ms": fs, "stall_ms": stall, "total_ms": 1, "reply_s": 1, "words": 1, "word_errors": 0}}

    def stream(by_buffer):
        return {"turns": turns(1000, 0), "buffers": {b: {"a": {"first_sound_ms": fs, "stall_ms": st}} for b, (fs, st) in by_buffer.items()}}

    blob = {"clips": ["a"], "rungs": {
        "r4-gpu-fast": {"turns": turns(5000, 0)},
        "r4-gpu-stream": stream({"0": (1500, 300), "0.25": (2600, 0), "0.5": (2900, 0), "1": (3300, 0)}),
        "r4-gpu-stream-sized": stream({"0": (1200, 100), "0.25": (2000, 0), "0.5": (2000, 0), "1": (2500, 0)})}}
    c = r4sizeddev.choose(blob)
    assert c["r4-gpu-stream"]["buffer_s"] == 0.25 and c["r4-gpu-stream"]["first_sound_p50"] == 2600
    assert c["r4-gpu-stream-sized"]["buffer_s"] == 0.25                     # a tie: the smaller buffer
    blob["rungs"]["r4-gpu-stream"]["buffers"] = {b: {"a": {"first_sound_ms": 1, "stall_ms": 9}} for b in ("0", "1")}
    assert r4sizeddev.choose(blob)["r4-gpu-stream"] is None


def test_a_cache_not_prepared_is_written_on_its_first_turn_and_captured_on_the_next():
    from transformers import StaticCache
    talker = tiny_talker()
    fast_route(talker)
    inputs = turn(talker)
    want, _ = generate_codes(talker, inputs, static_cache(talker))
    big = StaticCache(config=talker.config, max_cache_len=PROMPT + 200)
    sized = r4sized.SizedTalker(talker, {PROMPT + 90: StaticCache(config=talker.config, max_cache_len=PROMPT + 90), PROMPT + 200: big},
                                room=95)                                 # the turn takes the big one, never prepared
    assert not big.layers[0].is_initialized
    assert sized_codes(sized, inputs) == want and sized.loops[PROMPT + 200]._g is None      # eager: the cache written
    assert big.layers[0].is_initialized
    assert sized_codes(sized, inputs) == want and sized.loops[PROMPT + 200]._g is not None  # now its step is made
    assert sized.loops[PROMPT + 90]._g is None                          # the small one never asked for
