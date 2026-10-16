"""
Tests for s2slab.r4stream and s2slab.r4streamdev -- the fast and streamed R4 -- with a fake Omni model
built like the real one, on the CPU. torch.compile is replaced by a recorder; nothing compiles.

The fake DiT gives every code a constant level (its code number / 1000) for its 480 samples, the
same in any window: so a correct block decoder, crossfades included, reproduces the whole reply
exactly, and any sample out of place shows.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s2slab.r4 import OUT_SR
from s2slab.r4stream import (EOS, MAX_TURN_S, SAMPLES_PER_CODE, BlockDecoder, CodeStreamer, FastOmni,
                             FastSpeechToSpeech, StartBuffer, StreamingSpeechToSpeech, windows_for)
from s2slab.r4streamdev import buffered, format_dev, run, summary
from s2slab.rungs import RUNGS, get

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPC = SAMPLES_PER_CODE


def level_audio(codes):
    """What a perfect decoder returns for these codes: each code's level, 480 samples each."""
    return np.repeat(np.asarray(codes, dtype=np.float32) / 1000, SPC)


class Recorder:
    def __init__(self):
        self.made = []

    def __call__(self, fn, **kw):
        self.made.append(kw)
        return fn


# ------------------------------------------------------------------ the fake model, like Qwen2.5-Omni


class DiT:
    block_size, repeats = 24, 2                                  # 12 codes a block, as the real one
    config = SimpleNamespace(look_backward_layers=(0, 20), look_ahead_layers=(10,))

    def __init__(self):
        self.lengths = []

    def forward(self, hidden_states=None, **kw):
        self.lengths.append(hidden_states.shape[1])
        return hidden_states

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def sample(self, cond, ref, code):
        torch.randn(1)                                           # draws noise, as the real one does
        self(hidden_states=torch.zeros(1, code.shape[-1] * self.repeats, 80))
        return code.float() / 1000                               # the "mel": one level a code


class Vocoder:
    def forward(self, mel):
        return mel.repeat_interleave(SPC, dim=-1).reshape(-1)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Token2Wav:
    def __init__(self):
        self.code2wav_dit_model, self.code2wav_bigvgan_model, self.calls = DiT(), Vocoder(), 0

    def forward(self, code, conditioning=None, reference_mel=None, **kw):
        self.calls += 1
        return self.code2wav_bigvgan_model(self.code2wav_dit_model.sample(conditioning, reference_mel, code))

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Layers:
    def forward(self, inputs_embeds=None, attention_mask=None, **kw):
        return inputs_embeds

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Talker:
    def __init__(self):
        self.model, self.config = Layers(), "talker-config"

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, **kwargs):
        return {"attention_mask": kwargs.get("attention_mask")}


class FakeBase:
    """OmniS2S's surface: load, speak (R4's call, with extra arguments for generate), model, speaker, device.
    speak samples its codes from the global RNG, as the Talker does, so a decoder that moved the RNG
    would change them."""

    def __init__(self, n_codes=150):
        self.n_codes, self.model, self.speaker, self.device, self.asked, self.loads = n_codes, None, "Chelsie", "cpu", [], 0
        self.seed = 0

    def load(self):
        self.loads += 1
        self.model = SimpleNamespace(talker=Talker(), token2wav=Token2Wav(),
                                     speaker_map={"Chelsie": {"cond": torch.zeros(4), "ref_mel": torch.zeros(3, 4)}})

    def speak(self, audio, sr, **extra):
        self.asked.append(extra)
        torch.manual_seed(self.seed)
        streamer = extra.get("talker_streamer")
        if streamer is not None:
            streamer.put(torch.zeros(1, 7))                      # the prompt first
        codes = []
        for _ in range(self.n_codes):                            # one code a step, drawn as the Talker samples
            codes.append(int(torch.randint(1, 8000, (1,))))
            if streamer is not None:
                streamer.put(torch.tensor([codes[-1]]))          # the stream may decode a block here
        if streamer is not None:
            streamer.put(torch.tensor([EOS[0]]))
            streamer.end()
        wav = self.model.token2wav(torch.tensor([codes]), conditioning=torch.zeros(1, 4), reference_mel=torch.zeros(1, 3, 4))
        self.last_codes = codes
        return "**Sure**, here it is.", np.asarray(wav.reshape(-1), dtype=np.float32)


class FakeCache:
    def __init__(self, config, max_len):
        self.config, self.max_len, self.resets = config, max_len, 0

    def reset(self):
        self.resets += 1


def fast_model(n_codes=150):
    return FastOmni(FakeBase(n_codes), compile_fn=Recorder(), cache_factory=FakeCache)


# ------------------------------------------------------------------ the pieces


def test_the_streamer_skips_the_prompt_and_stops_at_an_end_code():
    got = []
    s = CodeStreamer(got.append)
    s.put(torch.zeros(1, 9))
    for c in (5, 6, EOS[1], 7):
        s.put(torch.tensor([c]))
    assert got == [5, 6] == s.codes and s.done


@pytest.mark.parametrize("n", [5, 36, 37, 150, 300])
def test_the_block_decoder_reproduces_the_whole_reply_sample_for_sample(n):
    codes = list(range(1, n + 1))
    seen = []
    dec = BlockDecoder(lambda w: (seen.append(len(w)), level_audio(w))[1], first=24, block=96, left=24, right=12)
    pieces = []
    for c in codes:
        pieces += dec.push(c)
    pieces += dec.finish()
    out = np.concatenate(pieces)
    assert len(out) == n * SPC and np.allclose(out, level_audio(codes))
    for ws, we in dec.windows:                                   # every window starts on a DiT block
        assert ws % 12 == 0 and we <= n


def test_a_block_waits_for_its_right_context_and_the_first_block_comes_first():
    dec = BlockDecoder(level_audio, first=24, block=96, left=24, right=12)
    ready = [len(dec.push(c)) for c in range(1, 37)]
    assert ready[:35] == [0] * 35 and ready[35] == 1             # the 36th code (24 + 12) frees the first block
    assert dec.windows == [(0, 36)]
    for c in range(37, 37 + 96):
        dec.push(c)
    assert dec.windows[1] == (0, 132)                            # [24 - 24, 120 + 12): two blocks back, one ahead


def test_the_crossfade_mixes_the_overlap_of_two_different_windows():
    calls = []

    def noisy(w):                                                # each window adds its own offset, as fresh noise would
        calls.append(len(w))
        return level_audio(w) + 0.1 * len(calls)

    dec = BlockDecoder(noisy, first=24, block=24, left=24, right=12)
    pieces = []
    for c in range(1, 61):
        pieces += dec.push(c)
    pieces += dec.finish()
    out = np.concatenate(pieces)
    edge = 24 * SPC                                              # where block 1 hands over to block 2
    assert len(out) == 60 * SPC
    assert out[edge - SPC] == pytest.approx(0.024 + 0.1)        # the overlap (code 24) starts as block 1 ...
    assert out[edge - 1] == pytest.approx(0.024 + 0.2, abs=1e-3)  # ... and ends as block 2


def test_the_start_buffer_holds_back_until_enough_audio_exists():
    sent = []
    b = StartBuffer(sent.append, seconds=0.05, sr=1000)          # 50 samples
    b.add([np.ones(20), np.ones(20)])
    assert sent == []
    b.add([np.ones(20)])
    assert [len(x) for x in sent] == [60]
    b.add([np.ones(5)])
    b.flush()
    assert [len(x) for x in sent] == [60, 5]
    sent2 = []
    StartBuffer(sent2.append, 0.0).add([np.ones(3), np.ones(4)])
    assert [len(x) for x in sent2] == [3, 4]


def test_the_compiled_windows_are_the_steady_ones():
    assert windows_for(24, 96, 24, 12) == (36, 132)
    assert windows_for(48, 96, 24, 12) == (60, 132)
    assert windows_for(12, 96, 24, 12) == (24, 120, 132)        # a first block shorter than the left context


# ------------------------------------------------------------------ the rungs


def test_both_rungs_are_registered_and_build_without_any_model_library():
    assert RUNGS["r4-gpu-fast"] == "s2slab.r4stream:FastSpeechToSpeech"
    assert RUNGS["r4-gpu-stream"] == "s2slab.r4stream:StreamingSpeechToSpeech"
    assert get("r4-gpu-fast").rung_id == "r4-gpu-fast" and get("r4-gpu-stream").rung_id == "r4-gpu-stream"


def test_the_fast_model_installs_the_talker_fixes_once_and_resets_its_cache_each_turn():
    fast = fast_model()
    fast.load()
    fast.load()
    talker = fast.model.talker
    assert "prepare_inputs_for_generation" in vars(talker) and "forward" in vars(talker.model)
    assert fast.compile_fn.made == [{"mode": "reduce-overhead", "dynamic": False}]      # once, not twice
    fast.speak(np.zeros(16000, dtype=np.float32), 16000)
    fast.speak(np.zeros(16000, dtype=np.float32), 16000)
    assert fast.cache.resets == 2 and fast.cache.max_len == 6144
    assert fast.base.asked[-1] == {"talker_past_key_values": fast.cache, "talker_disable_compile": True,
                                   "talker_max_new_tokens": 4096}
    with pytest.raises(ValueError, match="too short"):
        fast.speak(np.zeros(int((MAX_TURN_S + 1) * 16000), dtype=np.float32), 16000)


def test_the_fast_rung_answers_whole_as_r4_does():
    rung = FastSpeechToSpeech(fast_model())
    turn = rung.respond(np.zeros(16000, dtype=np.float32), 16000)
    assert turn.reply_text == "Sure, here it is." and len(turn.chunks) == 1 and turn.reply_s == 150 * SPC / OUT_SR
    assert list(rung.load_phases) == ["model", "throwaway turn"]                 # its load timed by phase (M36)


def test_the_stream_hands_out_blocks_as_the_talker_writes_and_its_codes_are_the_fast_rungs():
    fast = fast_model(n_codes=300)
    whole = FastSpeechToSpeech(fast).respond(np.zeros(16000, dtype=np.float32), 16000)
    rung = StreamingSpeechToSpeech(fast, first=24, block=96, start_buffer_s=0.0)
    rung.load()
    t2w = fast.model.token2wav
    calls = t2w.calls
    turn = rung.respond(np.zeros(16000, dtype=np.float32), 16000)
    codes = fast.base.last_codes
    assert rung.last_codes == codes                              # the DiT's noise did not move the Talker's
    assert np.allclose(turn.audio, level_audio(codes)) and np.allclose(turn.audio, whole.audio)
    # blocks [0,24) [24,120) [120,216) while the Talker writes, [216,300) at the end; the first holds back 20 ms
    assert len(turn.chunks) == 4 and turn.chunks[0][1] == 24 * SPC - SPC
    assert turn.first_audio_ms <= turn.total_ms and turn.reply_text == "Sure, here it is."
    assert t2w.calls == calls and "forward" not in vars(t2w)    # no whole-reply decode in a streamed turn, and restored
    assert fast.dit_lengths == {36, 132}


def test_the_stream_with_a_start_buffer_hands_out_fewer_bigger_first_pieces():
    fast = fast_model(n_codes=300)
    turn = StreamingSpeechToSpeech(fast, first=24, block=96, start_buffer_s=2.5).respond(np.zeros(16000, dtype=np.float32), 16000)
    assert turn.chunks[0][1] >= int(2.5 * OUT_SR) and len(turn.audio) == 300 * SPC


# ------------------------------------------------------------------ the development run


def test_buffered_gives_what_the_start_buffer_would_have():
    chunks = [(1000.0, 24000), (3000.0, 48000), (3500.0, 48000)]           # 1 s, then 2 s, then 2 s of audio
    b0 = buffered(chunks, 24000, 0.0, 0.0)
    assert b0 == {"first_audio_ms": 1000.0, "first_sound_ms": 1000.0, "stall_ms": 1000.0}   # 1 s of audio, 2 s wait
    b2 = buffered(chunks, 24000, 2.0, 0.0)
    assert b2["first_audio_ms"] == 3000.0 and b2["stall_ms"] == 0.0
    b9 = buffered(chunks, 24000, 9.0, 100.0)                                 # a buffer longer than the reply
    assert b9["first_audio_ms"] == 3500.0 and b9["first_sound_ms"] == 3600.0


class FakeASR:
    def load(self):
        pass

    def transcribe(self, audio):
        return "sure here it is"


def test_the_development_run_measures_r4_the_fast_rung_and_each_stream_setting():
    base = FakeBase(n_codes=200)
    clips = {f"p{i}": np.zeros(16000, dtype=np.float32) for i in range(4)}
    blob = run(clips, 16000, asr=FakeASR(), base=base, settings=((24, 96), (48, 96)),
               make_fast=lambda b: FastOmni(b, compile_fn=Recorder(), cache_factory=FakeCache))
    assert list(blob["rungs"]) == ["r4-gpu", "r4-gpu-fast", "r4-gpu-stream/24+96", "r4-gpu-stream/48+96"]
    for name in ("r4-gpu-stream/24+96", "r4-gpu-stream/48+96"):
        assert all(t["codes_as_fast"] for t in blob["rungs"][name]["turns"].values())
        assert set(blob["rungs"][name]["buffers"]) == {"0", "0.5", "1", "1.5", "2", "3"}
    assert all(t["word_errors"] == 0 for r in blob["rungs"].values() for t in r["turns"].values())
    assert len(blob["listen"]) == 4 * 3 and all(k.endswith(".wav") for k in blob["listen"])
    s = summary(blob)
    assert s["r4-gpu"]["stalled"] == 0 and s["r4-gpu-stream/24+96"]["codes_as_fast"] == 4 and s["r4-gpu"]["wer"] == 0.0
    text = format_dev(blob)
    assert "| r4-gpu-stream/48+96 |" in text and "with a start buffer" in text and "4 of 4" in text


def test_a_setting_off_the_dits_blocks_stops_the_development_run_before_any_turn():
    base = FakeBase()
    with pytest.raises(ValueError, match="do not align"):
        run({"p0": np.zeros(16000, dtype=np.float32)}, 16000, asr=FakeASR(), base=base, settings=((24, 90),),
            make_fast=lambda b: FastOmni(b, compile_fn=Recorder(), cache_factory=FakeCache))
    assert len(base.asked) == 1                                  # R4's load and throwaway turn, no clip


def test_the_l40s_app_has_the_streamdev_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.stream_dev, "remote") and hasattr(mod.streamdev, "info")
    assert mod.STREAM_DEV_PATH == ROOT / "runs" / "r4-gpu" / "stream-dev-l40s.json"


def test_the_l40s_app_has_the_loopspeed_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.loop_speed, "remote") and hasattr(mod.loopspeed, "info")
    assert mod.LOOP_SPEED_PATH == ROOT / "runs" / "r4-gpu" / "loop-speed-l40s.json"


def test_the_l40s_app_has_the_stepprofile_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.step_profile, "remote") and hasattr(mod.stepprofile, "info")
    assert mod.STEP_PROFILE_PATH == ROOT / "runs" / "r4-gpu" / "step-profile-l40s.json"


def test_the_l40s_app_has_the_sizeddev_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.sized_dev, "remote") and hasattr(mod.sizeddev, "info")
    assert mod.SIZED_DEV_PATH == ROOT / "runs" / "r4-gpu" / "sized-dev-l40s.json"


def test_the_l40s_app_has_the_timeddev_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.timed_dev, "remote") and hasattr(mod.timeddev, "info")
    assert mod.TIMED_DEV_PATH == ROOT / "runs" / "r4-gpu" / "timed-dev-l40s.json"


def test_the_l40s_app_has_the_pr016_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.pr016_run, "remote") and hasattr(mod.pr016, "info")


def test_the_l40s_app_has_the_pr017_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.pr017_run, "remote") and hasattr(mod.pr017, "info")
