"""
Tests for s2slab.r4speed -- the Talker and the DiT, eager and compiled, with a fake Omni model built
like the real one (torch tensors on the CPU). torch.compile is replaced by a recorder, so nothing
compiles; what is checked is the wiring: what runs compiled, what is compared, what is recorded.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s2slab.r4 import OmniS2S
from s2slab.r4speed import (EVERY, SPEED, TALKER_VARIANTS, WARMUPS, compile_dit, compile_layers, format_speed, run,
                            summary)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPC = 480


class Recorder:
    """Stands in for torch.compile: returns the function unchanged and counts the calls through it."""

    def __init__(self):
        self.made, self.calls = [], 0

    def __call__(self, fn, **kw):
        self.made.append(kw)

        def counted(*a, **k):
            self.calls += 1
            return fn(*a, **k)
        return counted


class Layers:                                    # talker.model: the decoder layers
    def __init__(self):
        self.lengths = []

    def forward(self, inputs_embeds=None, **kw):
        self.lengths.append(inputs_embeds.shape[1])
        return SimpleNamespace(last_hidden_state=inputs_embeds)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Talker:
    def __init__(self, omni):
        self.omni, self.model = omni, Layers()

    def generate(self, **kw):
        extra = self.omni.extra
        key = extra.get("talker_cache_implementation", "eager") + str(extra.get("talker_disable_compile", ""))
        if self.omni.fail == key and len(self.omni.asked) > 1:              # after R4's throwaway turn
            raise RuntimeError("cannot compile this")
        self.model(inputs_embeds=torch.zeros(1, 5, 8))                       # the prefill
        for _ in range(self.omni.n_codes):
            self.model(inputs_embeds=torch.zeros(1, 1, 8))                   # one code a step
        codes = list(range(self.omni.n_codes))
        if extra.get("talker_disable_compile"):
            codes[5] += 1                                                    # drifts at code 5
        return torch.tensor([codes])


class DiT:
    def forward(self, x):
        return x * 0.5

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def sample(self, cond, ref, code):
        x = torch.randn(1, 2 * code.shape[-1])
        for _ in range(4):                                                   # the solver keeps several outputs
            x = x + self(x)
        return x


class Vocoder:
    def forward(self, mel):
        return mel.repeat_interleave(SPC // 2, dim=-1).reshape(-1)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Token2Wav:
    def __init__(self):
        self.code2wav_dit_model, self.code2wav_bigvgan_model, self.lengths = DiT(), Vocoder(), []

    def forward(self, code, conditioning=None, reference_mel=None, **kw):
        assert conditioning.shape == (1, 4) and reference_mel.shape == (1, 3, 4)
        self.lengths.append(code.shape[-1])
        return self.code2wav_bigvgan_model(self.code2wav_dit_model.sample(conditioning, reference_mel, code))

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class FakeOmni:
    def __init__(self, n_codes=120, fail=None):
        self.n_codes, self.fail, self.model, self.extra, self.asked = n_codes, fail, None, {}, []
        self.speaker, self.device = "Chelsie", "cpu"

    def load(self):
        thinker = SimpleNamespace(generate=lambda input_ids, **k: SimpleNamespace(sequences=torch.zeros(1, input_ids.shape[1] + 9)))
        self.model = SimpleNamespace(thinker=thinker, talker=Talker(self), token2wav=Token2Wav(),
                                     speaker_map={"Chelsie": {"cond": torch.zeros(4), "ref_mel": torch.zeros(3, 4)}})

    def speak(self, audio, sr, **extra):
        self.extra = extra
        self.asked.append(dict(extra))
        torch.manual_seed(0)
        self.model.thinker.generate(input_ids=torch.zeros(1, 30))
        codes = self.model.talker.generate(**extra)
        wav = self.model.token2wav(codes, conditioning=torch.zeros(1, 4), reference_mel=torch.zeros(1, 3, 4))
        return "Sure, I can help.", wav


def clips(n=6):
    return {f"c{i}": np.zeros(16000, dtype=np.float32) for i in range(n)}


def test_only_the_talkers_one_code_steps_run_compiled_and_undo_restores_it():
    talker = Talker(SimpleNamespace(extra={}, fail=None, n_codes=3))
    rec = Recorder()
    undo = compile_layers(talker, rec)
    assert rec.made == [{"mode": "reduce-overhead", "dynamic": False}]
    talker.model(inputs_embeds=torch.zeros(1, 5, 8))
    talker.model(inputs_embeds=torch.zeros(1, 1, 8))
    talker.model(inputs_embeds=torch.zeros(1, 1, 8))
    assert talker.model.lengths == [5, 1, 1] and rec.calls == 2          # the prefill ran eager
    undo()
    talker.model(inputs_embeds=torch.zeros(1, 1, 8))
    assert rec.calls == 2 and "forward" not in vars(talker.model)


def test_the_compiled_dit_copies_each_output_and_undo_restores_it():
    dit, rec = DiT(), Recorder()
    undo = compile_dit(dit, "reduce-overhead", rec)
    x = torch.ones(1, 4)
    y = dit(x)
    assert rec.made == [{"mode": "reduce-overhead", "dynamic": False}] and rec.calls == 1
    assert torch.equal(y, x * 0.5)
    undo()
    assert "forward" not in vars(dit)


def test_the_probe_compares_every_variant_with_eager_on_the_same_clips():
    omni, rec = FakeOmni(), Recorder()
    blob = run(omni, clips(6), 16000, compile_fn=rec)
    assert blob["clips"] == ["c0", "c3"] and EVERY == 3
    t = blob["talker"]
    assert list(t) == list(TALKER_VARIANTS)
    assert t["eager"]["warmup_ms"] == [] and len(t["static+compile"]["warmup_ms"]) == WARMUPS
    assert all(r["identical"] for r in t["static+compile"]["turns"].values())
    assert all(not r["identical"] and r["first_difference"] == 5 for r in t["static+layers"]["turns"].values())
    assert all(r["n_codes"] == 120 and r["ms_per_code"] is not None for r in t["eager"]["turns"].values())
    assert "codes" not in t["eager"]["turns"]["c0"]                     # the codes are not saved
    # R4's own call each time, with the variant's arguments for generate
    assert {"talker_cache_implementation": "static", "talker_disable_compile": True} in omni.asked
    # the DiT: eager and compiled on eager's codes, each block size, the same noise seed
    d = blob["dit"]
    assert d["mode"] == "reduce-overhead" and d["errors"] == []
    assert set(d["blocks"]) == {"25", "50", "100"}
    assert all(r["max_abs_diff"] == 0 for rows in d["blocks"].values() for r in rows.values())
    s = summary(blob)
    assert s["talker"]["static+compile"]["identical"] == 2 and s["talker"]["static+layers"]["identical"] == 0
    assert s["best_talker"] in TALKER_VARIANTS and set(s["estimates"]) == {"25", "50", "100"}
    text = format_speed(blob)
    assert "| static+layers |" in text and "0 of 2" in text and "2 of 2" in text and "not a streaming rung" in text


def test_a_variant_that_fails_is_recorded_and_the_next_runs():
    omni = FakeOmni(fail="static")                                     # the library's route fails
    blob = run(omni, clips(3), 16000, compile_fn=Recorder())
    assert blob["talker"]["static+compile"]["error"] == "RuntimeError: cannot compile this"
    assert blob["talker"]["static+compile"]["turns"] == {}
    assert len(blob["talker"]["static+layers"]["turns"]) == 1 and blob["talker"]["static+layers"]["error"] is None
    assert "static+compile failed: RuntimeError: cannot compile this" in format_speed(blob)
    assert "forward" not in vars(omni.model.talker.model)               # the layers' compile was undone


def test_if_r4_itself_fails_the_probe_stops_and_says_so():
    blob = run(FakeOmni(fail="eager"), clips(3), 16000, compile_fn=Recorder())   # fails after the throwaway turn
    assert blob["talker"]["eager"]["error"] and list(blob["talker"]) == ["eager"] and blob["dit"]["blocks"] == {}
    s = summary(blob)
    assert s["estimates"] == {} and "eager failed" in format_speed(blob)


def test_r4s_speak_passes_extra_arguments_to_generate_and_nothing_else_changes():
    seen = {}

    class Inputs(dict):
        def to(self, device):
            return self

    omni = OmniS2S()
    omni.processor = type("P", (), {"apply_chat_template": lambda self, *a, **k: "prompt",
                                    "batch_decode": lambda self, ids, **k: ["Hello there."],
                                    "__call__": lambda self, **k: Inputs(input_ids=torch.zeros(1, 4, dtype=torch.long))})()

    def generate(**kw):
        seen.update(kw)
        return torch.zeros(1, 6, dtype=torch.long), torch.zeros(10)

    omni.model = SimpleNamespace(generate=generate)
    text, wav = omni.speak(np.zeros(16000, dtype=np.float32), 16000, talker_cache_implementation="static")
    assert text == "Hello there." and len(wav) == 10
    assert seen["talker_cache_implementation"] == "static" and seen["thinker_do_sample"] is False
    seen.clear()
    omni.speak(np.zeros(16000, dtype=np.float32), 16000)
    assert "talker_cache_implementation" not in seen and seen["speaker"] == "Chelsie"


def test_the_l40s_app_has_the_speed_entry_point():
    spec = importlib.util.spec_from_file_location("r4_bench", ROOT / "modal_app" / "r4_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod.speed_probe, "remote") and hasattr(mod.speed, "info")
    assert mod.SPEED_PATH == ROOT / SPEED
