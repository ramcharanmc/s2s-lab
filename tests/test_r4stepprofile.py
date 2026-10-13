"""
M33: the Talker step profile (s2slab.r4stepprofile) on the tiny Talker of tests/test_talkerloop.py, on
the CPU: each cache size timed through the captured loop's step (run eagerly here), the places each turn
filled, a cache too small cutting the reply and saying so, the breakdown by kind of operation, and the
follow-up rule. Skipped without transformers.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_talkerloop import NEW, PROMPT, FakeOmni  # noqa: E402

from s2slab import r4stepprofile as sp  # noqa: E402

FULL = PROMPT + NEW + 8                 # FakeOmni's own cache


def test_each_cache_size_is_timed_its_turns_counted_and_a_short_cache_cuts_and_says_so():
    clips = {f"c{i:02d}": np.full(16, float(i), dtype=np.float32) for i in range(4)}
    blob = sp.run(clips, 16, omni=FakeOmni(), compile_fn=lambda fn, **kw: fn, sizes=(FULL, PROMPT + NEW + 2, PROMPT + 40),
                  profiled=(FULL,))
    assert blob["clips"] == ["c00", "c03"] and list(blob["timing"]) == [str(FULL), str(PROMPT + NEW + 2), str(PROMPT + 40)]
    s = sp.summary(blob)["sizes"]
    roomy, tight = s[str(PROMPT + NEW + 2)], s[str(PROMPT + 40)]
    assert s[str(FULL)]["identical"] is None and s[str(FULL)]["used_max"] == PROMPT + NEW and s[str(FULL)]["cut"] == 0
    assert roomy["identical"] == 2 and roomy["cut"] == 0                 # masked places change nothing here
    assert tight["cut"] == 2 and tight["identical"] == 0 and tight["used_max"] == PROMPT + 40
    assert all(len(rec["warmup_ms"]) == sp.WARMUPS for rec in blob["timing"].values())
    b = blob["breakdown"][str(FULL)]
    assert b["codes"] == NEW and b["device"] == "cpu" and b["total_ms_a_code"] > 0 and "matmul" in b["ms_a_code"]
    text = sp.format_step_profile(blob)
    assert "The Talker: 2 layers" in text and "turns cut" in text and "cannot be read" in text


def test_the_talkers_size_and_the_floors_an_l40s_sets():
    z = sp.talker_size(FakeOmni().talker, sizes=(1024, 2048))
    assert z["layers"] == 2 and z["kv_heads"] == 2 and z["head_dim"] == 8
    assert z["kv_bytes"]["2048"] == 2 * z["kv_bytes"]["1024"] == 2 * 2 * 2048 * 2 * 8 * 4   # layers, K and V, places, heads, width, float32
    assert 0 < z["bytes_a_code"] < z["params"] * 4 and z["floor_ms"] >= 0


def test_kernels_are_sorted_into_kinds():
    assert sp.kind("void fmha_cutlassF_bf16_aligned_64x64_rf_sm80(...)") == "attention"
    assert sp.kind("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n") == "matmul"
    assert sp.kind("aten::addmm") == "matmul"
    assert sp.kind("void at::native::radixSortKVInPlace<...>") == "sampling"
    assert sp.kind("void at::native::vectorized_elementwise_kernel<...>") == "other"


def test_the_follow_up_rule_reads_the_2048_place_cache():
    row = lambda ms: {"ms_per_code": ms, "identical": True}                         # noqa: E731
    filled = lambda cut: [{"prompt": 400, "codes": 300, "cut": cut}]                # noqa: E731
    blob = {"sizes": [6144, 2048], "follow_up_ms": 8.0, "clips": ["a"], "load_ms": 1.0, "breakdown": {},
            "timing": {"6144": {"error": None, "warmup_ms": [], "turns": {"a": row(10.1)}},
                       "2048": {"error": None, "warmup_ms": [], "turns": {"a": row(7.6)}}},
            "turns_filled": {"6144": filled(False), "2048": filled(False)}}
    assert sp.summary(blob)["follow_up"] and "design note" in sp.summary(blob)["verdict"]
    blob["timing"]["2048"]["turns"]["a"] = row(8.4)
    assert not sp.summary(blob)["follow_up"] and "ends here" in sp.summary(blob)["verdict"]
    blob["timing"]["2048"]["turns"]["a"] = row(7.0)
    blob["turns_filled"]["2048"] = filled(True)
    assert not sp.summary(blob)["follow_up"] and "cannot be read" in sp.summary(blob)["verdict"]
