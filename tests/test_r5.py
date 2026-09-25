"""
Tests for s2slab.r5's adapters that need no model: what each is set up with.
Their streaming is tested through tests/test_duplex.py's fake model; the real
ones run only on Modal (modal_app/r5_bench.py).
"""

import pytest

from s2slab import r0, r5


def test_personaplex_is_registered_with_r0s_instruction_as_its_role_and_offline_pys_sampling():
    m = r5.get("r5-personaplex")
    assert (m.repo, m.voice, m.role, m.device) == ("nvidia/personaplex-7b-v1", "NATF2.pt", r0.SYSTEM_PROMPT, "cuda")
    assert len(m.revision) == 40 and (m.temp, m.temp_text, m.top_k, m.top_k_text) == (0.8, 0.7, 250, 25)
    assert m.rung_id == "r5-personaplex" and "NATF2" in m.description
    m.end()                                                                      # nothing open: a no-op


def test_the_role_is_wrapped_in_system_tags_once():
    assert r5.system_tags("  Be kind. ") == "<system> Be kind. <system>"
    assert r5.system_tags("<system> Be kind. <system>") == "<system> Be kind. <system>"
    assert r5.system_tags(r0.SYSTEM_PROMPT).startswith("<system> You are a friendly phone agent")


def test_the_three_models_are_known_and_nothing_else_is():
    assert set(r5.MODELS) == {"r5-moshi", "r5-moshi-lora", "r5-personaplex"}
    with pytest.raises(KeyError, match="r5-moshi, r5-moshi-lora, r5-personaplex"):
        r5.get("r5-nothing")
