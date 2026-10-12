"""
M32b: our Talker decode loop (s2slab.talkerloop) against transformers' generate, on a tiny Talker of
Qwen2.5-Omni's own architecture with random weights, on the CPU. The codes must be the same, code for
code, on the same seed: through the route R4's fast rungs take (a static cache, the 2-D mask kept, the
decoder's one-code steps run without a mask: r4speed's fixes), through plain generate, and with the step
on fixed buffers (what is captured as a CUDA graph on the GPU, run eagerly here). Skipped without
transformers.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from s2slab import talkerloop  # noqa: E402
from s2slab.r4speed import compile_layers, keep_2d_mask  # noqa: E402

PROMPT, REPLY, NEW = 14, 9, 80          # prompt positions, the Thinker's reply part, codes asked for
SAMPLING = dict(do_sample=True, top_k=40, top_p=0.8, temperature=0.9, eos_token_id=[8292, 8294], pad_token_id=8292,
                repetition_penalty=1.05)


def tiny_talker(seed=0):
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTalkerConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniTalkerForConditionalGeneration
    torch.manual_seed(seed)
    cfg = Qwen2_5OmniTalkerConfig(embedding_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                  rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1]})
    cfg._attn_implementation = "sdpa"
    return Qwen2_5OmniTalkerForConditionalGeneration(cfg).eval()


def turn(talker, seed=1):
    """What Qwen2.5-Omni's generate hands talker.generate for one reply, made up at the tiny size."""
    g = torch.Generator().manual_seed(seed)
    e = talker.config.embedding_size
    ids = torch.cat([torch.full((1, PROMPT - 2), talker.codec_mask_token),
                     torch.tensor([[talker.codec_pad_token, talker.codec_bos_token]])], dim=1)
    text = torch.randint(0, 1000, (1, PROMPT), generator=g)
    return dict(input_ids=ids, input_text_ids=text, inputs_embeds=torch.randn(1, PROMPT, e, generator=g),
                thinker_reply_part=torch.randn(1, REPLY, e, generator=g), attention_mask=torch.ones(1, PROMPT, dtype=torch.long),
                suppress_tokens=[talker.codec_bos_token], use_audio_in_video=False)


def static_cache(talker):
    from transformers import StaticCache
    return StaticCache(config=talker.config, max_cache_len=PROMPT + NEW + 8)


def fast_route(talker):
    """r4speed's fixes, as FastOmni installs them, with an identity 'compile' (the CPU has no graphs)."""
    keep_2d_mask(talker)
    compile_layers(talker, lambda fn, **kw: fn)


def generate_codes(talker, inputs, cache=None, seed=7, **extra):
    torch.manual_seed(seed)
    with torch.inference_mode():
        if cache is not None:
            cache.reset()
        out = talker.generate(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                              max_new_tokens=NEW, past_key_values=cache, disable_compile=True, **SAMPLING, **extra)
    return out[0, PROMPT:].tolist(), torch.random.get_rng_state()


def loop_codes(loop, inputs, seed=7, **extra):
    torch.manual_seed(seed)
    with torch.inference_mode():
        loop.cache.reset()
        out = loop.generate(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                            max_new_tokens=NEW, past_key_values=loop.cache, disable_compile=True, **SAMPLING, **extra)
    return out[0, PROMPT:].tolist()


def test_one_draw_is_torch_multinomials_and_the_processors_are_generates():
    from transformers.generation.logits_process import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                                                        SuppressTokensLogitsProcessor, TemperatureLogitsWarper,
                                                        TopKLogitsWarper, TopPLogitsWarper)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 500, (1, 30), generator=g)
    procs = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.05), SuppressTokensLogitsProcessor([7]),
                                 TemperatureLogitsWarper(0.9), TopKLogitsWarper(40), TopPLogitsWarper(0.8)])
    seen = torch.zeros(1, 500, dtype=torch.bool).scatter_(1, ids, True)
    suppress = torch.zeros(500, dtype=torch.bool)
    suppress[7] = True
    for k in range(50):
        logits = torch.randn(1, 500, generator=g) * 3
        torch.manual_seed(k)
        want = torch.multinomial(torch.nn.functional.softmax(procs(ids, logits.clone()), dim=-1), 1)
        torch.manual_seed(k)
        got = talkerloop.pick(logits, seen, suppress, True, 0.9, 40, 0.8, 1.05)
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_loop_writes_generates_codes_on_the_fast_route(seed):
    talker = tiny_talker()
    fast_route(talker)
    inputs = turn(talker)
    cache = static_cache(talker)
    want, rng_after = generate_codes(talker, inputs, cache, seed)
    loop = talkerloop.TalkerLoop(talker, cache)
    assert loop_codes(loop, inputs, seed) == want and len(want) == NEW


def test_the_loop_writes_generates_codes_on_plain_generate():
    from transformers import DynamicCache
    talker = tiny_talker(1)
    inputs = turn(talker, 2)
    want, _ = generate_codes(talker, inputs)
    loop = talkerloop.TalkerLoop(talker, DynamicCache(config=talker.config))
    assert loop_codes(loop, inputs) == want


def test_the_step_on_fixed_buffers_writes_the_same_codes():
    talker = tiny_talker()
    fast_route(talker)
    inputs = turn(talker)
    cache = static_cache(talker)
    want, _ = generate_codes(talker, inputs, cache)
    loop = talkerloop.TalkerLoop(talker, cache, graph=True)
    assert loop_codes(loop, inputs) == want
    assert loop_codes(loop, turn(talker, 5)) == generate_codes(talker, turn(talker, 5), cache)[0]   # a second turn, buffers kept


def test_it_stops_at_an_end_code_includes_it_and_streams_each_code_after_the_prompt():
    talker = tiny_talker()
    fast_route(talker)
    inputs = turn(talker)
    cache = static_cache(talker)
    head = torch.nn.Linear(talker.codec_head.in_features, talker.codec_head.out_features, bias=True)
    with torch.no_grad():                                 # an end code likely enough to end the reply early
        head.weight.copy_(talker.codec_head.weight)
        head.bias.zero_()
        head.bias[8294] = 0.6
    talker.codec_head = head
    want, _ = generate_codes(talker, inputs, cache)
    puts = []

    class Streamer:
        def put(self, value):
            puts.append(value.reshape(-1).tolist())

        def end(self):
            puts.append("end")

    loop = talkerloop.TalkerLoop(talker, cache)
    torch.manual_seed(7)
    with torch.inference_mode():
        cache.reset()
        out = loop.generate(**inputs, max_new_tokens=NEW, past_key_values=cache, streamer=Streamer(), **SAMPLING)
    got = out[0, PROMPT:].tolist()
    assert got == want and got[-1] in (8292, 8294) and len(got) < NEW
    assert puts[0] == inputs["input_ids"].reshape(-1).tolist() and puts[-1] == "end"
    assert [p[0] for p in puts[1:-1]] == got and loop.last_codes == len(got)


def test_it_installs_switches_back_to_generate_and_undoes():
    talker = tiny_talker()
    fast_route(talker)
    loop = talkerloop.TalkerLoop(talker, static_cache(talker))
    undo = loop.install()
    assert talker.generate.__name__ == "generate" and "generate" in vars(talker)
    loop.on = False
    inputs = turn(talker)
    want, _ = generate_codes(talker, inputs, loop.cache)       # off: generate itself, through the switch
    loop.on = True
    assert generate_codes(talker, inputs, loop.cache)[0] == want
    undo()
    assert "generate" not in vars(talker)


class FakeOmni:
    """FastOmni's shape around the tiny Talker: load() installs r4speed's fixes and the cache; speak() hands
    talker.generate a made-up turn (one per clip) and token2wav its codes, as R4's call does."""

    def __init__(self):
        from types import SimpleNamespace
        self.talker, self.cache = tiny_talker(), None

        class T2W:
            def forward(self, code, **kw):
                return torch.zeros(10)

        t2w = T2W()
        t2w.code2wav_dit_model = SimpleNamespace(sample=lambda *a, **k: None)
        t2w.code2wav_bigvgan_model = SimpleNamespace(forward=lambda *a, **k: None)
        self.model = SimpleNamespace(thinker=SimpleNamespace(generate=lambda **k: None), talker=self.talker, token2wav=t2w)

    def load(self):
        if self.cache is None:
            fast_route(self.talker)
            self.cache = static_cache(self.talker)

    def speak(self, audio, sr, **extra):
        torch.manual_seed(0)
        with torch.inference_mode():
            self.cache.reset()
            out = self.model.talker.generate(**turn(self.talker, int(audio[0]) + 1), max_new_tokens=NEW,
                                             past_key_values=self.cache, disable_compile=True, **SAMPLING)
            self.model.token2wav.forward(out[:, PROMPT:])
        return "a reply", np.zeros(10, dtype=np.float32)


def test_the_speed_probe_times_each_variant_and_finds_the_loops_codes_generates():
    from s2slab import r4loopspeed
    clips = {f"c{i:02d}": np.full(16, float(i), dtype=np.float32) for i in range(7)}
    blob = r4loopspeed.run(clips, 16, omni=FakeOmni(), compile_fn=lambda fn, **kw: fn)
    assert blob["clips"] == ["c00", "c03", "c06"] and list(blob["talker"]) == list(r4loopspeed.VARIANTS)
    for name in ("loop", "loop+graph"):
        v = blob["talker"][name]
        assert v["error"] is None and len(v["warmup_ms"]) == 2
        assert all(t["identical"] and t["n_codes"] == NEW for t in v["turns"].values())
    s = r4loopspeed.summary(blob)
    assert s["variants"]["loop"]["identical"] == 3 and s["variants"]["generate"]["identical"] is None
    text = r4loopspeed.format_loop_speed(blob)
    assert "| loop+graph | 3 |" in text and "Stop rule:" in text


def test_the_stop_rule_reads_the_faster_loop():
    from s2slab import r4loopspeed
    row = lambda ms: {"ms_per_code": ms, "n_codes": 100, "identical": True}          # noqa: E731
    blob = {"stop_ms": 10.0, "clips": ["a"], "load_ms": 1.0, "talker": {
        "generate": {"error": None, "warmup_ms": [], "turns": {"a": row(12.7)}},
        "loop": {"error": None, "warmup_ms": [1.0, 1.0], "turns": {"a": row(11.0)}},
        "loop+graph": {"error": "RuntimeError: capture failed", "warmup_ms": [], "turns": {}}}}
    s = r4loopspeed.summary(blob)
    assert s["best_loop"] == "loop" and not s["go"] and "stops here" in s["verdict"]
    blob["talker"]["loop+graph"] = {"error": None, "warmup_ms": [], "turns": {"a": row(4.2)}}
    s = r4loopspeed.summary(blob)
    assert s["best_loop"] == "loop+graph" and s["go"] and "step 3" in s["verdict"]
    assert "loop+graph failed" not in r4loopspeed.format_loop_speed(blob)


def test_it_refuses_what_it_does_not_do():
    talker = tiny_talker()
    loop = talkerloop.TalkerLoop(talker, static_cache(talker))
    inputs = turn(talker)
    with pytest.raises(TypeError, match="no_repeat_ngram_size"):
        loop_codes(loop, inputs, no_repeat_ngram_size=3)
    two = dict(inputs, input_ids=inputs["input_ids"].repeat(2, 1))
    with pytest.raises(ValueError, match="one reply"):
        loop_codes(loop, two)
    talker.generation_config.min_new_tokens = 5
    with pytest.raises(ValueError, match="min_new_tokens"):
        loop_codes(loop, inputs)
