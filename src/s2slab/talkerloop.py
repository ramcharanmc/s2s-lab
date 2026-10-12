"""
M32b: the Talker's decode loop, ours, in place of transformers' generate.

WHY (docs/m32-log.md, M32b). R4 streamed's first block exists 1.4 s into a turn, but the stream is
made at only 0.86 of real time: a block of 96 codes (1.92 s of audio) takes about 1.66 s, so playback
must wait for the second block before it can start without a stall, 3.1 s in. The Talker writes a code
in 12.7 ms with its layers compiled (M31), yet reading its weights once on an L40S takes about a
millisecond: most of each code's time is spent around the layers, in generate's Python, its masks and
stopping checks, and in two host synchronisations inside torch.multinomial's validity checks.

WHAT IT DOES. TalkerLoop.generate takes what Qwen2.5-Omni's generate hands talker.generate, and returns
what talker.generate returns (the prompt's ids, then the codes, the end code included), for one reply:

    prefill    through the Talker's own prepare_inputs_for_generation and forward, as generate's first
               step (the rotary positions, the codec start and pad added to the prompt, the cache)
    each code  the step generate takes, done directly: the code's embedding plus the Thinker's reply
               part, the projection, the decoder (talker.model, as installed), the codec head; then the
               logits processors generate builds for these settings (repetition penalty, the
               suppressed start code, temperature, top-k, top-p), as tensor operations of fixed shape
    sampling   as torch.multinomial samples one: q ~ Exp(1), argmax(p / q), which is ATen's own path for
               one sample, without its two validity checks (each a host synchronisation)
    graph=True the per-code step on fixed buffers, captured once as a CUDA graph and replayed; `layers`
               runs the decoder inside it (default: talker.model's own eager forward, so that a
               compile_layers wrapper, a CUDA graph of its own, is not nested)

Each code is read back to the host once (to stop at an end code and to hand it to the streamer).

THE SAME CODES. Same weights, inputs, operations and random draws, so the same codes as generate on the
same seed: checked on a tiny Talker in tests/test_talkerloop.py, and on R4's own by the speed probe
(s2slab.r4loopspeed), which reports what it finds. One difference is known and harmless to the codes:
generate may run one step past the end code and undo it, drawing random numbers this loop does not, so
the noise token2wav draws after a whole-reply decode can differ. The streamed rung decodes its blocks
inside fork_rng (r4stream), so its audio does not depend on it.
"""

from __future__ import annotations

REPLY_MAX = 256           # the Thinker's reply part the graph's buffer holds; R4's reply is capped at 60 tokens
WARMUP_STEPS = 3          # steps run before the capture, on a side stream (CUDA graphs need it)
# Generation settings that would add a logits processor this loop does not apply (transformers 5.17,
# GenerationMixin._get_logits_processor): each must be off in the Talker's own generation config.
UNHANDLED = ("guidance_scale", "sequence_bias", "encoder_repetition_penalty", "no_repeat_ngram_size",
             "encoder_no_repeat_ngram_size", "bad_words_ids", "min_length", "min_new_tokens", "forced_bos_token_id",
             "forced_eos_token_id", "remove_invalid_values", "exponential_decay_length_penalty",
             "begin_suppress_tokens", "top_h", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
             "watermarking_config")
_OFF = {"guidance_scale": (None, 1, 1.0), "encoder_repetition_penalty": (None, 1.0), "no_repeat_ngram_size": (None, 0),
        "encoder_no_repeat_ngram_size": (None, 0), "min_length": (None, 0), "min_new_tokens": (None, 0),
        "remove_invalid_values": (None, False), "typical_p": (None, 1.0), "epsilon_cutoff": (None, 0.0),
        "eta_cutoff": (None, 0.0)}
IGNORED = ("disable_compile", "return_audio")      # passed by the outer generate; nothing to do with the codes


def check_generation_config(cfg) -> None:
    """Refuse a Talker whose generation config would make generate apply a processor this loop does not."""
    on = [k for k in UNHANDLED if getattr(cfg, k, None) not in _OFF.get(k, (None,))]
    if on:
        raise ValueError(f"the Talker's generation config sets {', '.join(on)}: TalkerLoop does not apply them")


def pick(logits, seen, suppress, do_sample: bool, temperature: float, top_k: int, top_p: float, penalty: float):
    """The next code from the last position's logits (1, V), float32: generate's processors for these
    settings, then one draw as torch.multinomial makes it. `seen` (1, V) marks every id in the sequence so
    far (the prompt's and the codes'); `suppress` (V,) the suppressed ids. Returns (1, 1) long."""
    import torch
    x = logits
    if penalty != 1.0:                                   # RepetitionPenaltyLogitsProcessor, on every id seen
        x = torch.where(seen, torch.where(x < 0, x * penalty, x / penalty), x)
    x = torch.where(suppress, -float("inf"), x)           # SuppressTokensLogitsProcessor
    if not do_sample:
        return torch.argmax(x, dim=-1, keepdim=True)
    if temperature != 1.0:                               # TemperatureLogitsWarper
        x = x / temperature
    if top_k:                                            # TopKLogitsWarper
        x = x.masked_fill(x < torch.topk(x, min(top_k, x.size(-1)))[0][..., -1, None], -float("inf"))
    if top_p < 1.0:                                      # TopPLogitsWarper, keeping at least one
        s, idx = torch.sort(x, descending=False)
        drop = s.softmax(dim=-1).cumsum(dim=-1) <= (1 - top_p)
        drop[..., -1:] = 0
        x = x.masked_fill(drop.scatter(1, idx, drop), -float("inf"))
    probs = torch.nn.functional.softmax(x, dim=-1)
    q = torch.empty_like(probs).exponential_(1)          # torch.multinomial(probs, 1), as ATen draws one sample
    return torch.argmax(torch.div(probs, q), dim=-1, keepdim=True)


class TalkerLoop:
    """talker.generate for one reply, with the per-code step done directly (see the module's docstring).

    `cache` is the Talker's own cache (FastOmni's StaticCache; graph=True needs a static one). `on` lets a
    caller switch back to generate without uninstalling (the speed probe does)."""

    def __init__(self, talker, cache, graph: bool = False, layers=None):
        self.talker, self.cache, self.graph = talker, cache, graph
        self.layers = layers
        self.on = True
        self.last_codes = 0
        self._g = None                                   # the captured graph, once made
        self._settings = None

    def install(self):
        """talker.generate becomes this loop while `on`, generate otherwise. Returns the undo."""
        had = "generate" in vars(self.talker)            # a wrapper already installed (StageClock's, say)
        inner = self.talker.generate

        def generate(*args, **kwargs):
            return self.generate(*args, **kwargs) if self.on else inner(*args, **kwargs)

        self.talker.generate = generate
        return (lambda: setattr(self.talker, "generate", inner)) if had else (lambda: delattr(self.talker, "generate"))

    # ------------------------------------------------------------------ the loop
    def generate(self, input_ids=None, inputs_embeds=None, attention_mask=None, thinker_reply_part=None,
                 suppress_tokens=None, max_new_tokens: int = 4096, do_sample: bool = True, top_k: int = 40,
                 top_p: float = 0.8, temperature: float = 0.9, eos_token_id=(8292, 8294), pad_token_id=None,
                 repetition_penalty: float = 1.05, past_key_values=None, streamer=None, **model_kwargs):
        import torch
        from transformers import GenerationConfig
        for k in IGNORED:
            model_kwargs.pop(k, None)
        settings = [k for k in model_kwargs if hasattr(GenerationConfig(), k)]
        if settings:
            raise TypeError(f"TalkerLoop does not handle the generation setting(s) {', '.join(sorted(settings))}")
        if input_ids is None or inputs_embeds is None or attention_mask is None or input_ids.shape[0] != 1:
            raise ValueError("TalkerLoop speaks one reply at a time, from the prompt's ids, embeddings and mask")
        if past_key_values is not None and past_key_values is not self.cache:
            raise ValueError("TalkerLoop runs on its own cache")
        check_generation_config(self.talker.generation_config)
        tk, dev = self.talker, input_ids.device
        eos = {int(e) for e in (eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])}
        sample = dict(do_sample=bool(do_sample), temperature=float(temperature), top_k=int(top_k or 0),
                      top_p=float(top_p if top_p is not None else 1.0), penalty=float(repetition_penalty or 1.0))
        with torch.no_grad():
            if self.graph and self._g is None:
                self._capture(dev, sample)
            if self.graph and sample != self._settings:
                raise ValueError("the captured step was made for other sampling settings")
            past = int(self.cache.get_seq_length())
            mi = tk.prepare_inputs_for_generation(
                input_ids, next_sequence_length=inputs_embeds.shape[1] - past, past_key_values=self.cache,
                attention_mask=attention_mask, inputs_embeds=inputs_embeds, is_first_iteration=True, use_cache=True,
                thinker_reply_part=thinker_reply_part, **model_kwargs)
            out = tk(**mi, return_dict=True)
            logits = out.logits[:, -1].to(copy=True, dtype=torch.float32)
            reply, rope = out.thinker_reply_part, tk.rope_deltas
            vocab = logits.shape[-1]
            seen = torch.zeros((1, vocab), dtype=torch.bool, device=dev)
            seen.scatter_(1, input_ids, True)
            suppress = torch.zeros(vocab, dtype=torch.bool, device=dev)
            if suppress_tokens:
                suppress[torch.as_tensor(list(suppress_tokens), device=dev)] = True
            if streamer is not None:
                streamer.put(input_ids.cpu())
            code = pick(logits, seen, suppress, **sample)
            codes = []
            if self.graph:
                self._begin(code, seen, suppress, reply, rope)
            else:
                ones = attention_mask.new_ones((1, attention_mask.shape[1] + max_new_tokens))
            for i in range(max_new_tokens):
                c = int(code)                                    # the one host synchronisation a code
                codes.append(c)
                if streamer is not None:
                    streamer.put(code.view(1).cpu())
                if c in eos or i == max_new_tokens - 1:
                    break
                if self.graph:
                    code = self._replay()
                    continue
                seen.scatter_(1, code, True)
                mask = ones[:, :attention_mask.shape[1] + i + 1]         # generate's 2-D mask, one longer each code
                code = pick(self._forward(code, i, reply, rope, mask), seen, suppress, **sample)
            if streamer is not None:
                streamer.end()
        self.last_codes = len(codes)
        return torch.cat([input_ids, torch.tensor([codes], dtype=input_ids.dtype, device=dev)], dim=1)

    def _forward(self, code, i, reply, rope, mask):
        """One code through the Talker, as its forward does after the prefill: the logits (1, V), float32."""
        import torch
        tk = self.talker
        j = min(i, reply.shape[1] - 1)
        emb = tk.get_input_embeddings()(code) + reply[:, j:j + 1, :]
        delta = self.cache.get_seq_length() + rope
        pos = torch.arange(1, device=code.device).view(1, -1).expand(1, -1).add(delta).unsqueeze(0).expand(3, -1, -1)
        out = tk.model(attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                       inputs_embeds=tk.thinker_to_talker_proj(emb), use_cache=True, return_dict=True)
        return tk.codec_head(out[0]).float()[:, -1]

    # ------------------------------------------------------------------ the captured step
    def _buffers(self, dev, dtype, width, vocab) -> None:
        import torch
        z = lambda *shape, **kw: torch.zeros(*shape, device=dev, **kw)       # noqa: E731
        self._code, self._out = z((1, 1), dtype=torch.long), z((1, 1), dtype=torch.long)
        self._i, self._last = z(1, dtype=torch.long), z(1, dtype=torch.long)
        self._reply, self._rope = z((1, REPLY_MAX, width), dtype=dtype), z((1, 1), dtype=torch.long)
        self._seen, self._suppress = z((1, vocab), dtype=torch.bool), z(vocab, dtype=torch.bool)

    def _static_step(self) -> None:
        """One code on the fixed buffers: what _forward then pick do, with every changing value in a tensor."""
        import torch
        tk = self.talker
        j = torch.minimum(self._i, self._last)
        emb = tk.get_input_embeddings()(self._code) + self._reply.index_select(1, j)
        delta = self.cache.layers[0].cumulative_length + self._rope
        pos = torch.arange(1, device=emb.device).view(1, -1).expand(1, -1).add(delta).unsqueeze(0).expand(3, -1, -1)
        run = self.layers or (lambda **kw: type(tk.model).forward(tk.model, **kw))
        out = run(attention_mask=None, position_ids=pos, past_key_values=self.cache,
                  inputs_embeds=tk.thinker_to_talker_proj(emb), use_cache=True, return_dict=True)
        logits = tk.codec_head(out[0]).float()[:, -1]
        self._seen.scatter_(1, self._code, True)
        code = pick(logits, self._seen, self._suppress, **self._settings)
        self._out.copy_(code)
        self._code.copy_(code)
        self._i.add_(1)

    def _capture(self, dev, sample) -> None:
        """Make the buffers, run the step a few times and capture it (on CUDA); the cache and the random
        state are put back as they were, so the turn that triggered it is not disturbed."""
        import torch
        tk = self.talker
        self._settings = dict(sample)
        self._buffers(dev, tk.get_input_embeddings().weight.dtype, tk.config.embedding_size, tk.codec_head.out_features)
        if dev.type != "cuda":                           # no graph off CUDA: the same step, run eagerly
            self._g, self._replay_fn = "eager", self._static_step
            return
        ids = [dev.index if dev.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=ids):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(WARMUP_STEPS):
                    self._static_step()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._static_step()
        self.cache.reset()
        self._g, self._replay_fn = graph, graph.replay

    def _begin(self, code, seen, suppress, reply, rope) -> None:
        if reply.shape[1] > REPLY_MAX:
            raise ValueError(f"a reply part of {reply.shape[1]} tokens: the captured step holds {REPLY_MAX}")
        self._code.copy_(code)
        self._i.zero_()
        self._last.fill_(reply.shape[1] - 1)
        self._reply[:, :reply.shape[1]].copy_(reply)
        self._rope.copy_(rope)
        self._seen.copy_(seen)
        self._suppress.copy_(suppress)

    def _replay(self):
        self._replay_fn()
        return self._out
