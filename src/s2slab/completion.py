"""
T1b's signal (PR-001): how likely a small pretrained language model thinks the sentence ends right
after the words heard so far.

    PYTHONPATH=src python -m s2slab.completion dev      T1 against T1b at every cut-off in GRID, on DEV
                                                        sentences (seeds 0 and 1000), and the cut-off RULE picks

WHAT THE SIGNAL IS. P(end) is the probability mass the model puts on a next token that begins with
. ? or !, or that ends the text, given the partial transcript as it stands (lower-case, no
punctuation, as a recognizer gives it). Nothing in it was fitted to any sentence here: the model is
Qwen2.5-0.5B-Instruct as published, R0 to R2's LLM, already on the machine. The idea is TurnGPT's
(Ekstedt & Skantze, 2020): a language model over the words so far predicts whether the turn can end.

COST. About a quarter of a second a prefix on a laptop CPU. A sweep asks the same prefixes many
times (a sentence's prefixes recur across turns and policies), so scores are cached by text.

The cut-off is chosen on DEV by RULE, written down before the tuning run, and then frozen in
T1B_THETA (PR-001, V2). The EVAL sentences are never scored here.
"""

from __future__ import annotations

import json
import pathlib

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GRID = (0.005, 0.01, 0.02, 0.05, 0.1)
MARGIN_P50_MS, MARGIN_P90_MS = 25.0, 50.0       # half of S3's allowances (50 and 100 ms)
DEV_SEEDS, DEV_N = (0, 1000), 250
DEV_OUT = pathlib.Path("runs/turn-taking/pr001-dev.json")
RULE = ("the largest cut-off in GRID for which, on each DEV seed (0 and 1000, 250 turns a cohort, "
        "whole-sentence turns), T1b's clean median wait is at most T1's + 25 ms and its clean p90 at most "
        "T1's + 50 ms: half of S3's allowances, a margin for EVAL")
T1B_THETA: float | None = 0.01                  # chosen by RULE on DEV (runs/turn-taking/pr001-dev.json), frozen (PR-001, V2)


class LMEnd:
    """P(the sentence ends here | the words so far), from MODEL, cached by text; loaded on first use."""

    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.cache: dict[str, float] = {}
        self._tok = self._model = self._ends = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float32).eval()
        ends = {i for t, i in tok.get_vocab().items()
                if tok.convert_tokens_to_string([t]).lstrip()[:1] in (".", "?", "!")}
        ends |= {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>"),
                             tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int)}
        self._tok, self._ends = tok, torch.tensor(sorted(ends))

    def _score(self, partial: str) -> float:
        import torch
        if self._model is None:
            self._load()
        with torch.no_grad():
            ids = self._tok(partial, return_tensors="pt").input_ids
            probs = self._model(ids).logits[0, -1].softmax(-1)
        return float(probs[self._ends].sum())

    def __call__(self, partial: str) -> float:
        if partial not in self.cache:
            self.cache[partial] = self._score(partial)
        return self.cache[partial]


def t1b(theta: float | None = None, scorer=None):
    """T1b at the frozen cut-off (or the one given), with the language model (or the scorer given)."""
    from .policies import CompletionThreshold
    theta = T1B_THETA if theta is None else theta
    if theta is None:
        raise RuntimeError("T1b's cut-off is not chosen yet: run `python -m s2slab.completion dev`")
    return CompletionThreshold(scorer if scorer is not None else LMEnd(), theta)


def dev(scorer=None, seeds=DEV_SEEDS, n: int = DEV_N) -> dict:
    """T1 against T1b at every cut-off in GRID: DEV sentences, whole-sentence turns, the given seeds.
    Returns every row and the cut-off RULE picks (None if no cut-off keeps the margin)."""
    from .policies import get
    from .simulate import make_all
    from .sweep import evaluate
    scorer = scorer if scorer is not None else LMEnd()
    t1 = get("T1_semantic_500ms")
    rows = []
    for seed in seeds:
        cohorts = make_all(n, seed, split="dev", turns="sentence")
        base = {c: evaluate(t1, u) for c, u in cohorts.items()}
        for theta in GRID:
            pol = t1b(theta, scorer)
            for c, u in sorted(cohorts.items()):
                r, b = evaluate(pol, u), base[c]
                rows.append({"seed": seed, "theta": theta, "cohort": c, "n": b.n,
                             "cut_t1": b.cutoff_rate, "cut_t1b": r.cutoff_rate,
                             "p50_t1": b.latency.p50, "p50_t1b": r.latency.p50,
                             "p90_t1": b.latency.p90, "p90_t1b": r.latency.p90})
    passing = [th for th in GRID
               if all(row["p50_t1b"] - row["p50_t1"] <= MARGIN_P50_MS and row["p90_t1b"] - row["p90_t1"] <= MARGIN_P90_MS
                      for row in rows if row["theta"] == th and row["cohort"] == "clean")]
    return {"model": getattr(scorer, "model_name", "a scorer given by the caller"), "rule": RULE,
            "grid": list(GRID), "seeds": list(seeds), "n_per_cohort": n, "split": "dev", "turns": "sentence",
            "rows": rows, "chosen": max(passing) if passing else None}


def format_dev(blob: dict) -> str:
    lines = ["| cut-off | seed | disfluent cut T1 -> T1b | slow cut T1 -> T1b | clean median wait T1 -> T1b | clean p90 T1 -> T1b |",
             "|---:|---:|---|---|---|---|"]
    rows = {(r["theta"], r["seed"], r["cohort"]): r for r in blob["rows"]}
    for th in blob["grid"]:
        for seed in blob["seeds"]:
            d, s, c = (rows[(th, seed, k)] for k in ("disfluent", "slow", "clean"))
            lines.append(f"| {th:g} | {seed} | {d['cut_t1']:.0%} -> {d['cut_t1b']:.0%} | {s['cut_t1']:.0%} -> {s['cut_t1b']:.0%} "
                         f"| {c['p50_t1']:.0f} -> {c['p50_t1b']:.0f} ms ({c['p50_t1b'] - c['p50_t1']:+.0f}) "
                         f"| {c['p90_t1']:.0f} -> {c['p90_t1b']:.0f} ms ({c['p90_t1b'] - c['p90_t1']:+.0f}) |")
    pick = blob["chosen"]
    lines += ["", f"rule: {blob['rule']}", f"chosen: {pick if pick is not None else 'none: no cut-off keeps the margin'}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    from .sweep import _git_commit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["dev"])
    ap.add_argument("--code", default="", help="the code the run used, when it is not a commit yet")
    a = ap.parse_args(argv)
    if DEV_OUT.exists():
        raise SystemExit(f"{DEV_OUT} exists: the cut-off was chosen once")
    blob = dev()
    blob["commit"] = a.code or _git_commit()
    print(format_dev(blob))
    DEV_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEV_OUT.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    print(f"\nsaved -> {DEV_OUT}")


if __name__ == "__main__":
    main()
