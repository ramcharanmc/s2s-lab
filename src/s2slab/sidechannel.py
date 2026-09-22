"""
The affect side-channel: a word about the voice, handed to the LLM beside the words that were said.

The v2 plan, R3: "Routing eGeMAPS tags around the text bottleneck is a poor
man's R3 ... how much of R3's benefit does a $0 side-channel recover?" and its
week 3: "affect side-channel w/ shuffled-tag control".

    clip -> R1's recognizer -> "I think I have a doctor's appointment."   the words: the same for both clips of a pair
         -> the affect tagger -> "fearful"                                the side-channel: one word about the voice
    LLM is asked:   A caller said: "I think I have a doctor's appointment." Their tone of voice was fearful.
                    Judging by their tone of voice, is the caller upset?

THE PROBE IS A QUESTION ONLY THE VOICE CAN ANSWER (the plan's "killer
experiment"). Nothing here can judge whether a spoken reply got kinder, so the
LLM -- R1's own Qwen2.5-0.5B, unchanged -- is asked outright. Its answer is not
generated and parsed: the probability of "Yes" against "No" as the next token is
read off directly, which is deterministic and cannot fail to parse.

FOUR CONDITIONS, the same clips and the same LLM each time:
    none       no tag                      R1 as it is. Both clips of a pair have (nearly always)
                                           the same words, so they get the same answer: blind
    tagger     the tagger's word           THE SIDE-CHANNEL
    oracle     the emotion that was acted  what a perfect tagger would buy: separates the
                                           tagger's mistakes from the LLM's
    shuffled   the tagger's words, dealt   THE CONTROL. If this scores like "tagger", the gain
               to the wrong clips          came from the prompt's wording, not from the voice

A PAIR IS RIGHT when the loaded clip gets Yes AND its neutral twin gets No
(p_yes above and below 0.5). "Ranked right" -- the loaded clip merely gets the
higher p_yes -- is reported beside it, because a small LLM may lean to one
answer whatever it is shown. Per emotion, never pooled.

THE WORDING WAS CHOSEN ON THE DEV PAIRS, with the ORACLE tag, so that only the
LLM's part was being judged. Four wordings were tried. All four ranked every
dev pair right; three left the 0.5B model answering "No" to nearly everything
(3 to 5 of 21 pairs answered right), because a bracketed "[voice: sounds
angry]" beside the transcript did not outweigh its prior. Saying the tone in a
sentence and asking "judging by their tone of voice" answered 21 of 21 right.
That wording is fixed below; the test pairs had no part in it.

Every (clip, word) the LLM could be shown is scored once and cached, so the
shuffled control is an average over N_SHUFFLES seeded deals at no further cost.

    python -m s2slab.sidechannel dev                        the 21 dev pairs
    python -m s2slab.sidechannel test --confirm PR-005 --save runs/prosody/pr005.json
                                                            the 39 locked pairs: a registered run, kept"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Callable

import numpy as np

from .affect import CACHE, LABELS, TAGGER, WORDS, Tagger, features_for
from .prosodyset import EMOTIONS, PROSODY_ROOT, SEED, LoadedPair, load

PROBE_SYSTEM = "Answer with one word: Yes or No."
QUESTION = "Judging by their tone of voice, is the caller upset?"
N_SHUFFLES = 1000
CONDITIONS = ("none", "tagger", "oracle", "shuffled")


def user_text(transcript: str, word: str | None) -> str:
    """What the LLM is shown. word=None is the no-tag condition."""
    said = f'A caller said: "{transcript}"' + (f" Their tone of voice was {word}." if word is not None else "")
    return f"{said}\n{QUESTION}"


class YesNoScorer:
    """R1's LLM, asked a yes/no question; returns the probability of Yes.

    Takes a QwenLLM (r0.py), loaded by load(). p_yes() runs ONE forward pass,
    reads the next-token logits, and compares the tokens "Yes" and "No":
        p_yes = P(Yes) / (P(Yes) + P(No)),   mass = P(Yes) + P(No)
    mass says how much the model wanted to answer yes-or-no at all.
    """

    def __init__(self, llm=None):
        if llm is None:
            from .r0 import QwenLLM
            llm = QwenLLM()
        self.llm = llm

    def load(self) -> "YesNoScorer":
        self.llm.load()
        ids = [self.llm.tokenizer.encode(w, add_special_tokens=False) for w in ("Yes", "No")]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"'Yes' and 'No' must be single tokens for this tokenizer, got {ids}")
        self.yes, self.no = ids[0][0], ids[1][0]
        return self

    def p_yes(self, transcript: str, word: str | None) -> tuple[float, float]:
        import torch
        tok = self.llm.tokenizer
        prompt = tok.apply_chat_template([{"role": "system", "content": PROBE_SYSTEM},
                                          {"role": "user", "content": user_text(transcript, word)}],
                                         tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(self.llm.device)
        with torch.inference_mode():
            p = torch.softmax(self.llm.model(**inputs).logits[0, -1].float(), dim=-1)
        yes, no = float(p[self.yes]), float(p[self.no])
        return yes / (yes + no), yes + no


def score_all(clips: dict[str, str], scorer: Callable[[str, str | None], tuple[float, float]],
              cache: pathlib.Path | None = None) -> dict[str, dict[str, float]]:
    """p_yes for every clip under every word it could be shown.

    Takes:   {clip name: transcript}; scorer(transcript, word) -> (p_yes, mass);
             a JSON cache (results/ is ignored by git).
    Returns: {clip: {"none": p, "neutral": p, "sad": p, "fearful": p, "angry": p, "mass": m}}
             -- mass is the no-tag prompt's.
    """
    known = json.loads(cache.read_text(encoding="utf-8")) if cache is not None and cache.exists() else {}
    fresh = False
    for clip, transcript in clips.items():
        if clip in known and known[clip].get("transcript") == transcript:
            continue
        p_none, mass = scorer(transcript, None)
        known[clip] = {"transcript": transcript, "none": p_none, "mass": mass,
                       **{WORDS[label]: scorer(transcript, WORDS[label])[0] for label in LABELS}}
        fresh = True
    if fresh and cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1) + "\n", encoding="utf-8")
    return {clip: known[clip] for clip in clips}


def _pair_right(p_neutral: float, p_loaded: float) -> tuple[bool, bool]:
    """(answered right: loaded Yes and neutral No; ranked right: loaded strictly higher)."""
    return (p_loaded > 0.5 and p_neutral <= 0.5), p_loaded > p_neutral


def evaluate(pairs, p: dict[str, dict[str, float]], tags: dict[str, str],
             n_shuffles: int = N_SHUFFLES, seed: int = SEED) -> dict[str, dict[str, tuple[float, float, int]]]:
    """{condition: {emotion: (pairs answered right, pairs ranked right, pairs)}}.

    Takes:   the pairs; score_all()'s table; {clip: the tagger's word}.
    The shuffled control deals the tagger's words to the clips at random --
    all clips of all pairs together -- n_shuffles times with a seeded generator,
    and reports the MEAN number of pairs right, so it is a float.
    """
    clips = [c for pr in pairs for c in (pr.neutral, pr.loaded)]
    rng = random.Random(seed)
    deals = []
    for _ in range(n_shuffles):
        words = [tags[c] for c in clips]
        rng.shuffle(words)
        deals.append(dict(zip(clips, words)))

    def count(word_of) -> dict[str, tuple[float, float, int]]:
        out = {}
        for emotion in EMOTIONS:
            mine = [pr for pr in pairs if pr.emotion == emotion]
            got = [_pair_right(p[pr.neutral][word_of(pr.neutral, pr, False)], p[pr.loaded][word_of(pr.loaded, pr, True)])
                   for pr in mine]
            out[emotion] = (float(sum(a for a, _ in got)), float(sum(r for _, r in got)), len(mine))
        return out

    result = {
        "none": count(lambda clip, pr, is_loaded: "none"),
        "tagger": count(lambda clip, pr, is_loaded: tags[clip]),
        "oracle": count(lambda clip, pr, is_loaded: WORDS[pr.emotion] if is_loaded else WORDS["NEU"]),
    }
    dealt = [count(lambda clip, pr, is_loaded, deal=deal: deal[clip]) for deal in deals]
    result["shuffled"] = {e: (float(np.mean([d[e][0] for d in dealt])), float(np.mean([d[e][1] for d in dealt])),
                              dealt[0][e][2]) for e in EMOTIONS}
    return result


def format_report(result: dict, p: dict[str, dict[str, float]]) -> str:
    fmt = lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"                 # noqa: E731
    lines = ["| condition | " + " | ".join(f"{e}: answered right · ranked right (of {result['none'][e][2]})"
                                           for e in EMOTIONS) + " |", "|---|" + "---|" * len(EMOTIONS)]
    for condition in CONDITIONS:
        lines.append(f"| {condition} | " + " | ".join(
            f"{fmt(result[condition][e][0])} · {fmt(result[condition][e][1])}" for e in EMOTIONS) + " |")
    none = np.array([v["none"] for v in p.values()])
    lines += ["", f"with no tag the LLM says Yes to {int((none > 0.5).sum())} of {len(none)} clips "
                  f"(median p_yes {np.median(none):.2f}); it puts a median {np.median([v['mass'] for v in p.values()]):.2f} "
                  "of its probability on Yes or No",
              "median p_yes by the word shown: " + ", ".join(
                  f"{WORDS[label]} {np.median([v[WORDS[label]] for v in p.values()]):.2f}" for label in LABELS)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# PR-005's criteria, the per-pair listing, and the evidence file.
# Reporting only: nothing above this line changes after the registration.
# --------------------------------------------------------------------------- #

THRESHOLDS = {"FEA": 8, "ANG": 7, "SAD": 7}   # PR-005 S1, S2, S3: pairs answered right, of 13
P_MAX = 0.05
DECIDING = ("FEA", "ANG")                     # S3 (sadness) is reported, expected to fail, and does not decide


def deal_counts(pairs, p: dict[str, dict[str, float]], tags: dict[str, str],
                n_shuffles: int = N_SHUFFLES, seed: int = SEED) -> dict[str, list[int]]:
    """{emotion: pairs answered right under each shuffled deal} -- the SAME deals
    evaluate() averages (same seed, same draw order), kept one by one so a
    permutation p can be read off them. A test holds that their mean equals
    evaluate()'s shuffled mean exactly."""
    clips = [c for pr in pairs for c in (pr.neutral, pr.loaded)]
    rng = random.Random(seed)
    counts = {e: [] for e in EMOTIONS}
    for _ in range(n_shuffles):
        words = [tags[c] for c in clips]
        rng.shuffle(words)
        deal = dict(zip(clips, words))
        for emotion in EMOTIONS:
            counts[emotion].append(sum(_pair_right(p[pr.neutral][deal[pr.neutral]], p[pr.loaded][deal[pr.loaded]])[0]
                                       for pr in pairs if pr.emotion == emotion))
    return counts


def permutation_p(count: float, null: list[int]) -> float:
    """The share of shuffled deals that answered at least `count` pairs right."""
    return sum(c >= count for c in null) / len(null)


def format_criteria(result: dict, null: dict[str, list[int]], confirm: str = "") -> str:
    """PR-005's table: per emotion, the tagger's count, the deals that matched or
    beat it, the p, and PASS / FAIL against the registered bar. The verdict line
    is printed only for a registered run."""
    fmt_p = lambda v: f"p = {v:.3f}" if v >= 0.001 else "p < 0.001"           # noqa: E731
    lines = ["| criterion | pairs answered right | shuffled deals at or above it | permutation p | registered bar | |",
             "|---|---:|---:|---:|---|---|"]
    verdicts = {}
    for name, emotion in (("S1 fear", "FEA"), ("S2 anger", "ANG"), ("S3 sadness", "SAD")):
        got, _, n = result["tagger"][emotion]
        pv = permutation_p(got, null[emotion])
        ok = got >= THRESHOLDS[emotion] and pv < P_MAX
        verdicts[emotion] = ok
        lines.append(f"| {name} | {got:.0f} of {n} | {sum(c >= got for c in null[emotion])} of {len(null[emotion])} | "
                     f"{fmt_p(pv)} | at least {THRESHOLDS[emotion]} and p < {P_MAX} | "
                     f"{'PASS' if ok else 'FAIL'}{'' if emotion in DECIDING else ' (reported, not deciding)'} |")
    if confirm:
        failed = [e for e in DECIDING if not verdicts[e]]
        lines += ["", f"{confirm}: " + ("SUPPORTED: fear and anger both hold" if not failed
                                        else "NOT SUPPORTED: failed " + ", ".join(WORDS[e] for e in failed))]
    return "\n".join(lines)


def per_pair(pairs, p: dict[str, dict[str, float]], tags: dict[str, str], transcripts: dict[str, str]) -> str:
    """Every pair: the two tags, the two p(Yes) under those tags, the outcome, and
    whether the recognizer heard the same words in both clips."""
    lines = ["| pair | neutral clip tagged | p(Yes) | loaded clip tagged | p(Yes) | answered | ranked | same transcript |",
             "|---|---|---:|---|---:|---|---|---|"]
    apart = {e: 0 for e in EMOTIONS}
    for pr in pairs:
        tn, tl = tags[pr.neutral], tags[pr.loaded]
        pn, pl = p[pr.neutral][tn], p[pr.loaded][tl]
        answered, ranked = _pair_right(pn, pl)
        apart[pr.emotion] += tn == "neutral" != tl
        lines.append(f"| {pr.id} | {tn} | {pn:.2f} | {tl} | {pl:.2f} | {'right' if answered else '**wrong**'} | "
                     f"{'right' if ranked else 'wrong'} | {'yes' if transcripts[pr.neutral] == transcripts[pr.loaded] else 'no'} |")
    lines += ["", "the tagger alone told apart (neutral clip tagged neutral, loaded clip tagged anything else): "
              + ", ".join(f"{e} {apart[e]} of {sum(pr.emotion == e for pr in pairs)}" for e in EMOTIONS)]
    return "\n".join(lines)


def refuse_to_overwrite(path: pathlib.Path) -> None:
    if pathlib.Path(path).exists():
        raise FileExistsError(f"{path} exists. A saved run is evidence and is never overwritten: choose another name.")


def save(path: pathlib.Path, head: str, pairs, p: dict, tags: dict, transcripts: dict, result: dict,
         null: dict[str, list[int]]) -> None:
    """Everything a reader needs to re-count PR-005 without the clips or the models."""
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run": head, "question": QUESTION, "system": PROBE_SYSTEM, "seed": SEED, "n_shuffles": N_SHUFFLES,
        "pairs": [pr._asdict() for pr in pairs], "transcripts": transcripts, "tags": tags, "p_yes": p,
        "result": {c: {e: list(v) for e, v in r.items()} for c, r in result.items()},
        "shuffled_deal_counts": null,
    }, indent=1) + "\n", encoding="utf-8")


def run(split: str, confirm: str = "", recognizer=None, scorer=None, save_to: pathlib.Path | None = None) -> str:
    """Tag, transcribe and probe one side of the set; returns the report.

    Takes:   "dev", or "test" with the registration it confirms (prosodyset.load
             enforces the lock); a loaded recognizer and a scorer function, for
             tests -- by default R1's own Whisper base.en and Qwen2.5-0.5B; and
             where to keep the run, checked BEFORE any model runs.
    """
    if save_to is not None:
        refuse_to_overwrite(save_to)
    loaded: list[LoadedPair] = load(split, confirm=confirm)
    pairs = [lp.pair for lp in loaded]
    audio = {c: a for lp in loaded for c, a in ((lp.pair.neutral, lp.neutral), (lp.pair.loaded, lp.loaded))}
    tagger = Tagger.load(TAGGER)
    X = features_for(list(audio), PROSODY_ROOT / "audio", CACHE / f"{split}.npz")
    tags = {clip: tagger.tag(x).word for clip, x in zip(audio, X)}
    if recognizer is None:
        from .rungs import get
        recognizer = get("r1").asr
        recognizer.load()
    transcripts = {clip: recognizer.transcribe(a) for clip, a in audio.items()}
    if scorer is None:
        scorer = YesNoScorer().load().p_yes
    p = score_all(transcripts, scorer, CACHE / f"pyes-{split}.json")
    head = f"{len(pairs)} {split} pairs" + (f" · confirms {confirm}" if confirm else "") + \
           f' · "{QUESTION}" · shuffled = mean of {N_SHUFFLES} deals, seed {SEED}'
    result, null = evaluate(pairs, p, tags), deal_counts(pairs, p, tags)
    criteria = format_criteria(result, null, confirm)
    if split == "dev":
        criteria += "\n(the registered bars are for 13 test pairs per emotion; on the dev pairs only the p means anything)"
    text = [head, format_report(result, p), criteria, per_pair(pairs, p, tags, transcripts)]
    if save_to is not None:
        save(save_to, head, pairs, p, tags, transcripts, result, null)
        text.append(f"saved to {save_to}")
    return "\n\n".join(text)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="The affect side-channel on the prosody probe set.")
    ap.add_argument("split", choices=("dev", "test"))
    ap.add_argument("--confirm", default="", help="the registration a test run confirms, e.g. PR-005")
    ap.add_argument("--save", default="", help="keep every transcript, tag, p(Yes) and outcome as JSON; never overwrites")
    a = ap.parse_args()
    print(run(a.split, a.confirm, save_to=pathlib.Path(a.save) if a.save else None))