"""
R3's probe: a speech-LLM that HEARS the clip is asked the side-channel's question.

The v2 plan, R3: "Ask something answerable only from prosody ... R1 physically
cannot answer -- the information was destroyed at the ASR boundary. R3 can.
That is the input bottleneck demonstrated in one prompt." Three points on one
axis, all on the same 39 pairs and the same question:

    R1 blind        transcript only                   0 · 0 · 1 of 13     (PR-005, "none")
    R1 + tags       transcript + the tagger's word    7 · 7 · 11          (PR-005, "tagger")
    R3              THE AUDIO: no transcript, no tag  this file

The model (Qwen2-Audio-7B-Instruct, 16 GB, an A10 on Modal) never sees a
transcript. It is given the clip and the question, and its answer is read off
the next-token probabilities of "Yes" against "No" -- exactly as
sidechannel.YesNoScorer reads R1's LLM, so the numbers sit on one scale. A
pair is answered right when the loaded clip gets Yes and its neutral twin gets
No (sidechannel._pair_right). Per emotion, never pooled.

WHAT RUNS WHERE. This module is the laptop side: which clips, which wordings,
the scoring and the report; it imports nothing from Modal or torch. The model
runs in modal_app/r3_probe.py, which takes the clips as WAV bytes and returns
p(Yes) per clip per wording, and the model's own transcript of each clip (a
sanity check that it heard the words; descriptive).

THE WORDING IS CHOSEN ON THE DEV PAIRS, and only there. There is no oracle for
R3, so the choice is by the dev score directly; WORDINGS holds every one tried,
and the registration (PR-006) fixes one before the test pairs are sent anywhere.

THE DEV RUN (2026-09-22, A10, $0.04): R3 ranked all 21 pairs right under every
wording -- the loaded clip's p(Yes) was about 1.00 every time -- and transcribed
every clip correctly. It leans Yes, so with the 0.5 threshold PR-005 uses:

    tone       4 · 4 · 4 of 7 answered right (SAD · FEA · ANG)
    upset      6 · 5 · 5                    <- registered
    not_words  0 · 0 · 0   Yes to all 42 clips
    emotion    0 · 0 · 0   mass 0.00: it wants to name the emotion, not say Yes or No

Its limit is calibration, not hearing. The metric stays the one PR-005 used,
so the two are compared on the same scale.

    modal run modal_app/r3_probe.py --split dev                          the 21 dev pairs, every wording
    modal run modal_app/r3_probe.py --split test --confirm PR-006 --wording upset --save runs/prosody/pr006.json
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .prosodyset import EMOTIONS, PROSODY_ROOT, load, verify
from .sidechannel import _pair_right, refuse_to_overwrite

SYSTEM = "Answer with one word: Yes or No."
WORDINGS = {
    "tone": "Judging by their tone of voice, is the caller upset?",           # the side-channel's own question
    "upset": "Is the person speaking upset?",
    "not_words": "Listen to how this was said, not to what was said. Does the speaker sound upset?",
    "emotion": "Does the speaker sound sad, fearful or angry, rather than calm?",
}
TRANSCRIBE = "Transcribe the speech in this clip, word for word."
REGISTERED_WORDING = "upset"    # chosen on the dev pairs, 2026-09-22 (see the docstring); PR-006 asks this one


def clips_for(split: str, confirm: str = "", root: pathlib.Path = PROSODY_ROOT):
    """The pairs of one side and their clips as WAV bytes, ready to send.

    Takes:   "dev", or "test" with the registration it confirms -- prosodyset.load
             enforces the lock, so the test clips cannot be sent without one.
    Calls:   prosodyset.verify() first: a changed or missing clip stops the run
             before any GPU time is spent.
    Returns: (pairs, {clip name: bytes of its .wav})
    """
    problems = verify(root)
    if problems:
        raise RuntimeError("the prosody set is not the frozen one:\n  " + "\n  ".join(problems[:10]))
    pairs = [lp.pair for lp in load(split, root, confirm)]
    clips = {c: (pathlib.Path(root) / "audio" / f"{c}.wav").read_bytes()
             for pr in pairs for c in (pr.neutral, pr.loaded)}
    return pairs, clips


def score_pairs(pairs, p_yes: dict[str, float]) -> dict[str, tuple[int, int, int]]:
    """{emotion: (pairs answered right, pairs ranked right, pairs)} from {clip: p(Yes)}."""
    out = {}
    for emotion in EMOTIONS:
        mine = [pr for pr in pairs if pr.emotion == emotion]
        got = [_pair_right(p_yes[pr.neutral], p_yes[pr.loaded]) for pr in mine]
        out[emotion] = (sum(a for a, _ in got), sum(r for _, r in got), len(mine))
    return out


def format_report(pairs, answers: dict[str, dict], wordings: dict[str, str]) -> str:
    """Takes: the pairs; the container's answers {clip: {"p_yes": {wording: p},
    "mass": {wording: m}, "transcript": str}}; the wordings asked.
    Returns: one row per wording (answered right · ranked right per emotion),
    how the model leans, and the per-pair table for the FIRST wording."""
    names = list(wordings)
    lines = ["| wording | " + " | ".join(f"{e}: answered right · ranked right (of {sum(pr.emotion == e for pr in pairs)})"
                                         for e in EMOTIONS) + " |", "|---|" + "---|" * len(EMOTIONS)]
    for name in names:
        p = {c: a["p_yes"][name] for c, a in answers.items()}
        s = score_pairs(pairs, p)
        lines.append(f"| {name} | " + " | ".join(f"{s[e][0]} · {s[e][1]}" for e in EMOTIONS) + " |")
    lines.append("")
    for name in names:
        ps = np.array([a["p_yes"][name] for a in answers.values()])
        mass = np.median([a["mass"][name] for a in answers.values()])
        lines.append(f'{name}: "{wordings[name]}" -- Yes to {int((ps > 0.5).sum())} of {len(ps)} clips, '
                     f"median p(Yes) {np.median(ps):.2f}, median mass on Yes-or-No {mass:.2f}")
    first = names[0]
    lines += ["", f"per pair, wording '{first}':",
              "| pair | p(Yes) neutral | p(Yes) loaded | answered | ranked | the model heard (neutral clip) |",
              "|---|---:|---:|---|---|---|"]
    for pr in pairs:
        pn, pl = answers[pr.neutral]["p_yes"][first], answers[pr.loaded]["p_yes"][first]
        a, r = _pair_right(pn, pl)
        heard = answers[pr.neutral].get("transcript", "")
        lines.append(f"| {pr.id} | {pn:.2f} | {pl:.2f} | {'right' if a else '**wrong**'} | "
                     f"{'right' if r else 'wrong'} | {heard[:60]!r} |")
    return "\n".join(lines)


def save(path: pathlib.Path, head: str, pairs, answers: dict, wordings: dict[str, str]) -> None:
    """Everything needed to re-count: the pairs, every p(Yes) and mass under every
    wording, and the model's transcripts. Never overwrites."""
    path = pathlib.Path(path)
    refuse_to_overwrite(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": head, "system": SYSTEM, "wordings": wordings,
                                "pairs": [pr._asdict() for pr in pairs], "answers": answers,
                                "scores": {name: {e: list(v) for e, v in
                                                  score_pairs(pairs, {c: a["p_yes"][name] for c, a in answers.items()}).items()}
                                           for name in wordings}}, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# PR-006's criteria: R3 against the side-channel, pair by pair.
# Reporting only, added after the registration; nothing above changes.
# --------------------------------------------------------------------------- #

RANKED_MIN = 12               # Claim A: ranked right, of 13, per emotion
P_MAX = 0.05                  # Claim B2


def sidechannel_outcomes(run: dict) -> dict[str, bool]:
    """{pair id: answered right by the side-channel} from a saved PR-005 run
    (runs/prosody/pr005.json): the tagger's word for each clip, and the LLM's
    p(Yes) under that word, scored exactly as evaluate() scored them."""
    return {pr["id"]: _pair_right(run["p_yes"][pr["neutral"]][run["tags"][pr["neutral"]]],
                                  run["p_yes"][pr["loaded"]][run["tags"][pr["loaded"]]])[0]
            for pr in run["pairs"]}


def compare(pairs, p_yes: dict[str, float], sidechannel: dict[str, bool]) -> dict:
    """Everything PR-006 asks, as numbers.

    Takes:   the pairs; R3's {clip: p(Yes)} under the registered wording; the
             side-channel's answered-right outcome per pair id.
    Returns: {"r3": score_pairs(...), "tag": {emotion: answered right},
              "fixed": [ids R3 got right and the tag wrong], "broken": [the reverse],
              "p": the exact two-sided sign test on fixed and broken,
              "A": {emotion: bool}, "B1": {emotion: bool}, "B2": bool}
    """
    from .bargein_eval import sign_test
    r3 = score_pairs(pairs, p_yes)
    right = {pr.id: _pair_right(p_yes[pr.neutral], p_yes[pr.loaded])[0] for pr in pairs}
    tag = {e: sum(sidechannel[pr.id] for pr in pairs if pr.emotion == e) for e in EMOTIONS}
    fixed = sorted(pr.id for pr in pairs if right[pr.id] and not sidechannel[pr.id])
    broken = sorted(pr.id for pr in pairs if sidechannel[pr.id] and not right[pr.id])
    p = sign_test(len(fixed), len(broken))
    return {"r3": r3, "tag": tag, "fixed": fixed, "broken": broken, "p": p,
            "A": {e: r3[e][1] >= RANKED_MIN for e in EMOTIONS},
            "B1": {e: r3[e][0] >= tag[e] for e in EMOTIONS},
            "B2": sum(r3[e][0] for e in EMOTIONS) > sum(tag.values()) and p < P_MAX}


def format_criteria(result: dict, confirm: str = "") -> str:
    """The two claims as tables, and their verdicts (only for a registered run)."""
    r3, tag = result["r3"], result["tag"]
    name = {"SAD": "sad", "FEA": "fear", "ANG": "anger"}
    lines = ["Claim A -- the input bottleneck: R3 ranks the upset clip above its calm twin",
             "| criterion | ranked right | bar | |", "|---|---:|---|---|"]
    for k, e in enumerate(EMOTIONS, 1):
        lines.append(f"| A{k} {name[e]} | {r3[e][1]} of {r3[e][2]} | at least {RANKED_MIN} | "
                     f"{'PASS' if result['A'][e] else 'FAIL'} |")
    lines += ["", "Claim B -- native beats the proxy: answered right, R3 against the side-channel",
              "| criterion | R3 | side-channel | bar | |", "|---|---:|---:|---|---|"]
    for e in EMOTIONS:
        lines.append(f"| B1 {name[e]} | {r3[e][0]} of {r3[e][2]} | {tag[e]} of {r3[e][2]} | at least {tag[e]} | "
                     f"{'PASS' if result['B1'][e] else 'FAIL'} |")
    total_r3, total_tag = sum(r3[e][0] for e in EMOTIONS), sum(tag.values())
    fmt_p = f"p = {result['p']:.3f}" if result["p"] >= 0.001 else "p < 0.001"
    lines.append(f"| B2 overall | {total_r3} of 39 | {total_tag} of 39 | more, and sign test p < {P_MAX} | "
                 f"{'PASS' if result['B2'] else 'FAIL'} |")
    lines.append(f"  fixed {len(result['fixed'])}: {', '.join(result['fixed']) or '-'}")
    lines.append(f"  broken {len(result['broken'])}: {', '.join(result['broken']) or '-'}")
    lines.append(f"  exact two-sided sign test on the {len(result['fixed']) + len(result['broken'])} pairs that differ: {fmt_p}")
    if confirm:
        a_ok, b_ok = all(result["A"].values()), all(result["B1"].values()) and result["B2"]
        lines += ["", f"{confirm} Claim A: " + ("SUPPORTED: R3 ranks at least 12 of 13 right for every emotion" if a_ok
                                                else "NOT SUPPORTED: failed " + ", ".join(name[e] for e in EMOTIONS if not result["A"][e])),
                  f"{confirm} Claim B: " + ("SUPPORTED: at least the side-channel's count for every emotion, and more overall" if b_ok
                                            else "NOT SUPPORTED: failed " + ", ".join(
                                                [f"B1 {name[e]}" for e in EMOTIONS if not result["B1"][e]] + ([] if result["B2"] else ["B2"])))]
    return "\n".join(lines)


def per_pair(pairs, p_yes: dict[str, float], sidechannel: dict[str, bool], transcripts: dict[str, str]) -> str:
    """Every pair: R3's two p(Yes), its outcome, the side-channel's outcome, and
    whether R3's transcript of the neutral clip has the sentence's words."""
    from .prosodycheck import SENTENCES
    from .wer import light
    lines = ["| pair | p(Yes) neutral | p(Yes) loaded | R3 answered | R3 ranked | side-channel | R3 heard the sentence |",
             "|---|---:|---:|---|---|---|---|"]
    heard_right = 0
    for pr in pairs:
        pn, pl = p_yes[pr.neutral], p_yes[pr.loaded]
        a, r = _pair_right(pn, pl)
        heard = light(SENTENCES[pr.sentence]) in light(transcripts.get(pr.neutral, ""))
        heard_right += heard
        lines.append(f"| {pr.id} | {pn:.2f} | {pl:.2f} | {'right' if a else '**wrong**'} | {'right' if r else 'wrong'} | "
                     f"{'right' if sidechannel[pr.id] else 'wrong'} | {'yes' if heard else 'no'} |")
    lines += ["", f"R3's transcript of the neutral clip contains the sentence, word for word after normalizing: "
              f"{heard_right} of {len(pairs)}"]
    return "\n".join(lines)
