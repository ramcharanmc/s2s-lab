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
