"""
PR-014's VAP side: the prepared turns exported for VAP's own environment, its probabilities read back, TV
run beside the other policies, θ chosen on DEV, and the one confirmation.

    PYTHONPATH=src python -m s2slab.vapturns export dev                  results/pr014/dev/: WAVs and turns.json
    ~/.venvs/vap/bin/python src/s2slab/vaprun.py results/pr014/dev       VAP over them -> vap.json
    ~/.venvs/vap/bin/python src/s2slab/vaprun.py results/pr014/dev --causality
    PYTHONPATH=src python -m s2slab.vapturns tune                         θ on DEV, by RULE -> runs/turn-taking/pr014-dev-vap.json

The confirmation, once, in this order (PR-014's analysis plan, step 3):

    PYTHONPATH=src python -m s2slab.vapturns export eval --confirm PR-014
    ~/.venvs/vap/bin/python src/s2slab/vaprun.py results/pr014/eval
    PYTHONPATH=src HF_HUB_OFFLINE=1 python -m s2slab.vapturns confirm --confirm PR-014

WHY FILES BETWEEN TWO ENVIRONMENTS. VAP's code is pinned to torch 2.4 (its loader calls torch.load on a
checkpoint holding objects, which torch 2.6 and later refuse by default); s2slab runs torch 2.14. So
s2slab exports each turn exactly as the policies see it (cropped, ended, padded, as float32 WAVs with
their SHA-256), VAP's environment writes one probability a 20 ms frame (s2slab/vaprun.py, standalone), and
s2slab reads them back into each turn's context, where TV reads them. The other policies see the same
audio, words and ends as before: the WAV round trip is exact.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

OUT = pathlib.Path("results/pr014")
REGISTRATION = "PR-014"
# PR-014, Deviation 1: VAP is told there are two parties. Before each caller's turn, the system says
# this on VAP's second channel, then is silent; rendered once by Kokoro (R0's voice) and pinned, because
# Kokoro under torch does not repeat itself exactly. The file stays at PROMPT; it is not in git (audio).
PROMPT = OUT / "prompt.wav"
PROMPT_TEXT, PROMPT_VOICE, GAP_S = "How can I help you today?", "af_heart", 0.3
PROMPT_SHA256 = "e78e6b589f44f9eee630b84e9b16cc9958d1e5f24ebd5ddd2d6bc2d9bd546296"
VAP_COMMIT = "f39a78b"                          # ErikEkstedt/VoiceActivityProjection, the clone in ~/vap
STATE_DICT_SHA256 = "043f2800bb95a7c27ad4271b1034fe2c1b70766f20e3b4ff91fe76ee9dc2360e"       # example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt
GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MAX_MEDIAN_WAIT_MS = 1000.0
RULE = ("the value in GRID with the fewest DEV cut-offs among those whose median wait on correctly "
        "ended DEV turns is at most 1,000 ms; a tie goes to the smaller value (PR-014)")
TV_THETA: float | None = None                   # no value in GRID met RULE on DEV, with or without the prompt:
                                                # PR-014 was not run (docs/preregistration.md, its Results)
DEV_VAP = pathlib.Path("runs/turn-taking/pr014-dev-vap.json")                    # with the prompt
DEV_VAP_SILENT = pathlib.Path("runs/turn-taking/pr014-dev-vap-silent.json")      # the first attempt, a silent partner
RUN = pathlib.Path("runs/turn-taking/pr014.json")
VAP_TRACKS = pathlib.Path("runs/turn-taking/pr014-vap.json")
T1B = "T1b_complete_500ms"


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def prompt_file(path: pathlib.Path = PROMPT, render: bool = True) -> pathlib.Path:
    """The system's prompt at 16 kHz: rendered by Kokoro the first time, then always this file, checked
    against PROMPT_SHA256 once that is pinned."""
    import soundfile as sf
    path = pathlib.Path(path)
    if not path.exists():
        if not render:
            raise FileNotFoundError(f"{path} is missing: PR-014's prompt is the pinned file, not a new render")
        from scipy.signal import resample_poly

        from .r0 import KokoroTTS
        tts = KokoroTTS(voice=PROMPT_VOICE)
        tts.load()
        x = resample_poly(tts.speak(PROMPT_TEXT), 2, 3).astype("float32")      # 24 kHz -> 16 kHz
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), x, 16_000, subtype="FLOAT")
    if PROMPT_SHA256 != "PROMPT_SHA256" and _sha256(path) != PROMPT_SHA256:
        raise RuntimeError(f"{path} is not PR-014's prompt (its SHA-256 differs)")
    return path


def export(items: list, out_dir: pathlib.Path, registration: str = "", prompt: pathlib.Path | None = None) -> pathlib.Path:
    """Each (id, Utterance) as a float32 WAV, and turns.json with what the policies read besides audio, and
    the prompt VAP hears first (None: a silent partner)."""
    import soundfile as sf
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: an export is written once")
    out_dir.mkdir(parents=True)
    pr = None
    if prompt is not None:
        import os
        pr = {"file": os.path.relpath(pathlib.Path(prompt).resolve(), out_dir.resolve()), "sha256": _sha256(prompt),
              "gap_s": GAP_S, "text": PROMPT_TEXT, "voice": PROMPT_VOICE}
    turns = {}
    for tid, u in items:
        wav = out_dir / f"{tid}.wav"
        sf.write(str(wav), u.audio, u.sr, subtype="FLOAT")
        turns[tid] = {"cohort": u.cohort, "true_end_s": u.true_end_s, "words": [list(w) for w in u.words],
                      "mid_turn_pauses_s": u.mid_turn_pauses_s, "n_syllables": u.n_syllables,
                      "wav_sha256": _sha256(wav)}
    meta = {"registration": registration or None, "prompt": pr, "turns": turns}
    (out_dir / "turns.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return out_dir


def load(out_dir: pathlib.Path, vap: bool = True) -> list:
    """The exported turns as (id, Utterance), each WAV checked; with VAP's track in each context."""
    import soundfile as sf

    from .probes import _pitch_track
    from .simulate import Utterance
    out_dir = pathlib.Path(out_dir)
    meta = json.loads((out_dir / "turns.json").read_text(encoding="utf-8"))
    tracks = None
    if vap:
        v = json.loads((out_dir / "vap.json").read_text(encoding="utf-8"))
        if v["vap_commit"] != VAP_COMMIT or v["state_dict_sha256"] != STATE_DICT_SHA256:
            raise RuntimeError(f"vap.json came from VAP {v['vap_commit']}, state dict {v['state_dict_sha256'][:12]}; "
                               f"PR-014 pins {VAP_COMMIT} and {STATE_DICT_SHA256[:12]}")
        if v.get("prompt_sha256") != (meta.get("prompt") or {}).get("sha256"):
            raise RuntimeError("vap.json was run with another prompt than this export names")
        tracks = v["p_system"]
    out = []
    for tid, t in meta["turns"].items():
        wav = out_dir / f"{tid}.wav"
        if _sha256(wav) != t["wav_sha256"]:
            raise RuntimeError(f"{wav} is not the file that was exported")
        x, sr = sf.read(str(wav), dtype="float32")
        u = Utterance(audio=x, sr=sr, true_end_s=t["true_end_s"], cohort=t["cohort"],
                      words=[(w, e) for w, e in t["words"]], n_syllables=t["n_syllables"],
                      mid_turn_pauses_s=t["mid_turn_pauses_s"], f0_track=_pitch_track(x))
        if tracks is not None:
            u.context().vap_p = tracks[tid]
        out.append((tid, u))
    return out


def tv(theta: float | None = None):
    """TV at the frozen θ (or the one given)."""
    from .policies import ProjectionThreshold
    theta = TV_THETA if theta is None else theta
    if theta is None:
        raise RuntimeError("TV's θ is not chosen yet: run `python -m s2slab.vapturns tune`")
    return ProjectionThreshold(theta)


def fallbacks(policy, utterances) -> int:
    """How many turns this TV decided by its 1,500 ms fallback rather than by VAP."""
    return sum(1 for u in utterances if policy.decide(u.context()).reason.endswith("+fallback"))


def tune(out_dir: pathlib.Path = OUT / "dev") -> dict:
    """TV at every θ in GRID on the DEV turns, and the θ RULE picks."""
    from .sweep import evaluate
    items = load(out_dir)
    us = [u for _, u in items]
    rows = []
    for th in GRID:
        p = tv(th)
        r = evaluate(p, us)
        rows.append({"theta": th, "n": r.n, "cut": r.n_cutoff, "no_fire": r.n_never_fired,
                     "p50": r.latency.p50, "p90": r.latency.p90, "fallback": fallbacks(p, us),
                     "cut_items": [tid for (tid, _), c in zip(items, r.cut_mask) if c]})
    ok = [r for r in rows if r["p50"] == r["p50"] and r["p50"] <= MAX_MEDIAN_WAIT_MS]
    chosen = min(ok, key=lambda r: (r["cut"], r["theta"]))["theta"] if ok else None
    return {"rule": RULE, "grid": list(GRID), "n": len(us), "rows": rows, "chosen": chosen,
            "vap_commit": VAP_COMMIT, "state_dict_sha256": STATE_DICT_SHA256}


def format_tune(blob: dict) -> str:
    lines = ["| θ | cut off | median wait ms | p90 ms | by the fallback | items cut off |", "|---:|---:|---:|---:|---:|---|"]
    for r in blob["rows"]:
        p50 = "—" if r["p50"] != r["p50"] else f"{r['p50']:.0f}"
        p90 = "—" if r["p90"] != r["p90"] else f"{r['p90']:.0f}"
        lines.append(f"| {r['theta']:g} | {r['cut']} of {r['n']} | {p50} | {p90} | {r['fallback']} | {', '.join(r['cut_items'])} |")
    lines += ["", f"rule: {blob['rule']}", f"chosen: {blob['chosen']}"]
    return "\n".join(lines)


def pr014_criteria(results: list, comps: list, tv_name: str) -> dict:
    """PR-014's S1 to S4, from T1b and TV on the four cohorts and their comparisons (A = T1b, B = TV)."""
    by = {(r.policy, r.cohort): r for r in results}
    dis = next(c for c in comps if c.cohort == "disfluent")
    a, b = by[(T1B, "clean")], by[(tv_name, "clean")]
    d50, d90 = b.latency.p50 - a.latency.p50, b.latency.p90 - a.latency.p90
    s = {"S1": by[(tv_name, "disfluent")].n_cutoff < by[(T1B, "disfluent")].n_cutoff and dis.ci[0] > 0,
         "S2": dis.p_adjusted < 0.05,
         "S3": d50 <= 100 and d90 <= 300,
         "S4": b.n_cutoff <= a.n_cutoff + 2}
    return {"disfluent_cut_t1b": by[(T1B, "disfluent")].n_cutoff, "disfluent_cut_tv": by[(tv_name, "disfluent")].n_cutoff,
            "disfluent_diff": dis.diff, "disfluent_ci": list(dis.ci), "disfluent_p_holm": dis.p_adjusted,
            "clean_p50_delta_ms": d50, "clean_p90_delta_ms": d90,
            "clean_cut_t1b": a.n_cutoff, "clean_cut_tv": b.n_cutoff,
            **{k: bool(v) for k, v in s.items()}, "supported": all(s.values())}


def format_pr014(crit: dict) -> str:
    lo, hi = crit["disfluent_ci"]
    rows = [("S1 · effect", f"disfluent cut-offs {crit['disfluent_cut_t1b']} (T1b) against {crit['disfluent_cut_tv']} (TV); "
                            f"T1b minus TV {crit['disfluent_diff'] * 100:+.1f} pp, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]; needs fewer, CI above 0", crit["S1"]),
            ("S2 · confidence", f"Holm p {crit['disfluent_p_holm']:.3g}; needs < 0.05", crit["S2"]),
            ("S3 · speed", f"clean median {crit['clean_p50_delta_ms']:+.0f} ms, p90 {crit['clean_p90_delta_ms']:+.0f} ms; needs <= +100 and <= +300", crit["S3"]),
            ("S4 · safety", f"clean cut-offs {crit['clean_cut_t1b']} (T1b) against {crit['clean_cut_tv']} (TV); needs at most 2 more", crit["S4"])]
    out = ["### PR-014's criteria", "", "| criterion | measured | met |", "|---|---|---|"]
    out += [f"| {k} | {m} | {'yes' if ok else 'no'} |" for k, m, ok in rows]
    out += ["", f"**PR-014: {'supported' if crit['supported'] else 'not supported'}**"]
    return "\n".join(out)


def confirm(seed: int = 0) -> dict:
    """The one confirmation: T0 at 900 ms, T1b and TV on the exported eval turns; S1 to S4."""
    from .completion import t1b
    from .policies import get
    from .sweep import _git_commit, evaluate, format_cohort, format_comparisons, run_comparisons
    meta = json.loads((OUT / "eval" / "turns.json").read_text(encoding="utf-8"))
    if meta.get("registration") != REGISTRATION:
        raise SystemExit(f"{OUT / 'eval'} was not exported for {REGISTRATION}")
    if (meta.get("prompt") or {}).get("sha256") != PROMPT_SHA256:
        raise SystemExit(f"{OUT / 'eval'} was not exported with PR-014's prompt")
    items = load(OUT / "eval")
    commit = _git_commit()
    cohorts: dict[str, list] = {}
    for _, u in items:
        cohorts.setdefault(u.cohort, []).append(u)
    pol_tv = tv()
    pols = [get("T0_fixed_900ms"), t1b(), pol_tv]
    results = [evaluate(p, us) for c, us in sorted(cohorts.items()) for p in pols]
    comps = run_comparisons(results, [(T1B, pol_tv.name)], seed=seed)
    crit = pr014_criteria(results, comps, pol_tv.name)
    fb = {c: fallbacks(pol_tv, us) for c, us in sorted(cohorts.items())}
    for c in sorted(cohorts):
        print(format_cohort(results, c) + "\n")
    print(format_comparisons(comps) + "\n")
    print(format_pr014(crit))
    print(f"\nTV: θ {TV_THETA} (frozen), VAP {VAP_COMMIT}; decided by the 1,500 ms fallback: "
          + ", ".join(f"{c} {n}" for c, n in fb.items()))
    blob = {"commit": commit, "registration": REGISTRATION, "theta": TV_THETA, "vap_commit": VAP_COMMIT,
            "state_dict_sha256": STATE_DICT_SHA256, "seed": seed,
            "results": [r.as_dict() for r in results],
            "comparisons": [{"cohort": c.cohort, "a": c.policy_a, "b": c.policy_b, "n": c.n, "rate_a": c.rate_a,
                             "rate_b": c.rate_b, "only_a": c.test.only_a, "only_b": c.test.only_b, "p": c.test.p,
                             "p_holm": c.p_adjusted, "diff": c.diff, "ci": list(c.ci), "verdict": c.verdict()}
                            for c in comps],
            "criteria": crit, "fallbacks": fb,
            "cut_items": {f"{r.policy}|{r.cohort}": [tid for (tid, u), cut in
                                                      zip([it for it in items if it[1].cohort == r.cohort], r.cut_mask) if cut]
                          for r in results}}
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    VAP_TRACKS.write_text((OUT / "eval" / "vap.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\ncommit {commit} -> {RUN}, VAP's probabilities -> {VAP_TRACKS}")
    return blob


def main(argv=None) -> None:
    import argparse
    from .realturns import dev_items, eval_items
    from .sweep import _git_commit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["export", "tune", "confirm"])
    ap.add_argument("which", nargs="?", choices=["dev", "dev-silent", "eval"])
    ap.add_argument("--confirm", default="")
    ap.add_argument("--code", default="", help="the code the run used, when it is not a commit yet")
    a = ap.parse_args(argv)
    if a.what == "export":
        if a.which == "eval":
            if a.confirm != REGISTRATION:
                ap.error(f"the eval turns are {REGISTRATION}'s confirmation's: add --confirm {REGISTRATION}")
            if RUN.exists():
                ap.error(f"{RUN} exists: {REGISTRATION}'s confirmation runs once")
            if TV_THETA is None:
                ap.error("TV's θ is not frozen yet")
            out = export(eval_items(confirm=a.confirm), OUT / "eval", REGISTRATION, prompt_file(render=False))
        elif a.which == "dev":
            out = export(dev_items(), OUT / "dev", prompt=prompt_file())
        elif a.which == "dev-silent":
            out = export(dev_items(), OUT / "dev-silent")
        else:
            ap.error("export dev, dev-silent or eval")
        print(f"exported -> {out}")
    elif a.what == "tune":
        which = a.which or "dev"
        if which == "eval":
            ap.error("θ is chosen on DEV only")
        path = DEV_VAP if which == "dev" else DEV_VAP_SILENT
        if path.exists():
            ap.error(f"{path} exists: θ is chosen once")
        blob = {"commit": a.code or _git_commit(), "prompt": which == "dev", **tune(OUT / which)}
        print(format_tune(blob))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
        print(f"\nsaved -> {path}")
    else:
        if a.confirm != REGISTRATION:
            ap.error(f"add --confirm {REGISTRATION}")
        if RUN.exists():
            ap.error(f"{RUN} exists: {REGISTRATION}'s confirmation runs once")
        confirm()


if __name__ == "__main__":
    main()
