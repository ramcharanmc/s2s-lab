"""
PR-012 (docs/preregistration.md): PR-009's five criteria, unchanged, on renders made by giving the
severity targets to FastPitch as its own inputs -- on the base LJSpeech voice and on the voice
fine-tuned to five minutes of Hi-Fi TTS speaker 9017, over eight sentences neither has spoken.

    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.pr012 dry
        the run's own code on PR-009's four DEV sentences, no lock -> results/ft3/pr012-dry.json
    PYTHONPATH=ft:src ~/.venvs/s2slab-ft3/bin/python -m src.ft3.pr012 test --confirm PR-012
        the registered run, once -> runs/ft3/pr012.json, renders under results/ft3/pr012/

The registered run checks, BEFORE any model loads: the confirmation, that its evidence does not
exist yet, that the tree is clean, and that the four model files are the registered ones (SHA-256).
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

from s2slab import severity
from src.ft3 import synth, sweep

REGISTRATION = "PR-012"
RUN_PATH = "runs/ft3/pr012.json"
AUDIO_DIR = "results/ft3/pr012"              # the renders and the exact neutral clips (ignored by git)
REPO = pathlib.Path(__file__).resolve().parents[3]
NEMO_CACHE = "~/.cache/torch/NeMo/NeMo_3.0.0"  # where from_pretrained put the two NGC checkpoints
# voice -> model -> (the file: a path from the repo's root, or a pattern with exactly one match; its SHA-256)
VOICES = {
    "tts_en_fastpitch_align": {        # NGC tts_en_fastpitch 1.8.1 + tts_hifigan 1.0.0rc1: LJSpeech, as downloaded
        "fastpitch": (f"{NEMO_CACHE}/tts_en_fastpitch_align/*/tts_en_fastpitch_align.nemo",
                      "69bfdf02cf8023458622fe6c09c9ff2490273a08515f82d831b70bb80e9240dd"),
        "hifigan": (f"{NEMO_CACHE}/tts_hifigan/*/tts_hifigan.nemo",
                    "e2f05884312672b2bd54064788eeafe29bcd8f88554e56ad68fa46a4a0804684")},
    "fastpitch_9017": {                # FT-3: runs fastpitch-12bc015-20260927-140956 and hifigan-84cf350-20260927-150730
        "fastpitch": ("data/ft3/models/fastpitch_9017.nemo",
                      "afa0731bd238d192a41f704ffcb39e6bda43d83fe6a7f3f7b7bec9dcdf77af60"),
        "hifigan": ("data/ft3/models/hifigan_9017.nemo",
                    "1ebd9d77f105cccb5ff5e592ff5a1b1cdc79f2b70f2068099d863b902d5a0d4b")},
}
SENTENCES = {   # neutral content, a sentence break and a comma each, as PR-009's; written with the registration
    "u1": "The kettle was already on. I made some tea, and then I opened the mail.",
    "u2": "We drove to the lake on Sunday. The water was cold, so we stayed on the shore.",
    "u3": "My brother fixed the bicycle. It needed a new chain, and now it runs quietly.",
    "u4": "The shop on the corner closed early. I went to the market, and I found what I needed.",
    "u5": "I put the boxes in the hallway. They were heavy, but the stairs were not far.",
    "u6": "The printer stopped again this afternoon. I changed the paper, and then it worked.",
    "u7": "We had lunch in the garden. It was warm outside, so we stayed until three.",
    "u8": "The keys were on the shelf. I locked the door, and I walked to the station.",
}
SETTINGS = {"centre_mode": "audio", "range_gain": 1.2, "min_pause_s": 0.15}     # the dev decisions (control.py), frozen


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_files(voices: dict = VOICES, root: pathlib.Path = REPO) -> dict:
    """{voice: {"fastpitch": path, "hifigan": path}}, each file found (exactly one) and its SHA-256
    the registered one. Raises before anything is loaded."""
    out = {}
    for voice, models in voices.items():
        out[voice] = {}
        for kind, (pattern, digest) in models.items():
            pattern = os.path.expanduser(pattern)
            found = sorted(glob.glob(pattern if os.path.isabs(pattern) else str(pathlib.Path(root) / pattern)))
            if len(found) != 1:
                raise FileNotFoundError(f"{voice} {kind}: {len(found)} files match {pattern}, expected one")
            got = sha256(found[0])
            if got != digest:
                raise ValueError(f"{voice} {kind}: {found[0]} has SHA-256 {got}, the registered file has {digest}")
            out[voice][kind] = found[0]
    return out


def commit(root: pathlib.Path = REPO) -> str:
    out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty"], capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def run(files: dict, sentences: dict, listen: pathlib.Path | None = None, load=synth.load, sweep_run=sweep.run) -> dict:
    """Every voice through sweep.run, the rows pooled in severity.run's shape: one row per voice x
    sentence x condition. One voice is loaded at a time."""
    rows, infos, sr = [], [], None
    for voice, paths in files.items():
        v = load(paths["fastpitch"], paths["hifigan"])
        if v.name != voice:
            raise ValueError(f"the checkpoint at {paths['fastpitch']} names itself {v.name!r}, not {voice!r}")
        part = sweep_run(v, sentences, listen=listen)
        if part["settings"] != SETTINGS:
            raise ValueError(f"settings {part['settings']}, the registered ones are {SETTINGS}")
        if sr not in (None, part["sr"]):
            raise ValueError(f"{voice} renders at {part['sr']} Hz, the voice before it at {sr}")
        sr = part["sr"]
        rows += part["rows"]
        infos.append(part["voice_info"])
        del v
    return {"sr": sr, "tolerance": severity.TOLERANCE, "rows": rows, "settings": dict(SETTINGS), "voice_info": infos}


def by_voice(results: dict) -> dict:
    """PR-009's counts for each voice alone: {"voice": {"S1": {"met", "of"}, ...}}. Descriptive: the
    bars are for the pooled run (S2's is a count of the 32 sweeps), so no pass or fail here."""
    out = {}
    for voice in dict.fromkeys(r["voice"] for r in results["rows"]):
        c = severity.criteria({**results, "rows": [r for r in results["rows"] if r["voice"] == voice]})
        out[voice] = {k: {"met": c[k]["met"], "of": c[k]["of"]} for k in ("S1", "S2", "S3", "S4", "S5")}
    return out


def validity(results: dict) -> list[str]:
    """What would make a run not the registered one."""
    problems = []
    if list(results.get("voices", ())) != list(VOICES) or results.get("sentences") != SENTENCES:
        problems.append("not the registered voices and sentences")
    if {k: tuple(v) for k, v in results.get("scores", {}).items()} != severity.SCORES:
        problems.append("not the registered scores")
    want = len(VOICES) * len(SENTENCES) * len(severity.SCORES)
    if len(results["rows"]) != want:
        problems.append(f"{len(results['rows'])} sweeps, not {want}")
    if results.get("settings") != SETTINGS:
        problems.append(f"settings {results.get('settings')}")
    if results.get("model_sha256") != {v: {k: d for k, (_, d) in m.items()} for v, m in VOICES.items()}:
        problems.append("not the registered model files")
    if str(results.get("commit", "")).endswith("-dirty") or results.get("commit") in (None, "", "unknown"):
        problems.append(f"commit {results.get('commit')!r}: not a clean committed tree")
    return problems


def format_criteria(results: dict) -> str:
    """The verdict block, as the registration words it, then each voice's counts."""
    c = severity.criteria(results)
    pct = lambda k: f" ({100 * c[k]['met'] / c[k]['of']:.1f}%)" if c[k]["of"] else ""
    mark = lambda k: "PASS" if c[k]["pass"] else "FAIL"
    lines = [f"{REGISTRATION}, PR-009's criteria on FastPitch's renders (docs/preregistration.md):",
             f"  S1  pitch offset, rate, pauses in order, every sweep: {c['S1']['met']} of {c['S1']['of']}   {mark('S1')}",
             f"  S2  pitch range in order, at least {severity.S2_MIN_RANGE_IN_ORDER} of 32: {c['S2']['met']} of {c['S2']['of']}   {mark('S2')}",
             f"  S3  rate and pauses within 10%, every check: {c['S3']['met']} of {c['S3']['of']}   {mark('S3')}",
             f"  S4  pitch offset within 3 Hz or 10%, at least 90%: {c['S4']['met']} of {c['S4']['of']}{pct('S4')}   {mark('S4')}",
             f"  S5  pitch range within 10%, at least 80%: {c['S5']['met']} of {c['S5']['of']}{pct('S5')}   {mark('S5')}"]
    problems = validity(results)
    if problems:
        lines.append(f"{REGISTRATION}: NOT THE REGISTERED RUN -- " + "; ".join(problems))
    elif c["supported"]:
        lines.append(f"{REGISTRATION}: SUPPORTED")
    else:
        lines.append(f"{REGISTRATION}: NOT SUPPORTED -- failed " + ", ".join(k for k in ("S1", "S2", "S3", "S4", "S5")
                                                                         if not c[k]["pass"]))
    lines += ["", "each voice alone (descriptive; the bars are the pooled run's):"]
    for voice, counts in by_voice(results).items():
        lines.append(f"  {voice:24s} " + "   ".join(f"{k} {v['met']} of {v['of']}" for k, v in counts.items()))
    rate = [(s["target"]["rate"], s["achieved"]["rate"]) for r in results["rows"] for s in r["steps"]]
    lines.append(f"rate within 10%: {sum(severity.within('rate', t, a) for t, a in rate)} of {len(rate)}")
    return "\n".join(lines)


def stamp(results: dict, sentences: dict, files: dict, commit_now: str, seconds: float, **more) -> dict:
    results.update(commit=commit_now, ran=datetime.datetime.now().isoformat(timespec="seconds"), seconds=round(seconds),
                   voices=list(files), sentences=dict(sentences), scores={k: list(v) for k, v in severity.SCORES.items()},
                   model_sha256={v: {k: d for k, (_, d) in m.items()} for v, m in VOICES.items() if v in files},
                   model_files={v: {k: os.path.basename(p) for k, p in paths.items()} for v, paths in files.items()}, **more)
    results["criteria"] = severity.criteria(results)
    results["criteria_by_voice"] = by_voice(results)
    return results


def registered_run(confirm: str, run_path=RUN_PATH, audio_dir=AUDIO_DIR, files=None, load=synth.load,
                   sweep_run=sweep.run, commit_fn=commit) -> dict:
    """The one PR-012 run. Every refusal comes before a model loads."""
    if confirm != REGISTRATION:
        raise PermissionError(f"the eight sentences are locked: run with --confirm {REGISTRATION}, once, "
                              "after the registration is committed")
    run_path = pathlib.Path(run_path)
    if run_path.exists():
        raise FileExistsError(f"{run_path} exists. A saved run is evidence and is never overwritten.")
    commit_now = commit_fn()
    if commit_now.endswith("-dirty") or commit_now in ("", "unknown"):
        raise PermissionError(f"commit {commit_now!r}: the registered run starts from a clean committed tree")
    files = model_files() if files is None else files
    audio = pathlib.Path(audio_dir)
    audio.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = run(files, SENTENCES, listen=audio, load=load, sweep_run=sweep_run)
    stamp(results, SENTENCES, files, commit_now, time.time() - t0, registration=REGISTRATION)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results


def dry_run(out="results/ft3/pr012-dry.json", files=None, load=synth.load, sweep_run=sweep.run) -> dict:
    """The registered run's code path on PR-009's DEV sentences: both voices, the model files
    checked, the same pooling and report. Overwrites its own output; it is not evidence."""
    files = model_files() if files is None else files
    t0 = time.time()
    results = run(files, severity.DEV_SENTENCES, load=load, sweep_run=sweep_run)
    stamp(results, severity.DEV_SENTENCES, files, commit(), time.time() - t0, registration=f"{REGISTRATION} dry run, dev sentences")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return results


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if args == ["dry"]:
        results = dry_run()
        print(severity.report(results))
        print("\nDRY RUN on the dev sentences -- expect 'NOT THE REGISTERED RUN' below\n")
        print(format_criteria(results))
        print(f"\nsaved -> results/ft3/pr012-dry.json; {results['seconds']} s")
    elif len(args) == 3 and args[:2] == ["test", "--confirm"]:
        results = registered_run(args[2])
        print(severity.report(results))
        print("\n" + format_criteria(results))
        print(f"\nsaved -> {RUN_PATH}; renders and exact neutral clips: {AUDIO_DIR}/; {results['seconds']} s")
    else:
        sys.exit("usage: python -m src.ft3.pr012 dry\n"
                 f"       python -m src.ft3.pr012 test --confirm {REGISTRATION}")


if __name__ == "__main__":
    main()
