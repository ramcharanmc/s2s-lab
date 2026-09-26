"""
FT-1's helpers on the Vast.ai machine -- standard library only, so they run before anything is installed.

    python3 ft1_prep.py patch    REPO_DIR               add resume to moshi-finetune (idempotent)
    python3 ft1_prep.py manifest DATA_DIR               train.jsonl: every dialogue except the test fold
    python3 ft1_prep.py config   TEMPLATE OUT RUN_DIR TRAIN_JSONL
                                                        the training config; resumes from the latest checkpoint
    python3 ft1_prep.py last     RUN_DIR                "<step> <folder>" of the newest complete checkpoint
    python3 ft1_prep.py download DATA_DIR REVISION      DailyTalk at REVISION; waits out Hugging Face's 429s
                                                        (needs huggingface_hub, which moshi-finetune installs)

The config names train.jsonl as a FILE: given the folder, moshi-finetune would read every *.jsonl
in it, including DailyTalk's own manifest, test fold and all.

RESUME. moshi-finetune (kyutai-labs, Apache-2.0, pinned in ft1.sh) cannot resume: it refuses an
existing run directory and saves only the LoRA weights. If the rented machine fails, that means a
lost run, so this patch adds a `resume_from` argument that warm-starts from a saved checkpoint: the
adapter's weights, the step count and the learning-rate schedule are restored. NOT restored, because
the code never saves them: AdamW's moment estimates, and the data order (a resumed run reshuffles,
with seed = the resumed step). Both are disclosed in PR-010 if a resume happens.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

MARK = "# s2s-lab: resume"
TEST_FOLD = 0                       # dialogue number % 10 == 0: PR-010's test fold, never trained on

ARGS_ANCHOR = "    overwrite_run_dir: bool = False\n"
ARGS_ADD = f"    resume_from: str | None = None  {MARK}\n"
RUNDIR_ANCHOR = "        if run_dir.exists() and not args.overwrite_run_dir:\n"
RUNDIR_NEW = f"        if run_dir.exists() and not args.overwrite_run_dir and not args.resume_from:  {MARK}\n"
STATE_ANCHOR = "    state = TrainState(args.max_steps)\n"
RESUME_BLOCK = f"""
    {MARK}: warm start from a saved LoRA checkpoint (FT-1, if the rented machine fails). The adapter's
    # weights, the step and the learning-rate schedule are restored; AdamW's moments and the data
    # order are not (this code never saves them).
    if args.resume_from:
        import safetensors.torch as _st
        _dtype = next(model.parameters()).dtype
        _sd = {{k: v.to(dtype=_dtype) for k, v in _st.load_file(args.resume_from, device="cuda").items()}}
        _res = model.load_state_dict(_sd, strict=False)
        assert not _res.unexpected_keys, _res.unexpected_keys[:5]
        _missing = [k for k in _res.missing_keys if "lora" in k]
        assert not _missing, _missing[:5]
        _step = int(Path(args.resume_from).parent.parent.name.split("_")[-1])
        for _ in range(_step):
            scheduler.step()
        state.step = _step
        main_logger_info(f"s2s-lab: resumed from {{args.resume_from}} at step {{_step}}")
"""


def _replace_once(text: str, old: str, new: str, where: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"{where}: expected the anchor once, found it {text.count(old)} times -- "
                         "not the pinned moshi-finetune commit?")
    return text.replace(old, new)


def patch(repo: pathlib.Path) -> bool:
    """Adds resume_from to moshi-finetune. Returns False when the repo is already patched."""
    args_py, train_py = repo / "finetune" / "args.py", repo / "train.py"
    a, t = args_py.read_text(encoding="utf-8"), train_py.read_text(encoding="utf-8")
    if MARK in a and MARK in t:
        return False
    a = _replace_once(a, ARGS_ANCHOR, ARGS_ADD + ARGS_ANCHOR, "finetune/args.py")
    t = _replace_once(t, RUNDIR_ANCHOR, RUNDIR_NEW, "train.py (run dir check)")
    t = _replace_once(t, STATE_ANCHOR, STATE_ANCHOR + RESUME_BLOCK, "train.py (train state)")
    args_py.write_text(a, encoding="utf-8")
    train_py.write_text(t, encoding="utf-8")
    return True


def manifest(root: pathlib.Path) -> dict:
    """data/dailytalk.jsonl minus the test fold -> train.jsonl beside it. Returns the counts and the
    file's SHA-256, which the run log prints and PR-010 records."""
    src, out = root / "dailytalk.jsonl", root / "train.jsonl"
    kept, dropped = [], 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n = int(pathlib.PurePosixPath(json.loads(line)["path"]).stem)
        if n % 10 == TEST_FOLD:
            dropped += 1
        else:
            kept.append(line)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return {"train": len(kept), "test_fold_removed": dropped,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def latest_checkpoint(run_dir: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    """(step, lora.safetensors) of the newest complete checkpoint in run_dir, or None."""
    found = []
    for d in (run_dir / "checkpoints").glob("checkpoint_*"):
        m = re.fullmatch(r"checkpoint_(\d+)", d.name)
        lora = d / "consolidated" / "lora.safetensors"
        if m and lora.exists():
            found.append((int(m.group(1)), lora))
    return max(found) if found else None


def config(template: str, run_dir: pathlib.Path, train_jsonl: pathlib.Path) -> tuple[str, int]:
    """The training config: the template with its placeholders filled; resuming from the latest
    checkpoint in run_dir if there is one (then seed = that step). Returns (yaml text, resumed step)."""
    last = latest_checkpoint(run_dir)
    step, resume = (last[0], str(last[1])) if last else (0, "")
    text = (template.replace("{RUN_DIR}", str(run_dir)).replace("{TRAIN_DATA}", str(train_jsonl))
            .replace("{SEED}", str(step)).replace("{RESUME_FROM}", json.dumps(resume or None)))
    if "{" in re.sub(r"#.*", "", text):
        raise SystemExit("a placeholder was left unfilled in the config template")
    return text, step


DATASET = "kyutai/DailyTalkContiguous"


def rate_limited(error: Exception) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status == 429 or "429" in str(error)


def download(root: pathlib.Path, revision: str, attempts: int = 12, wait_s: float = 300,
             snapshot=None, sleep=None) -> int:
    """The whole dataset at `revision`, four files at a time. Hugging Face allows 1,000 requests per
    5 minutes with a token (fewer without); on a 429 this waits one window and goes on -- files
    already down are not fetched again. Returns the attempts it took."""
    if snapshot is None:
        from huggingface_hub import snapshot_download as snapshot
    if sleep is None:
        import time
        sleep = time.sleep
    for attempt in range(1, attempts + 1):
        try:
            snapshot(DATASET, repo_type="dataset", revision=revision, local_dir=str(root), max_workers=4)
            return attempt
        except Exception as error:                        # huggingface_hub's error class moved between versions
            if not rate_limited(error) or attempt == attempts:
                raise
            print(f"rate-limited (429), attempt {attempt} of {attempts}: waiting {wait_s:.0f} s", flush=True)
            sleep(wait_s)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "patch":
        print("patched" if patch(pathlib.Path(rest[0])) else "already patched")
    elif cmd == "manifest":
        print(json.dumps(manifest(pathlib.Path(rest[0]))))
    elif cmd == "last":
        last = latest_checkpoint(pathlib.Path(rest[0]))
        if not last:
            raise SystemExit("no complete checkpoint in " + rest[0])
        print(f"{last[0]} {last[1].parent}")
    elif cmd == "config":
        text, step = config(pathlib.Path(rest[0]).read_text(encoding="utf-8"), pathlib.Path(rest[2]), pathlib.Path(rest[3]))
        pathlib.Path(rest[1]).write_text(text, encoding="utf-8")
        print(f"config written; {'resuming from step ' + str(step) if step else 'fresh start'}")
    elif cmd == "download":
        print(f"downloaded {DATASET} @ {rest[1][:7]} in {download(pathlib.Path(rest[0]), rest[1])} attempt(s)")
    else:
        raise SystemExit("usage: ft1_prep.py patch REPO | manifest DATA_DIR | config TEMPLATE OUT RUN_DIR TRAIN_JSONL"
                         " | last RUN_DIR | download DATA_DIR REVISION")
