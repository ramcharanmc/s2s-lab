"""
Tests for FT-1's kit and PR-010's criterion: vast/ft1_prep.py (the resume patch, the training list
without the test fold, the config, the download's waits), vast/ft1.sh (syntax, pins, what FT-1's
run needed), dailytalk.pr010 on made-up rows,
and the LoRA Moshi adapter's shape. No GPU, no download, no network.
"""

import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess

import pytest

from s2slab import dailytalk, r5
from s2slab.turngaps import IPU, summary

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAST = ROOT / "vast"


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("ft1_prep", VAST / "ft1_prep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_repo(tmp_path, prep):
    """The three anchors of moshi-finetune (2acc879) in miniature."""
    (tmp_path / "finetune").mkdir()
    (tmp_path / "finetune" / "args.py").write_text("class TrainArgs:\n    param_dtype: str = 'bfloat16'\n\n" + prep.ARGS_ANCHOR)
    (tmp_path / "train.py").write_text("from pathlib import Path\n\ndef _train(args):\n    if True:\n" + prep.RUNDIR_ANCHOR
                                       + "            raise RuntimeError('exists')\n" + prep.STATE_ANCHOR + "    run()\n")
    return tmp_path


def test_the_resume_patch_adds_the_argument_and_the_warm_start_once(prep, fake_repo):
    assert prep.patch(fake_repo) is True
    args, train = (fake_repo / "finetune" / "args.py").read_text(), (fake_repo / "train.py").read_text()
    assert "resume_from: str | None = None" in args and "not args.resume_from" in train
    assert train.index(prep.STATE_ANCHOR) < train.index("if args.resume_from:") < train.index("    run()")
    assert "scheduler.step()" in train and "state.step = _step" in train
    compile(train, "train.py", "exec")                                   # the patched file is valid Python
    assert prep.patch(fake_repo) is False                                # a second run changes nothing
    assert train == (fake_repo / "train.py").read_text()


def test_the_patch_refuses_code_that_is_not_the_pinned_commit(prep, fake_repo):
    (fake_repo / "train.py").write_text("def _train(args):\n    pass\n")
    with pytest.raises(SystemExit, match="not the pinned moshi-finetune commit"):
        prep.patch(fake_repo)


def test_the_training_list_drops_exactly_the_test_fold(prep, tmp_path):
    lines = [json.dumps({"path": f"data_stereo/{n}.wav", "duration": 10.0}) for n in range(2541)]
    (tmp_path / "dailytalk.jsonl").write_text("\n".join(lines) + "\n")
    got = prep.manifest(tmp_path)
    kept = (tmp_path / "train.jsonl").read_text().splitlines()
    assert got["test_fold_removed"] == 255 and got["train"] == len(kept) == 2541 - 255
    assert not any(json.loads(line)["path"].endswith(("0.wav",)) for line in kept)
    assert got["sha256"] == hashlib.sha256((tmp_path / "train.jsonl").read_bytes()).hexdigest()
    assert all(dailytalk.fold(n) == "test" for n in range(0, 2541, 10))   # the same rule as the evaluation's


def test_the_config_starts_fresh_or_resumes_from_the_newest_complete_checkpoint(prep, tmp_path):
    template = (VAST / "ft1.yaml").read_text()
    run = tmp_path / "run"
    text, step = prep.config(template, run, tmp_path / "train.jsonl")
    assert step == 0 and "seed: 0" in text and "resume_from: null" in text
    assert f"train_data: '{tmp_path / 'train.jsonl'}'" in text                 # a file, never the folder
    for s, complete in ((100, True), (200, True), (300, False)):               # 300 was cut off mid-save
        d = run / "checkpoints" / f"checkpoint_{s:06d}" / "consolidated"
        d.mkdir(parents=True)
        if complete:
            (d / "lora.safetensors").write_bytes(b"x")
    text, step = prep.config(template, run, tmp_path / "train.jsonl")
    assert step == 200 and "seed: 200" in text and "checkpoint_000200/consolidated/lora.safetensors" in text
    with pytest.raises(SystemExit, match="placeholder"):
        prep.config("x: {NOPE}\n", run, tmp_path / "t.jsonl")


def test_the_example_settings_are_kyutais_and_the_script_pins_what_it_runs():
    y = (VAST / "ft1.yaml").read_text()
    for setting in ("rank: 128", "lr: 2e-6", "max_steps: 2000", "batch_size: 16", "duration_sec: 100", "ckpt_freq: 100"):
        assert setting in y
    sh = (VAST / "ft1.sh").read_text()
    assert "MF_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b" in sh and "MOSHI_COMMIT=9837ca3" in sh
    assert f"DATA_REV={dailytalk.REVISION}" in sh and "data/dailytalk/train.jsonl" in sh
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(VAST / "ft1.sh")]).returncode == 0     # parses


def test_the_script_carries_what_ft1s_run_needed():
    sh = (VAST / "ft1.sh").read_text()
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in sh
    assert sh.index("export CUDA_VISIBLE_DEVICES=") < sh.index("torchrun --nproc-per-node")
    assert "echo y | vastai" in sh and "ft1_prep.py\" download" in sh
    assert int(sh.split("MAX_HOURS=${MAX_HOURS:-")[1].split("}")[0]) >= 13        # FT-1 took 12.3 h of machine


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_vasts_instance_variables_are_read_from_the_containers_own_environment(tmp_path):
    sh = (VAST / "ft1.sh").read_text()
    start = sh.index("from_pid1() {")
    fn = sh[start:sh.index("\n}\n", start) + 2]
    environ = tmp_path / "environ"
    environ.write_bytes(b"PATH=/usr/bin\0CONTAINER_ID=52595121\0CONTAINER_API_KEY=k=with=equals\0")
    script = f"{fn}\nfrom_pid1 CONTAINER_ID; from_pid1 CONTAINER_API_KEY; from_pid1 MISSING; echo end"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         env={"PID1_ENVIRON": environ.as_posix(), "PATH": "/usr/bin:/bin"})
    assert out.stdout.split("\n") == ["52595121", "k=with=equals", "end", ""]


class _RateLimited(Exception):
    def __init__(self):
        super().__init__("429 Client Error: Too Many Requests")
        self.response = type("R", (), {"status_code": 429})()


def test_the_download_waits_out_the_rate_limit_and_nothing_else(prep, tmp_path):
    calls, waits = [], []

    def snapshot(repo, **kw):
        calls.append((repo, kw["revision"], kw["max_workers"]))
        if len(calls) < 3:
            raise _RateLimited()

    assert prep.download(tmp_path, "33e1b50", snapshot=snapshot, sleep=waits.append, wait_s=300) == 3
    assert calls == [("kyutai/DailyTalkContiguous", "33e1b50", 4)] * 3 and waits == [300, 300]

    def broken(repo, **kw):
        raise ValueError("404: no such revision")
    with pytest.raises(ValueError):
        prep.download(tmp_path, "bad", snapshot=broken, sleep=waits.append)
    assert waits == [300, 300]                                                    # no wait on other errors

    def always(repo, **kw):
        raise _RateLimited()
    with pytest.raises(_RateLimited):
        prep.download(tmp_path, "33e1b50", snapshot=always, sleep=lambda s: None, attempts=3)


def test_pr010_is_supported_only_when_the_whole_interval_is_above_zero():
    user = [IPU(1.0, 2.0), IPU(4.0, 5.0)]
    ref = {n: summary(user, [IPU(2.4, 3.0), IPU(5.4, 6.0)]) for n in dailytalk.TEST_IDS}         # 400 ms gaps
    base = {n: summary(user, [IPU(1.8, 3.0), IPU(5.1, 6.0)]) for n in dailytalk.TEST_IDS}        # -200, +100
    closer = {n: summary(user, [IPU(2.3, 3.0), IPU(5.35, 6.0)]) for n in dailytalk.TEST_IDS}     # 300, 350
    c = dailytalk.pr010(base, closer, ref, n=200)
    assert c["n_dialogues"] == 100 and c["distance_base_ms"] == 450.0 and c["distance_lora_ms"] == 75.0
    assert c["gain_ms"] == 375.0 and c["ci95_ms"][0] > 0 and c["supported"]
    same = dailytalk.pr010(base, base, ref, n=200)
    assert same["gain_ms"] == 0.0 and not same["supported"]
    results = {"reference": {str(n): r for n, r in ref.items()}, "commit": "abc1234", "adapter_sha256": "ff", "criteria": c}
    assert dailytalk.format_pr010(results).endswith("PR-010: SUPPORTED")
    results["commit"] = "abc1234-dirty"
    assert "NOT THE REGISTERED RUN" in dailytalk.format_pr010(results)
    assert dailytalk.TEST_IDS[:3] == (0, 10, 20) and len(dailytalk.TEST_IDS) == 100


def test_the_lora_adapter_reads_ft1s_volume_and_the_base_one_has_none():
    m = r5.get("r5-moshi-lora")
    assert (m.lora_weights, m.config_path) == ("/ft1/lora.safetensors", "/ft1/config.json") and m.rung_id == "r5-moshi-lora"
    assert r5.get("r5-moshi").lora_weights is None and m.repo == r5.MOSHI_REPO


def test_the_pr010_run_is_one_function_with_the_volume(r5_bench=None):
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("r5_bench_ft1", ROOT / "modal_app" / "r5_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    assert hasattr(bench.pr010_remote, "remote") and hasattr(bench.pr010, "info")
