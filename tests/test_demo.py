"""The demo (s2slab.demo, s2slab.demoweb) on made-up run files, the stub rung and audio made here: no
model, no GPU, no network."""
import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from s2slab import demo
from s2slab.rungs import StubRung
from test_severity import voice

DAY = 1_790_000_000.0            # an instant; the tests move a clock from here


def wav(seconds=2.0, sr=24_000, level=1.0) -> bytes:
    audio = np.resize(voice(sr=sr), int(seconds * sr)) * level
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def browser_audio(fmt="webm", codec="libopus", seconds=2.0) -> bytes:
    """What a browser's recorder sends: the synthetic voice, encoded by PyAV."""
    av = pytest.importorskip("av")
    audio = np.resize(voice(sr=48_000), int(seconds * 48_000))
    buf = io.BytesIO()
    with av.open(buf, "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=48_000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray((audio[None, :] * 32767).astype(np.int16), format="s16", layout="mono")
        frame.sample_rate = 48_000
        resampler = av.AudioResampler(format=stream.format.name, layout="mono", rate=48_000)
        for f in resampler.resample(frame) + resampler.resample(None):
            for packet in stream.encode(f):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


@pytest.fixture
def tree(tmp_path):
    """A repo in miniature: a manifest, four clips, a turn-based run and a full-duplex run."""
    items = [{"id": f"clean-{i:02d}", "cohort": "clean", "source_id": f"1-2-{i}", "text": f"SENTENCE NUMBER {i}", "duration_s": 3.0 + i}
             for i in range(6)] + [{"id": "slow-00", "cohort": "slow", "source_id": "9", "text": "SLOW", "duration_s": 1.0}]
    (tmp_path / "evalset" / "audio").mkdir(parents=True)
    (tmp_path / "evalset" / "manifest.json").write_text(json.dumps({"items": items}))
    for i in range(4):
        (tmp_path / "evalset" / "audio" / f"clean-{i:02d}.wav").write_bytes(wav(0.5, 16_000))
    (tmp_path / "evalset" / "audio" / "secret.wav").write_bytes(b"not for the page")
    turn = lambda i: {"cohort": "clean", "transcript": f"sentence number {i}", "reply_text": f"reply {i}", "reply_s": 2.0,
                      "stages_ms": {"asr": 100.0, "llm": 300.0 + i, "tts": 200.0}, "first_audio_ms": 400.0 + 10 * i, "total_ms": 600.0 + i}
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "a.json").write_text(json.dumps({"machine": "Modal Tesla T4", "commit": "abc1234",
                                                          "turns": {f"clean-{i:02d}": turn(i) for i in range(6)}}))
    duplex = lambda i: {"cohort": "clean", "gap_ms": None if i % 2 else 150.0 + i, "reply_text": "" if i % 2 else "hello", "setup_s": 6.0,
                        "overlap": 0.1}
    (tmp_path / "runs" / "d.json").write_text(json.dumps({"machine": "Modal NVIDIA A10", "commit": "def5678",
                                                          "turns": {f"clean-{i:02d}": duplex(i) for i in range(6)}}))
    said = {"dis-00": "And, uh, they have a, a policy.", "dis-01": "It is, um, fine.", "dis-02": "Yes."}
    manifest = json.loads((tmp_path / "evalset" / "manifest.json").read_text())
    manifest["items"] += [{"id": k, "cohort": "disfluent", "source_id": k, "text": v, "duration_s": 5.0} for k, v in said.items()]
    (tmp_path / "evalset" / "manifest.json").write_text(json.dumps(manifest))
    for k in said:
        (tmp_path / "evalset" / "audio" / f"{k}.wav").write_bytes(wav(0.5, 16_000))
    heard = {"dis-00": "And they have a policy.", "dis-01": "It is fine.", "dis-02": "Yes."}
    turns = lambda texts: {k: {"cohort": "disfluent", "text": v, "asr_ms": 140.0} for k, v in texts.items()}
    (tmp_path / "runs" / "v.json").write_text(json.dumps({
        "registration": "PR-011", "machine": "Modal Tesla T4", "commit": "65c833c",
        "criteria": {"n_target": 3, "s1": {"base_raw_wer": 0.099, "lora_raw_wer": 0.023}, "s2": {"fillers_in_refs": 29, "base_kept": 1, "lora_kept": 29}},
        "models": {"small.en": {"turns": turns(heard)}, "ft2": {"turns": turns(said)}}}))
    return tmp_path


ROWS = (("R0", "a cascade", "runs/a.json"), ("R5", "full duplex", "runs/d.json"), ("R9", "not run", "runs/none.json"))
PICK = ("clean-00", "clean-01", "clean-02", "clean-03")


def test_a_recording_is_decoded_in_memory_whatever_the_browser_sent():
    for data in (wav(2.0), browser_audio("webm", "libopus"), browser_audio("ogg", "libopus"), browser_audio("mp4", "aac")):
        audio = demo.decode(data)
        assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) / demo.SR == pytest.approx(2.0, abs=0.1)
        assert 0.05 < np.abs(audio).max() <= 1.0


def test_what_cannot_be_a_turn_is_refused_with_a_sentence_for_the_visitor():
    for data, word in ((b"not audio at all", "not audio"), (wav(0.1), "at least a few words"), (wav(20.0), "the most this takes is 15 s"),
                       (wav(2.0, level=0.0), "silent"), (b"0" * (demo.MAX_UPLOAD_BYTES + 1), "MB")):
        with pytest.raises(ValueError, match=word):
            demo.decode(data)


def test_the_access_code_lets_nobody_in_when_none_is_configured():
    assert demo.code_ok("open sesame", "open sesame")
    for given, expected in (("open sesame", None), ("open sesame", ""), (None, "open sesame"), ("", "open sesame"), ("Open sesame", "open sesame")):
        assert not demo.code_ok(given, expected)


def test_the_budget_counts_the_time_the_gpu_is_up_not_the_time_a_request_takes():
    store = {}
    b = demo.Budget(store, cap_dollars=1.0, dollars_per_hour=3600.0, window_s=60.0, cold_start_s=45.0)     # a dollar a second
    assert b.allows(DAY) and b.left(DAY) == 1.0
    assert b.charge(DAY, DAY + 5) == pytest.approx(45 + 5 + 60)                    # woken: the cold start, the turn, the window
    assert b.charge(DAY + 10, DAY + 15) == pytest.approx(10)                       # still up: only what it adds to the bill
    assert b.charge(DAY + 15, DAY + 15) == pytest.approx(0)
    assert b.charge(DAY + 500, DAY + 505) == pytest.approx(110)                    # it had gone to sleep
    assert b.spent(DAY) == pytest.approx(230) and not b.allows(DAY) and b.left(DAY) == 0.0
    tomorrow = DAY + 86_400
    assert b.day(tomorrow) != b.day(DAY) and b.allows(tomorrow) and b.spent(tomorrow) == 0.0
    real = demo.Budget({}, cap_dollars=1.0, dollars_per_hour=0.59)
    assert real.charge(DAY, DAY + 3) == pytest.approx((45 + 3 + 60) / 3600 * 0.59)  # a turn on a sleeping T4: about two cents


def test_the_measured_part_reads_the_run_files_and_lists_a_missing_one_as_missing(tree):
    data = demo.measured(tree, clips=PICK, rows=ROWS, verbatim_clips=("dis-00", "dis-01"), verbatim_run="runs/v.json")
    assert [c["id"] for c in data["clips"]] == list(PICK) and data["clips"][1]["reference"] == "Sentence number 1."
    assert "CC BY 4.0" in data["clips"][0]["source"] and "different GPUs" in data["note"]
    r0, r5, r9 = data["rungs"]
    assert (r0["kind"], r0["turns_in_run"], r0["first_audio_p50_ms"], r0["machine"], r0["commit"]) == ("turn", 6, 425.0, "Modal Tesla T4", "abc1234")
    assert [t["clip"] for t in r0["turns"]] == list(PICK) and r0["turns"][2]["stages_ms"] == {"asr": 100.0, "llm": 302.0, "tts": 200.0}
    assert (r5["kind"], r5["answered"], r5["gap_p50_ms"]) == ("duplex", 3, 152.0) and r5["turns"][1]["gap_ms"] is None
    assert r9 == {"label": "R9", "what": "not run", "file": "runs/none.json", "missing": True}
    json.dumps(data)


def test_the_clips_shown_are_the_shortest_clean_ones_and_every_run_named_is_committed(tree):
    manifest = json.loads((tree / "evalset" / "manifest.json").read_text())
    assert demo.pick_clips(manifest) == PICK
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    assert demo.CLIPS == demo.pick_clips(json.loads((root / "evalset" / "manifest.json").read_text(encoding="utf-8")))
    for label, what, rel in demo.MEASURED:
        assert (root / rel).exists(), rel
    assert [label for label, _, _ in demo.MEASURED] == ["R0", "R1", "R2", "R3", "R4", "R5 Moshi", "R5 PersonaPlex"]


def test_a_live_turn_is_a_record_with_the_reply_as_a_wav_and_a_rung_loads_once():
    made = []

    def get(rung_id):
        made.append(rung_id)
        return StubRung()

    live = demo.Live({"R0": "stub"}, get=get)
    first, second = live.turn("R0", voice()), live.turn("R0", voice())
    assert made == ["stub"] and live.load_ms["R0"] >= 0
    assert first["label"] == "R0" and first["rung"] == "stub" and set(first["stages_ms"]) == {"asr", "llm", "tts"}
    audio, sr = sf.read(io.BytesIO(base64.b64decode(first["reply_wav"])))
    assert sr == 24_000 and len(audio) == 12_000 and first["reply_s"] == 0.5
    assert first["first_sound_ms"] is None                                          # the stub's reply is silence: no first sound
    json.dumps(second)
    with pytest.raises(KeyError, match="no rung 'R7' here"):
        live.turn("R7", voice())


# --------------------------------------------------------------------------- #
# the web app
# --------------------------------------------------------------------------- #
@pytest.fixture
def web(tree, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")
    clock = {"t": DAY}
    budget = demo.Budget({}, cap_dollars=0.05, dollars_per_hour=0.59)
    live = demo.Live({"R0": "stub", "R1": "stub"}, get=lambda rung_id: StubRung())
    app = demoweb.create_app(live, budget, root=tree, machine="a test", code_env="TEST_DEMO_CODE", clock=lambda: clock["t"],
                             clips=PICK, rows=ROWS, verbatim_clips=("dis-00", "dis-01"), verbatim_run="runs/v.json")
    return TestClient(app), clock, budget


def test_the_page_the_measured_data_and_only_the_clips_it_shows(web):
    client, _, _ = web
    page = client.get("/")
    assert page.status_code == 200 and "Measured" in page.text and "Nothing is stored" in page.text
    assert "no clinical validity" in page.text and "http://" not in page.text and "https://" not in page.text   # nothing fetched from outside
    assert 'rel="icon" href="data:,"' in page.text                                        # and no request for an icon
    data = client.get("/api/measured").json()
    assert [r["label"] for r in data["rungs"]] == ["R0", "R5", "R9"]
    clip = client.get("/api/clip/clean-01")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav" and clip.content[:4] == b"RIFF"
    assert client.get("/api/clip/dis-01").status_code == 200 and data["verbatim"]["clips"][0]["lora"] == "And, uh, they have a, a policy."
    for other in ("secret", "clean-05", "dis-02", "..%2F..%2Fmanifest", "clean-01.wav"):
        assert client.get(f"/api/clip/{other}").status_code == 404
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_the_live_part_needs_the_code_answers_with_every_rung_and_charges_the_budget(web):
    client, clock, budget = web
    send = lambda code, data=None, rungs="": client.post("/api/live", files={"audio": ("turn", data or wav(2.0), "audio/wav")},
                                                          data={"rungs": rungs}, headers={"X-Access-Code": code} if code else {})
    assert send(None).status_code == 401 and send("wrong").status_code == 401 and budget.spent(DAY) == 0.0
    bad = send("open sesame", data=b"not audio")
    assert bad.status_code == 400 and "not audio" in bad.json()["detail"] and budget.spent(DAY) == 0.0   # a refusal costs nothing
    assert send("open sesame", rungs="R0,R7").status_code == 400
    ok = send("open sesame").json()
    assert [t["label"] for t in ok["turns"]] == ["R0", "R1"] and ok["clip_s"] == pytest.approx(2.0, abs=0.05)
    assert ok["charged"] == pytest.approx((45 + 60) / 3600 * 0.59, abs=1e-4) and ok["machine"] == "a test"
    one = send("open sesame", rungs="R1").json()
    assert [t["label"] for t in one["turns"]] == ["R1"] and one["charged"] == 0.0            # the clock did not move: still up
    status = client.get("/api/status").json()
    assert status["rungs"] == ["R0", "R1"] and status["code_set"] and status["open"] and status["budget_cap"] == 0.05
    assert status["budget_left"] == pytest.approx(0.05 - ok["charged"], abs=1e-3) and status["keeps"].startswith("nothing")
    for _ in range(3):                                                                       # three more bursts spend the day's five cents
        clock["t"] += 1_000
        last = send("open sesame")
    assert last.status_code == 429 and "budget" in last.json()["detail"]
    assert not client.get("/api/status").json()["open"]
    clock["t"] += 86_400
    assert send("open sesame").status_code == 200                                            # a new day


# --------------------------------------------------------------------------- #
# the verbatim panel, and the verbatim tier
# --------------------------------------------------------------------------- #
def test_the_verbatim_panel_is_pr011s_run_and_says_how_its_clips_were_chosen(tree):
    data = demo.measured(tree, clips=PICK, rows=ROWS, verbatim_clips=("dis-00", "dis-01"), verbatim_run="runs/v.json")
    v = data["verbatim"]
    assert (v["registration"], v["commit"], v["file"]) == ("PR-011", "65c833c", "runs/v.json")
    assert v["cohort"] == {"clips": 3, "base_raw_wer": 9.9, "lora_raw_wer": 2.3, "fillers": 29, "base_kept": 1, "lora_kept": 29}
    assert "chosen for that" in v["note"] and "same-speaker" in v["note"]
    first = v["clips"][0]
    assert (first["id"], first["reference"], first["base"], first["lora"]) == ("dis-00", "And, uh, they have a, a policy.",
                                                                               "And they have a policy.", "And, uh, they have a, a policy.")
    assert "Apache-2.0" in first["source"] and data["audio"] == list(PICK) + ["dis-00", "dis-01"]
    items = {i["id"]: i for i in json.loads((tree / "evalset" / "manifest.json").read_text())["items"]}
    assert demo.pick_verbatim(json.loads((tree / "runs" / "v.json").read_text()), items) == ("dis-00", "dis-01")   # 2 and 1 errors removed; dis-02 none
    gone = demo.measured(tree, clips=PICK, rows=ROWS, verbatim_run="runs/none.json")
    assert gone["verbatim"] is None and gone["audio"] == list(PICK)


def test_the_verbatim_clips_shown_are_the_ones_the_rule_picks_from_the_committed_run():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    items = {i["id"]: i for i in json.loads((root / "evalset" / "manifest.json").read_text(encoding="utf-8"))["items"]}
    run = json.loads((root / demo.VERBATIM_RUN).read_text(encoding="utf-8"))
    assert demo.VERBATIM_CLIPS == demo.pick_verbatim(run, items)
    from s2slab import r2verbatim
    assert run["export_sha256"] == r2verbatim.EXPORT_SHA256 and run["export_name"] == r2verbatim.EXPORT_NAME


def test_the_laptop_offers_the_verbatim_tier_only_where_the_export_is(tmp_path, monkeypatch, tree):
    pytest.importorskip("fastapi")
    from s2slab import demoweb, r2verbatim
    monkeypatch.setenv("DEMO_ROOT", str(tree))
    monkeypatch.setattr(demo, "CLIPS", PICK)
    monkeypatch.setattr(demoweb, "CLIPS", PICK)
    seen = {}
    monkeypatch.setattr(demoweb, "create_app", lambda live, budget, **kw: seen.update(offered=dict(live.offered)))
    monkeypatch.setenv(r2verbatim.EXPORT_ENV, str(tmp_path / "nowhere"))
    demoweb.laptop()
    assert seen["offered"] == {"R0": "r0", "R1": "r1", "R2": "r2"}
    (tmp_path / "ct2").mkdir()
    (tmp_path / "ct2" / "model.bin").write_bytes(b"a model")
    monkeypatch.setenv(r2verbatim.EXPORT_ENV, str(tmp_path / "ct2"))
    demoweb.laptop()
    assert "R2 verbatim" not in seen["offered"]                                  # a model, but not PR-011's
    import hashlib
    monkeypatch.setattr(r2verbatim, "EXPORT_SHA256", hashlib.sha256(b"a model").hexdigest())
    demoweb.laptop()
    assert seen["offered"] == {"R0": "r0", "R1": "r1", "R2": "r2", "R2 verbatim": "r2-verbatim"}
