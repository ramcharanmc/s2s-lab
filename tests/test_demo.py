"""The demo (s2slab.demo, s2slab.demoweb) on made-up run files, the stub rung and audio made here: no
model, no GPU, no network."""
import base64
import inspect
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
    assert not demo.code_ok("sesame", "sesame") and not demo.code_usable("123456789") and demo.code_usable("1234567890")   # too short to be a code
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
    assert [label for label, _, _ in demo.MEASURED] == ["R0", "R1", "R2", "R3", "R4", "R4 streamed", "R5 Moshi", "R5 PersonaPlex"]


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
    assert "What your voice carried" in page.text and "What the words do not carry" in page.text
    for said in ('id="rec" class="rec" role="status" hidden', "stops by itself", "it was stopped at the limit", 'id="mine" controls hidden',
                 "Listen, then send", "ago. "):                                           # the recorder says what it is doing
        assert said in page.text, said
    data = client.get("/api/measured").json()
    assert [r["label"] for r in data["rungs"]] == ["R0", "R5", "R9"] and data["prosody"] is None     # its run file is not in this tree
    assert data["findings"] == [] and "What else was claimed, and what happened" in page.text
    clip = client.get("/api/clip/clean-01")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav" and clip.content[:4] == b"RIFF"
    assert client.get("/api/clip/dis-01").status_code == 200 and data["verbatim"]["clips"][0]["lora"] == "And, uh, they have a, a policy."
    for other in ("secret", "clean-05", "dis-02", "..%2F..%2Fmanifest", "clean-01.wav"):
        assert client.get(f"/api/clip/{other}").status_code == 404
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404
    handlers = [r.endpoint for r in client.app.routes if getattr(r, "path", "").startswith("/api/") or getattr(r, "path", "") == "/"]
    assert len(handlers) == 8 and not any(inspect.iscoroutinefunction(h) for h in handlers)   # blocking calls stay off the event loop


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
    assert ok["cold"] and not one["cold"] and status["wake_s"] == ok["wake_s"] == 0.0 and status["about"] == {}
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


# --------------------------------------------------------------------------- #
# a rung that fails, and rungs on another machine
# --------------------------------------------------------------------------- #
class Deaf(StubRung):
    def _respond(self, audio, sr, sink):
        raise RuntimeError("out of memory at /a/path/nobody/should/see")


def test_a_rung_that_fails_is_answered_for_with_a_sentence_and_the_others_still_answer(capsys):
    def get(rung_id):
        if rung_id == "absent":
            raise ValueError("/a/path/nobody/should/see/model.bin has SHA-256 00, PR-011's model has b3")
        return Deaf() if rung_id == "deaf" else StubRung()

    live = demo.Live({"R0": "stub", "R1": "deaf", "R2 verbatim": "absent"}, get=get)
    assert live.load_all() >= 0 and live.unavailable == {"R2 verbatim": demo.NOT_LOADED}
    out = live.turns(["R0", "R1", "R2 verbatim"], voice())
    r0, r1, r2 = out["turns"]
    assert "error" not in r0 and r0["reply_wav"] and out["cold"]
    assert r1 == {"label": "R1", "rung": "deaf", "error": demo.NOT_ANSWERED}
    assert r2 == {"label": "R2 verbatim", "rung": "absent", "error": demo.NOT_LOADED}
    assert "path" not in json.dumps(out) and "/a/path/nobody/should/see" in capsys.readouterr().err   # the log has it; the visitor does not
    assert not live.turns(["R0"], voice())["cold"]
    with pytest.raises(KeyError, match="no rung 'R7' here"):
        live.turns(["R7"], voice())
    lazy = demo.Live({"R2 verbatim": "absent"}, get=get)                                            # not loaded ahead: the same answer
    assert lazy.turn("R2 verbatim", voice())["error"] == demo.NOT_LOADED and lazy.unavailable == {"R2 verbatim": demo.NOT_LOADED}


def test_rungs_on_another_machine_are_sent_the_samples_and_answer_as_if_here():
    far = demo.Live({"R0": "stub"}, get=lambda rung_id: StubRung())
    sent = []

    def call(labels, samples):
        sent.append((labels, type(samples), len(samples)))
        return demo.answer(far, labels, samples)

    near = demo.Remote({"R0": "stub"}, call)
    audio = voice()
    out = near.turns(("R0",), audio.astype(np.float64))
    assert sent == [(["R0"], bytes, 4 * len(audio))] and near.offered == {"R0": "stub"}
    assert out["cold"] and out["turns"][0]["label"] == "R0" and out["turns"][0]["reply_s"] == 0.5
    json.dumps(out)


# --------------------------------------------------------------------------- #
# wrong codes
# --------------------------------------------------------------------------- #
def test_thirty_wrong_codes_in_five_minutes_close_the_door_to_everyone_for_five():
    assert (demo.WRONG_CODES, demo.LOCK_S) == (30, 300.0)
    store = {}
    door = demo.Door(store)
    for i in range(29):
        door.wrong(DAY + i)
    assert door.locked_s(DAY + 29) == 0
    door.wrong(DAY + 400)                                         # the 29 have aged out: this is the first of a new count
    assert door.locked_s(DAY + 400) == 0 and store["wrong_codes"] == [DAY + 400]
    for i in range(29):
        door.wrong(DAY + 401 + i)
    assert door.locked_s(DAY + 430) == pytest.approx(299) and store["wrong_codes"] == []
    assert door.locked_s(DAY + 429 + 300) == 0
    assert demo.Door({}, tries=1, lock_s=5).locked_s(DAY) == 0


def test_a_locked_door_refuses_the_right_code_wakes_nothing_and_opens_again(web):
    client, clock, budget = web
    send = lambda code: client.post("/api/live", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": ""},
                                    headers={"X-Access-Code": code} if code else {})
    for _ in range(9):
        assert send(None).status_code == 401                      # no code is not a guess
    for i in range(1, 30):
        assert send(f"{i:05d}").status_code == 401
    assert client.get("/api/status").json()["locked_s"] == 0 and send("open sesame").status_code == 200
    spent = budget.spent(DAY)
    assert send("00030").status_code == 401                       # the 30th: the right code in between did not reset the count
    assert send("00031").status_code == 429                       # and a locked door does not say whether a code is wrong
    locked = send("open sesame")
    assert locked.status_code == 429 and "wrong codes" in locked.json()["detail"] and budget.spent(DAY) == spent
    assert client.get("/api/status").json()["locked_s"] == 300
    clock["t"] += 301
    assert client.get("/api/status").json()["locked_s"] == 0 and send("open sesame").status_code == 200


def test_a_code_too_short_to_be_one_keeps_the_live_part_closed_and_the_page_says_so(tree, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "12345")
    live = demo.Live({"R0": "stub"}, get=lambda rung_id: StubRung())
    client = TestClient(demoweb.create_app(live, demo.Budget({}, 1.0, 0.59), root=tree, code_env="TEST_DEMO_CODE", clock=lambda: DAY,
                                           clips=PICK, rows=ROWS, verbatim_run="runs/none.json"))
    status = client.get("/api/status").json()
    assert (status["code_set"], status["code_short"], status["min_code_len"]) == (False, True, 10)
    r = client.post("/api/live", files={"audio": ("turn", wav(2.0), "audio/wav")}, headers={"X-Access-Code": "12345"})
    assert r.status_code == 401 and "code_short" in client.get("/").text


# --------------------------------------------------------------------------- #
# prosody: the measured panel, and what a visitor's voice carried
# --------------------------------------------------------------------------- #
def tone(hz, seconds=2.0, sr=16_000):
    """A voice-like tone: a fundamental and two harmonics, so Praat finds a pitch."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t) + 0.15 * np.sin(4 * np.pi * hz * t) + 0.08 * np.sin(6 * np.pi * hz * t)).astype(np.float32)


def test_a_voices_pitch_and_range_are_measured_and_silence_is_not_given_one():
    pytest.importorskip("parselmouth")
    low, high = demo.voice(tone(120.0)), demo.voice(tone(240.0))
    assert low["pitch_hz"] == pytest.approx(120, abs=2) and high["pitch_hz"] == pytest.approx(240, abs=3)
    assert low["range_st"] < 0.5 and low["voiced_s"] == pytest.approx(2.0, abs=0.2)
    glide = np.concatenate([tone(110.0, 1.0), tone(220.0, 1.0)])                  # an octave: twelve semitones
    assert demo.voice(glide)["range_st"] == pytest.approx(12, abs=0.5)
    quiet = demo.voice(np.zeros(16_000, np.float32))
    assert quiet["pitch_hz"] is None and quiet["range_st"] is None and quiet["voiced_s"] == 0
    assert (low["floor_hz"], low["ceiling_hz"]) == (75, 600) and "not a reading of how anyone feels" in low["note"]
    assert "level_db" not in low and "emotion" not in json.dumps(low)               # no loudness, and no label
    json.dumps(low)


def prosody_run(tmp_path):
    pairs, answers = [], {}
    for e, name in (("SAD", "sad"), ("FEA", "fea"), ("ANG", "ang")):
        for i, (n, l) in enumerate(((0.1, 0.9), (0.2, 0.5), (0.6, 0.7))):           # differences 0.8, 0.3, 0.1: the middle is 0.3
            pid = f"{name}-t{i + 1:02d}"
            pairs.append({"id": pid, "emotion": e, "sentence": "IEO", "neutral": f"{pid}-n", "loaded": f"{pid}-l"})
            answers[f"{pid}-n"], answers[f"{pid}-l"] = {"p_yes": {"upset": n}}, {"p_yes": {"upset": l}}
    (tmp_path / "runs" / "p.json").write_text(json.dumps({
        "run": "9 test pairs · confirms PR-006 · a model on a GPU · commit 5c1f3aa", "wordings": {"upset": "Is the person speaking upset?"},
        "pairs": pairs, "answers": answers, "scores": {"upset": {"SAD": [2, 3, 3], "FEA": [2, 3, 3], "ANG": [2, 3, 3]}}}))
    return json.loads((tmp_path / "runs" / "p.json").read_text())


def test_the_prosody_panel_is_pr006s_run_its_pairs_the_middle_ones_and_no_audio(tree):
    run = prosody_run(tree)
    assert demo.pick_pairs(run) == ("ang-t02", "fea-t02", "sad-t02")
    data = demo.measured(tree, clips=PICK, rows=ROWS, verbatim_run="runs/none.json", prosody_pairs=demo.pick_pairs(run), prosody_run="runs/p.json")
    p = data["prosody"]
    assert (p["registration"], p["question"], p["file"]) == ("PR-006", "Is the person speaking upset?", "runs/p.json")
    assert p["scores"][0] == {"emotion": "sad", "answered": 2, "ranked": 3, "pairs": 3}
    assert p["pairs"][0] == {"id": "ang-t02", "emotion": "angry", "sentence": "It's eleven o'clock", "neutral": 0.2, "loaded": 0.5}
    assert "not played here" in p["note"] and "Open Database License" in p["source"] and data["audio"] == list(PICK)   # no clip of it is served
    json.dumps(data)


def test_the_prosody_pairs_shown_are_the_ones_the_rule_picks_from_the_committed_run():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    run = json.loads((root / demo.PROSODY_RUN).read_text(encoding="utf-8"))
    assert demo.PROSODY_PAIRS == demo.pick_pairs(run)
    p = demo.prosody(root)
    assert [(s["emotion"], s["answered"], s["ranked"], s["pairs"]) for s in p["scores"]] == [("sad", 11, 13, 13), ("fearful", 11, 13, 13),
                                                                                           ("angry", 9, 13, 13)]
    assert all(x["loaded"] > x["neutral"] for x in p["pairs"]) and "/" not in p["run"].replace("Qwen/Qwen2", "")    # no path in what is shown


def test_a_live_turn_carries_the_voices_measures_and_a_failed_measure_costs_no_turn(web, monkeypatch):
    pytest.importorskip("parselmouth")
    client, _, _ = web
    send = lambda: client.post("/api/live", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": "R0"},
                               headers={"X-Access-Code": "open sesame"})
    ok = send().json()
    assert 60 < ok["voice"]["pitch_hz"] < 600 and ok["voice"]["range_st"] >= 0 and [t["label"] for t in ok["turns"]] == ["R0"]
    from s2slab import demoweb

    def broken(audio):
        raise RuntimeError("Praat fell over at /a/path/nobody/should/see")

    monkeypatch.setattr(demoweb, "voice", broken)
    again = send()
    assert again.status_code == 200 and again.json()["voice"] is None and "path" not in again.text


# --------------------------------------------------------------------------- #
# more than one machine
# --------------------------------------------------------------------------- #
def test_each_machine_is_billed_for_its_own_time_under_one_cap():
    store = {}
    b = demo.Budget(store, cap_dollars=1.0, dollars_per_hour=3600.0, window_s=300.0, cold_start_s=0.0, machines={"a10": (7200.0, 120.0)})
    assert b.charge(DAY, DAY + 5) == pytest.approx(5 + 300)                       # the first machine: a dollar a second
    assert b.charge(DAY, DAY + 60, "a10") == pytest.approx(2 * (60 + 120))        # the other: two, and its own window
    assert b.charge(DAY + 70, DAY + 75, "a10") == pytest.approx(2 * 15)           # still up: only what it adds
    assert b.charge(DAY + 70, DAY + 75) == pytest.approx(70)                      # and the first is on its own clock
    assert b.spent(DAY) == pytest.approx(305 + 360 + 30 + 70) and set(store) >= {"billed_until", "billed_until:a10"}
    with pytest.raises(KeyError):
        b.charge(DAY, DAY + 1, "l40s")                                            # a machine nobody priced is not run


def several(clock, broken=False):
    first = demo.Live({"R0": "stub", "R1": "stub"}, get=lambda rung_id: StubRung())
    far = demo.Live({"R3": "stub"}, get=lambda rung_id: StubRung())

    def call(labels, samples):
        clock["t"] += 40                                                          # the other machine takes its time
        if broken:
            raise RuntimeError("no capacity at /a/path/nobody/should/see")
        return dict(demo.answer(far, labels, samples), machine="Modal NVIDIA A10G", load_s=12.0)

    return demo.Several(first, {"a10": demo.Remote({"R3": "stub"}, call)}, says={"a10": "an A10"}, clock=lambda: clock["t"])


def test_the_other_machines_answer_only_when_asked_and_beside_the_first():
    clock = {"t": DAY}
    live = several(clock)
    assert (live.default, live.optional, list(live.offered)) == (["R0", "R1"], {"R3": "a10"}, ["R0", "R1", "R3"])
    usual = live.turns(live.default, voice())
    assert [t["label"] for t in usual["turns"]] == ["R0", "R1"] and [u["machine"] for u in usual["used"]] == [None] and clock["t"] == DAY
    asked = live.turns(["R0", "R3", "R1"], voice())
    assert [t["label"] for t in asked["turns"]] == ["R0", "R3", "R1"] and asked["turns"][1]["machine"] == "Modal NVIDIA A10G"
    a10 = asked["used"][1]
    assert (a10["machine"], a10["says"], a10["cold"], a10["load_s"]) == ("a10", "Modal NVIDIA A10G", True, 12.0) and a10["end"] - a10["start"] == 40
    json.dumps(asked)


def test_a_machine_that_does_not_answer_is_a_sentence_on_its_cards_and_the_rest_answer(capsys):
    live = several({"t": DAY}, broken=True)
    out = live.turns(["R0", "R3"], voice())
    assert "error" not in out["turns"][0] and out["turns"][1] == {"label": "R3", "rung": "stub", "error": demo.NO_MACHINE, "machine": "an A10"}
    assert "path" not in json.dumps(out) and "nobody/should/see" in capsys.readouterr().err and len(out["used"]) == 2   # it was up, so it is charged


def test_the_page_asks_the_other_machines_only_for_what_is_ticked(tree, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")
    clock = {"t": DAY}
    budget = demo.Budget({}, cap_dollars=1.0, dollars_per_hour=0.59, window_s=300.0, cold_start_s=0.0, machines={"a10": (1.10, 120.0)})
    client = TestClient(demoweb.create_app(several(clock), budget, root=tree, machine="a test", code_env="TEST_DEMO_CODE",
                                           clock=lambda: clock["t"], clips=PICK, rows=ROWS, verbatim_run="runs/none.json"))
    status = client.get("/api/status").json()
    assert status["rungs"] == ["R0", "R1"] and status["optional"] == [{"label": "R3", "machine": "an A10", "wake_s": None, "dollars": 0.04,
                                                                          "streams": False}]
    send = lambda rungs: client.post("/api/live", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": rungs},
                                     headers={"X-Access-Code": "open sesame"}).json()
    usual = send("")
    assert [t["label"] for t in usual["turns"]] == ["R0", "R1"] and usual["charged"] == pytest.approx(300 / 3600 * 0.59, abs=1e-3)
    asked = send("R0,R1,R3")
    assert [t["label"] for t in asked["turns"]] == ["R0", "R1", "R3"]
    assert asked["charged"] == pytest.approx((40 + 120) / 3600 * 1.10, abs=8e-3)                         # the A10's turn and window; the T4 was up
    assert asked["woken"] == [{"machine": "Modal NVIDIA A10G", "wake_s": 40.0}] and budget.store["wake_s:a10"] == 40.0
    assert client.get("/api/status").json()["optional"][0]["wake_s"] == 40.0
    page = client.get("/").text
    assert "Also ask:" in page and "it wakes for this" in page


# --------------------------------------------------------------------------- #
# a full-duplex model: one call, not a turn
# --------------------------------------------------------------------------- #
class Answers:
    """A made-up full-duplex model: greets in the lead-in, answers some steps after the caller."""

    rung_id = "r5-made-up"

    def __init__(self, speak=(), text=None):
        self.speak, self.text, self.loaded = set(speak), dict(text or {}), 0

    def load(self):
        self.loaded += 1

    def begin(self, seed):
        self.k, self.rng, self.calls = 0, np.random.default_rng(seed), getattr(self, "calls", 0) + 1

    def step(self, frame):
        k, self.k = self.k, self.k + 1
        return (0.3 * self.rng.standard_normal(len(frame)) * (k in self.speak)).astype(np.float32), self.text.get(k)

    def end(self):
        pass


def test_a_recording_is_one_side_of_a_call_and_comes_back_measured_and_in_stereo():
    from s2slab import duplex
    caller = np.concatenate([np.zeros(8_000, np.float32), voice()[:24_000], np.zeros(8_000, np.float32)])     # speech from 0.5 to 2.0 s
    model = Answers(speak=list(range(5, 21)) + list(range(89, 101)), text={90: " Sure", 91: ",", 92: " I can"})
    live = demo.Live({"R5": "r5-made-up"}, get=lambda rung_id: demo.Call(model))
    assert live.load_all() >= 0 and model.loaded == 1 and live.unavailable == {}
    assert model.calls == 1 and model.k == 63                     # the throwaway call at load: 5 s of stream, 63 frames
    t = live.turns(["R5"], caller)["turns"][0]
    assert model.calls == 2
    assert (t["label"], t["rung"], t["kind"], t["transcript"]) == ("R5", "r5-made-up", "duplex", None)
    assert t["opened"] and t["reply_text"] == "Sure, I can" and t["lead_s"] == duplex.LEAD_S and t["overlap"] == 0
    assert 200 <= t["gap_ms"] <= 700 and t["stream_s"] == pytest.approx(4.8 + 2.5 + 8.0, abs=0.1) and t["rtf"] >= 0 and t["total_ms"] >= 0
    both, sr = sf.read(io.BytesIO(base64.b64decode(t["reply_wav"])))
    assert sr == 24_000 and both.shape[1] == 2 and len(both) / sr == pytest.approx(t["stream_s"], abs=0.2)
    lead = int(duplex.LEAD_S * sr)
    assert not both[:lead, 0].any() and np.abs(both[lead + 14_000:lead + 40_000, 0]).mean() > 0.01        # the caller, on the left, after the lead-in
    assert np.abs(both[int(7.3 * sr):int(7.9 * sr), 1]).mean() > 0.05                                   # the model's answer, on the right
    json.dumps(t)


def test_a_call_that_fails_is_a_sentence_and_the_registry_knows_the_duplex_models(capsys):
    class Falls(Answers):
        def step(self, frame):
            if self.calls > 1:                                    # the throwaway call at load goes through; the visitor's does not
                raise RuntimeError("CUDA out of memory at /a/path/nobody/should/see")
            return super().step(frame)

    live = demo.Live({"R5": "r5-made-up"}, get=lambda rung_id: demo.Call(Falls()))
    assert live.turn("R5", voice()) == {"label": "R5", "rung": "r5-made-up", "error": demo.NOT_ANSWERED}
    assert "nobody/should/see" in capsys.readouterr().err

    class Never(Answers):
        def step(self, frame):
            raise RuntimeError("no such weights")

    dead = demo.Live({"R5": "r5-made-up"}, get=lambda rung_id: demo.Call(Never()))
    assert dead.turn("R5", voice())["error"] == demo.NOT_LOADED    # it fell in the throwaway call: it never loaded
    call = demo.any_rung("r5-moshi")
    assert isinstance(call, demo.Call) and call.rung_id == "r5-moshi" and demo.any_rung("stub").rung_id == "stub"
    page = demo.PAGE.read_text(encoding="utf-8")
    assert "Not a conversation" in page and "you are on the left and the model on the right" in page


# --------------------------------------------------------------------------- #
# what else was registered and run
# --------------------------------------------------------------------------- #
def test_the_other_findings_are_read_from_their_committed_run_files_as_they_came_out():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    by = {f["registration"]: f for f in demo.findings(root)}
    assert list(by) == ["PR-003", "PR-010", "PR-009", "PR-012", "PR-013", "PR-014", "PR-015"]
    assert [f["file"] for f in by.values()] == list(demo.FINDINGS)
    assert {k: f["supported"] for k, f in by.items()} == {"PR-003": True, "PR-010": False, "PR-009": False, "PR-012": False,
                                                         "PR-013": True, "PR-014": False, "PR-015": True}
    assert by["PR-014"]["status"] == "not run" and "status" not in by["PR-013"]
    assert dict(by["PR-014"]["rows"])["setting 0.2"] == "16 of 30 turns cut off, a median wait of 1,480 ms"
    assert dict(by["PR-014"]["rows"])["the rule: the fewest cut-offs with a median wait of at most 1,000 ms"] == "no setting met it"
    s = dict(by["PR-015"]["rows"])
    assert s["first sound, the median: streamed against R4"] == "3,331 ms against 6,806 (0.49 of R4's)"
    assert s["first sound, p90"] == "3,728 ms against 11,784 (0.32)" and s["turns where playback ran dry"] == "0 of 200"
    assert s["the words, as Whisper hears the reply"] == "1.1% wrong against 1.3%" and by["PR-015"]["commit"] == "e2a4b4e"
    assert s["the whole reply, the median"] == "7,530 ms against 6,602 (1.14 times)" and s["R4 with its Talker compiled, not streamed"].startswith("7,010 ms")
    assert dict(by["PR-003"]["rows"])["after 1500 ms"] == "wrong on 8 of 20"
    assert dict(by["PR-003"]["rows"])["stop when the words are not a backchannel"] == "wrong on 1 of 20" and "not played" in by["PR-003"]["note"]
    assert by["PR-010"]["rows"][0][1] == "496 ms" and by["PR-010"]["rows"][1][1] == "1035 ms" and "further away" in by["PR-010"]["rows"][2][1]
    assert by["PR-010"]["rows"][2][1].startswith("+539 ms, 95% interval +402 to +685")
    s4 = dict(by["PR-009"]["rows"])["pitch offset within 3 Hz or 10%, at least 90%"]
    assert s4 == "169 of 192 (88.0%): failed" and dict(by["PR-009"]["rows"])["rate and pauses within 10%, every check"] == "384 of 384 (100.0%)"
    assert dict(by["PR-009"]["rows"])["pitch range in order, at least 28 of 32"] == "29 of 32 (90.6%)"      # the bar is severity's, not typed here
    assert all(v.endswith(": failed") for _, v in by["PR-012"]["rows"]) and "no clinical validity" in by["PR-012"]["note"]
    q = dict(by["PR-013"]["rows"])
    assert q["the 8-bit copy's pitch, against the full one"].startswith("-3.96 Hz on average") and "12 of 48" in q["the 8-bit copy's pitch, against the full one"]
    assert q["its words, by a recognizer"] == "10 errors in 820 words, against 8" and q["size"] == "92 MB against 326 MB"
    assert q["the 16-bit copy"].startswith("spoke on 42 of 48") and "slower" in q["speed on a laptop's CPU"]
    json.dumps(list(by.values()))
    assert "/" not in json.dumps([f["note"] for f in by.values()])                    # no path in what is said


def test_a_finding_whose_run_file_is_missing_is_left_out(tree):
    assert demo.findings(tree) == [] and demo.measured(tree, clips=PICK, rows=ROWS)["findings"] == []


# --------------------------------------------------------------------------- #
# one request a machine
# --------------------------------------------------------------------------- #
def test_the_door_is_asked_once_and_a_request_may_leave_the_voice_to_another(web):
    client, _, budget = web
    enter = lambda code: client.post("/api/enter", headers={"X-Access-Code": code} if code else {})
    assert enter(None).status_code == 401 and enter("wrong").status_code == 401 and budget.store["wrong_codes"] == [DAY]   # one guess, counted once
    assert enter("open sesame").json() == {"ok": True} and budget.spent(DAY) == 0.0                 # the door costs nothing
    send = lambda voice_too: client.post("/api/live", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": "R1", "voice_too": voice_too},
                                         headers={"X-Access-Code": "open sesame"}).json()
    assert send("0")["voice"] is None and [t["label"] for t in send("0")["turns"]] == ["R1"]
    pytest.importorskip("parselmouth")
    assert send("1")["voice"]["pitch_hz"] > 0
    page = client.get("/").text
    assert "api/enter" in page and "Each machine answers when it is ready" in page and "It wakes for this, and loads its model first." in page


def test_a_request_to_one_machine_does_not_wait_for_another(tree, monkeypatch):
    pytest.importorskip("fastapi")
    import threading
    import time

    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")

    class Slow(StubRung):
        def _respond(self, audio, sr, sink):
            time.sleep(0.8)                                       # the first machine, waking
            return super()._respond(audio, sr, sink)

    first = demo.Live({"R0": "stub"}, get=lambda rung_id: Slow())
    far = demo.Live({"R3": "stub"}, get=lambda rung_id: StubRung())
    live = demo.Several(first, {"a10": demo.Remote({"R3": "stub"}, lambda labels, samples: demo.answer(far, labels, samples))}, says={"a10": "an A10"})
    budget = demo.Budget({}, 1.0, 0.59, window_s=300.0, cold_start_s=0.0, machines={"a10": (1.10, 120.0)})
    app = demoweb.create_app(live, budget, root=tree, code_env="TEST_DEMO_CODE", clips=PICK, rows=ROWS, verbatim_run="runs/none.json")
    done = {}

    def send(rungs):
        r = TestClient(app).post("/api/live", files={"audio": ("turn", wav(1.0), "audio/wav")}, data={"rungs": rungs, "voice_too": "0"},
                                 headers={"X-Access-Code": "open sesame"})
        done[rungs] = (time.perf_counter(), r.status_code, [t["label"] for t in r.json()["turns"]])

    threads = [threading.Thread(target=send, args=(rungs,)) for rungs in ("R0", "R3")]
    threads[0].start(); time.sleep(0.2); threads[1].start()
    for t in threads:
        t.join()
    assert done["R0"][1:] == (200, ["R0"]) and done["R3"][1:] == (200, ["R3"])
    assert done["R3"][0] < done["R0"][0]                          # sent later, answered sooner: it did not wait for the first machine
    again = [threading.Thread(target=send, args=("R0",)) for _ in range(2)]
    t0 = time.perf_counter()
    for t in again:
        t.start()
    for t in again:
        t.join()
    assert time.perf_counter() - t0 > 1.5                         # two recordings on one machine: one after the other


# --------------------------------------------------------------------------- #
# the severity render, to be heard
# --------------------------------------------------------------------------- #
def speaks(text, voice_name):
    """A made-up synthesizer: the harmonic voice at Kokoro's rate, whatever the text."""
    return voice(sr=24_000)


def test_a_render_is_the_registrations_sentence_edited_to_its_score_and_measured_again():
    pytest.importorskip("parselmouth")
    from s2slab import severity
    ok = demo.render_choices()
    assert ok["voices"] == list(severity.TEST_VOICES) and ok["sentences"] == severity.TEST_SENTENCES
    assert ok["scores"] == {"depression": [0, 5, 10, 15, 20, 27], "mania": [0, 12, 20, 26, 40, 60]} and ok["scales"]["depression"] == "PHQ-9"
    r = demo.render_severity(speaks, "af_sarah", "t1", "depression", 15)
    t = severity.depression(15)
    assert r["target"] == {"f0_offset_hz": round(t.f0_offset_hz, 3), "f0_dev_scale": round(t.f0_dev_scale, 3), "rate": round(t.rate, 3),
                           "pause_scale": round(t.pause_scale, 3)}
    assert r["achieved"]["f0_offset_hz"] == pytest.approx(t.f0_offset_hz, abs=3) and r["achieved"]["rate"] == pytest.approx(t.rate, rel=0.1)
    assert r["rendered_s"] > r["neutral_s"] and r["text"] == severity.TEST_SENTENCES["t1"] and r["scale"] == "PHQ-9" and r["warning"] is None
    assert "not a diagnosis" in r["note"] and "No clinical validity" in r["note"]
    for key, title in (("neutral_wav", "synthetic neutral"), ("rendered_wav", "synthetic depression 15")):
        with sf.SoundFile(io.BytesIO(base64.b64decode(r[key]))) as f:
            assert f.samplerate == 24_000 and f.title == title and "SYNTHETIC SPEECH" in f.comment      # the file says what it is
    assert demo.render_severity(speaks, "am_adam", "t8", "mania", 60)["warning"] == demo.RENDER_WARNING
    json.dumps(r)


def test_nothing_but_the_registered_voices_sentences_and_scores_is_rendered():
    for asked in (("af_heart", "t1", "depression", 15), ("af_sarah", "d1", "depression", 15), ("af_sarah", "t1", "anxiety", 15),
                  ("af_sarah", "t1", "depression", 14), ("af_sarah", "Say anything I type.", "mania", 60)):
        with pytest.raises(ValueError, match="only the registered"):
            demo.render_severity(speaks, *asked)


def test_the_registered_run_is_read_beside_a_render():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    step = demo.registered_step(root, "af_sarah", "t1", "depression", 15)
    assert (step["registration"], step["supported"], step["file"]) == ("PR-009", False, "runs/severity/pr009.json")
    assert step["result"] == "4 of its five criteria met; pitch within 3 Hz or 10% in 169 of 192 checks, where it needed nine in ten"
    assert set(step["achieved"]) == {"f0_offset_hz", "f0_dev_scale", "rate", "pause_scale"} and step["achieved"]["rate"] == pytest.approx(0.722, abs=0.08)
    assert demo.registered_step(root, "af_sarah", "t1", "depression", 14) is None and demo.registered_step(root / "nowhere", "af_sarah", "t1", "depression", 15) is None


def test_the_page_renders_behind_the_door_and_charges_the_budget(tree, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("parselmouth")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")
    live = demo.Live({"R0": "stub"}, get=lambda rung_id: StubRung(), speaker=lambda: speaks)
    budget = demo.Budget({}, cap_dollars=1.0, dollars_per_hour=0.59, window_s=300.0, cold_start_s=0.0)
    client = TestClient(demoweb.create_app(live, budget, root=tree, code_env="TEST_DEMO_CODE", clock=lambda: DAY, clips=PICK, rows=ROWS,
                                           verbatim_run="runs/none.json"))
    assert client.get("/api/status").json()["render"]["voices"] == ["af_sarah", "am_adam"]
    ask = lambda code, **form: client.post("/api/render", data={"voice": "af_sarah", "sentence": "t2", "condition": "mania", "score": "20", **form},
                                           headers={"X-Access-Code": code} if code else {})
    assert ask(None).status_code == 401 and ask("wrong").status_code == 401 and budget.spent(DAY) == 0.0
    refused = ask("open sesame", sentence="Say anything I type.")
    assert refused.status_code == 400 and "only the registered" in refused.json()["detail"] and budget.spent(DAY) == 0.0
    ok = ask("open sesame").json()
    assert (ok["condition"], ok["score"], ok["scale"]) == ("mania", 20.0, "YMRS") and ok["rendered_s"] < ok["neutral_s"]
    assert ok["registered"] is None and ok["charged"] == pytest.approx(300 / 3600 * 0.59, abs=1e-3)       # no run file in this tree
    page = client.get("/").text
    assert 'id="render" class="card" hidden' in page and "measured in the registered run" in page and "does not say a sentence the same way twice" in page
    no_render = TestClient(demoweb.create_app(demo.Remote({"R0": "stub"}, lambda labels, samples: {}), budget, root=tree,
                                              code_env="TEST_DEMO_CODE", clips=PICK, rows=ROWS, verbatim_run="runs/none.json"))
    assert no_render.get("/api/status").json()["render"] is None
    assert no_render.post("/api/render", data={"voice": "af_sarah", "sentence": "t2", "condition": "mania", "score": "20"},
                          headers={"X-Access-Code": "open sesame"}).status_code == 404


def test_the_measured_part_shows_r4_streamed_from_the_pr015_run():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    rows = {r["label"]: r for r in demo.measured(root)["rungs"]}
    s = rows["R4 streamed"]
    assert s["file"] == "runs/r4-gpu/pr015.json" and s["turns_in_run"] == 200 and s["commit"] == "e2a4b4e"
    assert s["kind"] == "turn" and 2500 < s["first_audio_p50_ms"] < 4000                 # the stream, not R4 beside it
    t = s["turns"][0]
    assert t["clip"] == demo.CLIPS[0] and t["transcript"] is None and t["stages_ms"] == {} and t["reply_text"]


def test_the_l40s_pair_makes_both_rungs_on_one_model():
    from s2slab.r4stream import FastOmni, FastSpeechToSpeech, StreamingSpeechToSpeech, pair
    model = FastOmni()
    get = pair(model)
    fast, stream = get("r4-gpu-fast"), get("r4-gpu-stream")
    assert isinstance(fast, FastSpeechToSpeech) and isinstance(stream, StreamingSpeechToSpeech)
    assert fast.model is model and stream.model is model                               # one set of weights
    with pytest.raises(KeyError, match="pair"):
        get("r4-gpu")


# --------------------------------------------------------------------------- #
# a turn heard as it is made (M32a)
# --------------------------------------------------------------------------- #
class Streamer(StubRung):
    """A streaming rung, as r4-gpu-stream is to the demo: three pieces, each after a pause, then the turn."""
    rung_id, streams, out_sr = "streamer", True, 24_000

    def __init__(self, fail=False, pause_s=0.03):
        self.fail, self.pause_s = fail, pause_s

    def _respond(self, audio, sr, sink):
        import time
        from s2slab.rungs import Reply
        for k in range(3):
            time.sleep(self.pause_s)
            self.emit(np.full(2400, 0.1 * (k + 1), dtype=np.float32))
            if self.fail and k == 1:
                raise RuntimeError("broke mid-reply at /a/path/nobody/should/see")
        return Reply(None, 24_000, None, "Hello there.")


def test_pcm16_is_16_bit_little_endian_and_clipped():
    raw = base64.b64decode(demo.pcm16(np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32)))
    assert np.frombuffer(raw, dtype="<i2").tolist() == [0, 16384, -16384, 32767]


def test_a_streaming_turn_hands_each_piece_out_before_the_turn_is_over():
    import time
    rung, events, arrived = Streamer(), [], []
    for ev in demo.stream_turn(rung, "R4 streamed", "r4-gpu-stream", np.zeros(16_000, dtype=np.float32),
                               record=lambda label, t: {"label": label, "reply_text": t.reply_text, "chunks": len(t.chunks)}):
        events.append(ev)
        arrived.append(time.perf_counter())
    assert events[0] == {"label": "R4 streamed", "start": True, "sr": 24_000}
    pieces = [e for e in events if "piece" in e]
    assert len(pieces) == 3 and events[-1] == {"label": "R4 streamed", "done": {"label": "R4 streamed", "reply_text": "Hello there.", "chunks": 3}}
    assert [p["made_ms"] for p in pieces] == sorted(p["made_ms"] for p in pieces) and pieces[0]["made_ms"] > 20
    levels = [np.frombuffer(base64.b64decode(p["piece"]), dtype="<i2")[0] / 32768 for p in pieces]
    assert levels == pytest.approx([0.1, 0.2, 0.3], abs=1e-3)
    assert arrived[-1] - arrived[1] > 0.04                                  # the first piece came well before the end
    assert "emit" not in vars(rung)                                          # unwrapped after the turn


def test_a_streaming_turn_that_fails_says_so_and_unwraps(capsys):
    rung = Streamer(fail=True)
    events = list(demo.stream_turn(rung, "R4 streamed", "r4-gpu-stream", np.zeros(16_000, dtype=np.float32)))
    assert sum("piece" in e for e in events) == 2
    assert events[-1] == {"label": "R4 streamed", "done": {"label": "R4 streamed", "rung": "r4-gpu-stream", "error": demo.NOT_ANSWERED}}
    assert "emit" not in vars(rung) and "path" not in json.dumps(events) and "broke mid-reply" in capsys.readouterr().err


def test_a_machine_streams_its_streaming_rung_first_and_answers_the_other_whole():
    live = demo.Live({"R4": "r4-gpu-fast", "R4 streamed": "r4-gpu-stream"},
                     get=lambda rung_id: Streamer() if rung_id == "r4-gpu-stream" else StubRung())
    events = list(live.stream(["R4", "R4 streamed"], np.zeros(16_000, dtype=np.float32)))
    assert events[0] == {"cold": True} and [e.get("label") for e in events[1:]] == ["R4 streamed"] * 5 + ["R4"]
    streamed, whole = events[-2]["done"], events[-1]["done"]
    assert streamed["reply_text"] == "Hello there." and streamed["reply_wav"] is None and streamed["first_audio_ms"] > 0
    assert whole["reply_wav"] is not None and whole["rung"] == "r4-gpu-fast"               # the whole one keeps its WAV
    assert list(live.stream(["R4"], np.zeros(16_000, dtype=np.float32)))[0] == {"cold": False}


@pytest.fixture
def streaming_web(tree, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")
    clock = {"t": DAY}
    budget = demo.Budget({}, cap_dollars=3.0, dollars_per_hour=0.59, machines={"l40s": (1.95, 120.0)})
    first = demo.Live({"R0": "stub"}, get=lambda rung_id: StubRung())
    l40s = demo.Live({"R4": "r4-gpu-fast", "R4 streamed": "r4-gpu-stream"},
                     get=lambda rung_id: Streamer() if rung_id == "r4-gpu-stream" else StubRung())
    live = demo.Several(first, {"l40s": l40s}, says={"l40s": "an L40S"})
    app = demoweb.create_app(live, budget, root=tree, machine="a test", code_env="TEST_DEMO_CODE", clock=lambda: clock["t"],
                             clips=PICK, rows=ROWS, verbatim_clips=("dis-00", "dis-01"), verbatim_run="runs/v.json")
    return TestClient(app), clock, budget


def test_the_stream_passes_the_door_streams_the_machines_rungs_and_charges_it(streaming_web):
    client, clock, budget = streaming_web
    status = client.get("/api/status").json()
    assert [(o["label"], o["streams"]) for o in status["optional"]] == [("R4", False), ("R4 streamed", True)]
    send = lambda code, rungs: client.post("/api/stream", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": rungs},
                                          headers={"X-Access-Code": code} if code else {})
    assert send(None, "R4 streamed").status_code == 401 and budget.spent(DAY) == 0.0
    assert send("open sesame", "R0").status_code == 400                     # the first machine does not stream
    assert send("open sesame", "R4").status_code == 400                     # nor does a machine asked only for a whole turn
    r = send("open sesame", "R4 streamed,R4")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0] == {"cold": True} and events[1] == {"label": "R4 streamed", "start": True, "sr": 24_000}
    assert sum("piece" in e for e in events) == 3
    done = [e["done"] for e in events if "done" in e]
    assert [d["label"] for d in done] == ["R4 streamed", "R4"] and all(d["machine"] == "an L40S" for d in done)
    summary = events[-1]
    assert summary["charged"] > 0 and summary["woken"][0]["machine"] == "an L40S" and "billed_until:l40s" in budget.store
    assert send("open sesame", "R4 streamed").status_code == 200            # the machine's lock was let go


def test_a_stream_whose_machine_fails_says_so_and_still_charges(tree, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import demoweb
    monkeypatch.setenv("TEST_DEMO_CODE", "open sesame")
    budget = demo.Budget({}, cap_dollars=3.0, dollars_per_hour=0.59, machines={"l40s": (1.95, 120.0)})

    class Gone:
        offered = {"R4 streamed": "r4-gpu-stream"}

        def turns(self, labels, audio):
            raise RuntimeError("no")

        def stream(self, labels, audio):
            raise RuntimeError("the container did not start: /a/path/nobody/should/see")

    live = demo.Several(demo.Live({"R0": "stub"}, get=lambda rung_id: StubRung()), {"l40s": Gone()}, says={"l40s": "an L40S"})
    client = TestClient(demoweb.create_app(live, budget, root=tree, code_env="TEST_DEMO_CODE", clock=lambda: DAY, clips=PICK, rows=ROWS,
                                           verbatim_clips=("dis-00", "dis-01"), verbatim_run="runs/v.json"))
    r = client.post("/api/stream", files={"audio": ("turn", wav(2.0), "audio/wav")}, data={"rungs": "R4 streamed"},
                    headers={"X-Access-Code": "open sesame"})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0] == {"error": "the machine these rungs run on did not answer"} and "path" not in r.text
    assert events[-1]["charged"] > 0 and budget.spent(DAY) > 0
