"""The demo's Modal app (modal_app/demo.py) without Modal: what it sends, what its image holds, and
its web app with a function in the T4's place and a dict in the spend counter's."""
import importlib.util
import io
import pathlib

import numpy as np
import pytest
import soundfile as sf

from s2slab import demo, r2verbatim
from s2slab.rungs import RUNGS, StubRung, get
from test_severity import voice

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = 1_790_000_000.0


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_demo", ROOT / "modal_app" / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wav(seconds=2.0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.resize(voice(sr=16_000), int(seconds * 16_000)), 16_000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_the_image_holds_every_model_the_four_rungs_load_and_the_export_is_pr011s(app):
    assert list(app.OFFERED) == ["R0", "R1", "R2", "R2 verbatim"] and set(app.OFFERED.values()) <= set(RUNGS)
    for label, rung_id in app.OFFERED.items():
        rung = get(rung_id)
        assert rung.asr.device == "cuda" and rung.llm.model_name in app.MODELS
        assert f"voices/{rung.tts.voice}.pt" in app.MODELS["hexgrad/Kokoro-82M"]
        if label != "R2 verbatim":
            assert f"Systran/faster-whisper-{rung.asr.model_name}" in app.MODELS
    assert app.EXPORT_NAME == r2verbatim.EXPORT_NAME and app.EXPORT == f"/ft2/export/{r2verbatim.EXPORT_NAME}/ct2"
    assert set(app.ABOUT) <= set(app.OFFERED) | {"R3", "R4", "R4 streamed", "R5"} and "large-v3-turbo" in app.ABOUT["R2"]


def test_the_web_container_is_sent_what_the_measured_part_reads_and_no_more(app):
    assert list(app.RUN_FILES) == list(dict.fromkeys([rel for _, _, rel in demo.MEASURED] + [demo.VERBATIM_RUN, demo.PROSODY_RUN]
                                                    + list(demo.FINDINGS)))            # each file once
    assert app.CLIPS == demo.CLIPS + demo.VERBATIM_CLIPS
    for rel in app.RUN_FILES:
        assert (ROOT / rel).exists(), rel
    assert (app.SECRET, app.CAP_DOLLARS, app.WINDOW_S, app.T4_DOLLARS_PER_HOUR) == ("s2s-lab-demo", 3.0, 300, 0.59)


def test_the_gpu_is_reached_only_with_the_code_and_inside_the_budget(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")
    calls, store = [], {}
    rungs = demo.Live({label: "stub" for label in app.OFFERED}, get=lambda rung_id: StubRung())

    def call(labels, raw):                                   # the T4's end, here
        calls.append((labels, len(raw)))
        return dict(demo.answer(rungs, labels, raw), machine="Modal Tesla T4", load_s=41.0)

    client = TestClient(app.make_app(call, store, root=ROOT))
    status = client.get("/api/status").json()
    assert status["rungs"] == list(app.OFFERED) and status["about"] == app.ABOUT and status["wake_s"] is None
    assert client.get("/").status_code == 200 and len(client.get("/api/measured").json()["rungs"]) == 8
    assert [f["registration"] for f in client.get("/api/measured").json()["findings"]] == ["PR-003", "PR-010", "PR-009", "PR-012",
                                                                                          "PR-013", "PR-014", "PR-015", "PR-016"]
    send = lambda code, rungs="": client.post("/api/live", files={"audio": ("turn", wav(), "audio/wav")}, data={"rungs": rungs},
                                              headers={"X-Access-Code": code} if code else {})
    assert send(None).status_code == 401 and send("wrong").status_code == 401 and calls == []      # the T4 stays asleep
    ok = send("open sesame").json()
    assert calls == [(list(app.OFFERED), 2 * 16_000 * 4)] and ok["machine"] == "Modal Tesla T4" and ok["cold"]
    assert [t["label"] for t in ok["turns"]] == list(app.OFFERED) and ok["load_s"] == 41.0
    assert 80 < ok["voice"]["pitch_hz"] < 400 and ok["voice"]["voiced_s"] > 1            # measured beside the door, not on the T4
    assert ok["charged"] == pytest.approx(300 / 3600 * 0.59, abs=0.002) and store["wake_s"] >= 0   # the window, and the wall time
    assert app.CAP_DOLLARS / ok["charged"] > 45                                                    # bursts of use in a day's dollars
    assert client.get("/api/status").json()["wake_s"] == store["wake_s"]
    store[f"dollars:{demo.Budget.day(__import__('time').time())}"] = app.CAP_DOLLARS               # the day's dollars are spent
    assert send("open sesame").status_code == 429 and len(calls) == 1


def test_a_t4_that_does_not_answer_is_a_sentence_on_its_cards_and_is_still_charged(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")

    def call(labels, raw):
        raise RuntimeError("the container did not start: /a/path/nobody/should/see")

    store = {}
    r = TestClient(app.make_app(call, store, root=ROOT)).post("/api/live", files={"audio": ("turn", wav(), "audio/wav")},
                                                               headers={"X-Access-Code": "open sesame"})
    assert r.status_code == 200 and "path" not in r.text and len(r.json()["turns"]) == 4
    assert all(t["error"] == demo.NO_MACHINE for t in r.json()["turns"]) and r.json()["voice"]["pitch_hz"]     # the voice card is still there
    spent = [v for k, v in store.items() if k.startswith("dollars:")]
    assert len(spent) == 1 and spent[0] > 0                  # a T4 that started and failed was still up: the cap counts it


def test_r3_r4_and_r5_are_on_machines_of_their_own_that_hold_their_models(app):
    from s2slab import r5
    assert {name: o["offered"] for name, o in app.OTHERS.items()} == {"a10": {"R3": "r3-gpu"},
                                                                      "l40s": {"R4": "r4-gpu-fast", "R4 streamed": "r4-gpu-stream-sized"},
                                                                      "moshi": {"R5": "r5-moshi"}}
    assert list(app.MODELS_MOSHI) == [r5.MOSHI_REPO] and "r5-moshi" in r5.MODELS and app.OTHERS["moshi"]["rate"] == 1.10
    assert (app.OTHERS["a10"]["rate"], app.OTHERS["l40s"]["rate"], app.WINDOW_BIG_S) == (1.10, 1.95, 120)
    r3, r4, r4s = get("r3-gpu"), get("r4-gpu-fast"), get("r4-gpu-stream-sized")
    assert r3.llm.model_name in app.MODELS_A10 and f"voices/{r3.tts.voice}.pt" in app.MODELS_A10["hexgrad/Kokoro-82M"]
    assert r4.model.model_name in app.MODELS_L40S and r4s.model.model_name in app.MODELS_L40S
    assert set(app.ABOUT) == set(app.OFFERED) - {"R0", "R1"} | {"R3", "R4", "R4 streamed", "R5"} and "no transcript" in app.ABOUT["R3"]
    assert "2.6 s" in app.ABOUT["R4 streamed"] and "PR-016" in app.ABOUT["R4 streamed"] and "1.03" in app.ABOUT["R4"]
    assert "may describe you" in app.ABOUT["R3"] and "nothing here checks" in app.ABOUT["R3"]


def test_the_bigger_machines_wake_only_for_a_visitor_who_asks(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")
    woke, store = [], {}

    def machine(name, offered):
        rungs = demo.Live({label: "stub" for label in offered}, get=lambda rung_id: StubRung())

        def call(labels, raw):
            woke.append(name)
            return dict(demo.answer(rungs, labels, raw), machine=f"Modal {name}", load_s=1.0)
        return call

    client = TestClient(app.make_app(machine("t4", app.OFFERED), store, root=ROOT,
                                     others={name: machine(name, o["offered"]) for name, o in app.OTHERS.items()}))
    status = client.get("/api/status").json()
    assert status["rungs"] == list(app.OFFERED) and [(o["label"], o["machine"], o["dollars"]) for o in status["optional"]] == [
        ("R3", "an A10", 0.04), ("R4", "an L40S", 0.07), ("R4 streamed", "an L40S", 0.07),
        ("R5", "a second A10", 0.04)]                                                      # what each one's idle window costs
    send = lambda rungs: client.post("/api/live", files={"audio": ("turn", wav(), "audio/wav")}, data={"rungs": rungs},
                                     headers={"X-Access-Code": "open sesame"})
    assert send("").status_code == 200 and woke == ["t4"]                                  # a recording alone wakes the T4 only
    ok = send(",".join(list(app.OFFERED) + ["R3"])).json()
    assert sorted(woke[1:]) == ["a10", "t4"] and [t["label"] for t in ok["turns"]][-1] == "R3" and ok["turns"][-1]["machine"] == "Modal a10"
    assert send("R4").json()["turns"][0]["label"] == "R4" and woke[-1] == "l40s" and "billed_until:l40s" in store


def test_the_t4_holds_pr009s_voices_and_the_render_reaches_it_through_the_door(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s2slab import severity
    for name in ("af_heart", *severity.TEST_VOICES):
        assert f"voices/{name}.pt" in app.MODELS["hexgrad/Kokoro-82M"], name
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")
    asked, store = [], {}

    def render(voice, sentence, condition, score):                 # the T4's end, here
        asked.append((voice, sentence, condition, score))
        return {"voice": voice, "sentence": sentence, "condition": condition, "score": score, "misses": []}

    client = TestClient(app.make_app(lambda labels, raw: {}, store, root=ROOT, render=render))
    assert client.get("/api/status").json()["render"]["scores"]["mania"] == [0, 12, 20, 26, 40, 60]
    form = {"voice": "am_adam", "sentence": "t3", "condition": "depression", "score": "27"}
    assert client.post("/api/render", data=form, headers={"X-Access-Code": "wrong"}).status_code == 401 and asked == []
    ok = client.post("/api/render", data=form, headers={"X-Access-Code": "open sesame"}).json()
    assert asked == [("am_adam", "t3", "depression", 27.0)] and ok["charged"] > 0
    assert ok["registered"]["registration"] == "PR-009" and ok["registered"]["achieved"]["rate"] == pytest.approx(0.5, abs=0.06)   # from the run file
    assert TestClient(app.make_app(lambda labels, raw: {}, {}, root=ROOT)).get("/api/status").json()["render"] is None


def test_both_r4_cards_wake_one_machine_once(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")
    woke, store = [], {}

    def machine(name, offered):
        rungs = demo.Live({label: "stub" for label in offered}, get=lambda rung_id: StubRung())

        def call(labels, raw):
            woke.append((name, list(labels)))
            return dict(demo.answer(rungs, labels, raw), machine=f"Modal {name}", load_s=1.0)
        return call

    client = TestClient(app.make_app(machine("t4", app.OFFERED), store, root=ROOT,
                                     others={name: machine(name, o["offered"]) for name, o in app.OTHERS.items()}))
    ok = client.post("/api/live", files={"audio": ("turn", wav(), "audio/wav")}, data={"rungs": "R4,R4 streamed"},
                     headers={"X-Access-Code": "open sesame"}).json()
    assert ("l40s", ["R4", "R4 streamed"]) in woke and sum(name == "l40s" for name, _ in woke) == 1     # one request, both cards
    assert [t["label"] for t in ok["turns"]][-2:] == ["R4", "R4 streamed"]


def test_the_l40s_streams_and_the_web_app_reaches_it(app, monkeypatch):
    pytest.importorskip("fastapi")
    import json

    from fastapi.testclient import TestClient
    assert hasattr(app.Speaks, "stream")
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")
    l40s = demo.Live({"R4": "stub", "R4 streamed": "stub"}, get=lambda rung_id: StubRung())
    streamed = []

    def stream(labels, raw):                                  # the L40S's end, as Speaks.stream would yield it
        streamed.append(list(labels))
        for ev in demo.answer_stream(l40s, labels, raw):
            yield dict(ev, machine="Modal L40S", load_s=1.0) if "cold" in ev else ev

    client = TestClient(app.make_app(lambda labels, raw: {}, {}, root=ROOT,
                                     others={name: (lambda labels, raw: {}) for name in app.OTHERS},
                                     streams={"l40s": stream}))
    status = client.get("/api/status").json()
    assert [(o["label"], o["streams"]) for o in status["optional"] if o["machine"] == "an L40S"] == [("R4", False), ("R4 streamed", True)]
    r = client.post("/api/stream", files={"audio": ("turn", wav(), "audio/wav")}, data={"rungs": "R4 streamed,R4"},
                    headers={"X-Access-Code": "open sesame"})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert streamed == [["R4 streamed", "R4"]] and events[0]["cold"] and events[-1]["charged"] > 0
    assert {e["done"]["label"] for e in events if "done" in e} == {"R4", "R4 streamed"}
