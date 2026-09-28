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
    assert set(app.ABOUT) <= set(app.OFFERED) and "large-v3-turbo" in app.ABOUT["R2"]


def test_the_web_container_is_sent_what_the_measured_part_reads_and_no_more(app):
    assert list(app.RUN_FILES) == [rel for _, _, rel in demo.MEASURED] + [demo.VERBATIM_RUN, demo.PROSODY_RUN]
    assert app.CLIPS == demo.CLIPS + demo.VERBATIM_CLIPS
    for rel in app.RUN_FILES:
        assert (ROOT / rel).exists(), rel
    assert (app.SECRET, app.CAP_DOLLARS, app.WINDOW_S, app.T4_DOLLARS_PER_HOUR) == ("s2s-lab-demo", 1.0, 300, 0.59)


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
    assert client.get("/").status_code == 200 and len(client.get("/api/measured").json()["rungs"]) == 7
    send = lambda code, rungs="": client.post("/api/live", files={"audio": ("turn", wav(), "audio/wav")}, data={"rungs": rungs},
                                              headers={"X-Access-Code": code} if code else {})
    assert send(None).status_code == 401 and send("wrong").status_code == 401 and calls == []      # the T4 stays asleep
    ok = send("open sesame").json()
    assert calls == [(list(app.OFFERED), 2 * 16_000 * 4)] and ok["machine"] == "Modal Tesla T4" and ok["cold"]
    assert [t["label"] for t in ok["turns"]] == list(app.OFFERED) and ok["load_s"] == 41.0
    assert 80 < ok["voice"]["pitch_hz"] < 400 and ok["voice"]["voiced_s"] > 1            # measured beside the door, not on the T4
    assert ok["charged"] == pytest.approx(300 / 3600 * 0.59, abs=0.002) and store["wake_s"] >= 0   # the window, and the wall time
    assert app.CAP_DOLLARS / ok["charged"] > 15                                                    # bursts of use in a day's dollar
    assert client.get("/api/status").json()["wake_s"] == store["wake_s"]
    store[f"dollars:{demo.Budget.day(__import__('time').time())}"] = 1.0                           # the day's dollar is spent
    assert send("open sesame").status_code == 429 and len(calls) == 1


def test_a_t4_that_does_not_answer_is_a_sentence_not_a_fault_and_is_still_charged(app, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("DEMO_ACCESS_CODE", "open sesame")

    def call(labels, raw):
        raise RuntimeError("the container did not start: /a/path/nobody/should/see")

    store = {}
    r = TestClient(app.make_app(call, store, root=ROOT)).post("/api/live", files={"audio": ("turn", wav(), "audio/wav")},
                                                               headers={"X-Access-Code": "open sesame"})
    assert r.status_code == 503 and "path" not in r.text and "measured part" in r.json()["detail"]
    spent = [v for k, v in store.items() if k.startswith("dollars:")]
    assert len(spent) == 1 and spent[0] > 0                  # a T4 that started and failed was still up: the cap counts it
