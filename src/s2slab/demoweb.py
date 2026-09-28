"""
The demo's web app (FastAPI) around s2slab.demo: the page, the measured data and its clips, the
status, and the live turn. Needs requirements/demo.txt; nothing else in the library imports it.

    PYTHONPATH=src DEMO_ACCESS_CODE=... python -m uvicorn s2slab.demoweb:laptop --factory --port 8000

No `from __future__ import annotations` here: FastAPI reads the handlers' annotations as objects.
No handler is `async`: the spend counter (a modal.Dict in the deployment) and the rungs are blocking
calls, and FastAPI runs a plain function on a worker thread, so the page is served meanwhile.
"""

import os
import pathlib
import sys
import threading
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .demo import (CLIPS, CODE_ENV, MAX_CLIP_S, MAX_UPLOAD_BYTES, MEASURED, MIN_CODE_LEN, PAGE, SR, Budget, Door, Live, code_ok,
                   code_usable, decode, measured, voice)


def create_app(live: Live, budget: Budget, root=".", machine: str = "this machine", code_env: str = CODE_ENV, clock=time.time,
               clips=CLIPS, rows=MEASURED, about=None, **panel):
    """The FastAPI app: the page, the measured data and its clips, the status, and the live turn.
    `live` is a Live, a Remote or a Several; `about` is a sentence a rung's card carries, by label."""
    usual = list(getattr(live, "default", live.offered))           # what answers a recording when nothing is asked for
    asked_for = dict(getattr(live, "optional", {}))                # label -> the machine it wakes
    root = pathlib.Path(root)
    busy = {name: threading.Lock() for name in {None, *asked_for.values()}}     # one recording at a time, on each machine

    def enter(given: str, now: float) -> None:
        """The door: the lock on wrong codes, the code, the day's budget. Raises what the visitor is told."""
        if door.locked_s(now) > 0:                                 # before the code is looked at: a locked door tells nothing
            raise HTTPException(429, f"too many wrong codes; the live part opens again in {door.locked_s(now) / 60:.0f} min")
        if not code_ok(given, os.environ.get(code_env)):
            if given:                                              # a request with no code is not a guess
                door.wrong(now)
            raise HTTPException(401, "the live part needs an access code")
        if not budget.allows(now):
            raise HTTPException(429, f"today's budget of ${budget.cap:.2f} is spent; the measured part is still here")
    door = Door(budget.store)
    app = FastAPI(title="s2s-lab demo", docs_url=None, redoc_url=None, openapi_url=None)
    data = measured(root, clips, rows, **panel)
    clips = set(data["audio"])

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/measured")
    def api_measured():
        return data

    @app.get("/api/clip/{clip_id}")
    def api_clip(clip_id: str):
        if clip_id not in clips:                                   # only the clips the page shows; never a path from outside
            raise HTTPException(404, "no such clip")
        return FileResponse(root / "evalset" / "audio" / f"{clip_id}.wav", media_type="audio/wav")

    @app.get("/api/status")
    def api_status():
        now = clock()
        code = os.environ.get(code_env)
        return {"machine": machine, "rungs": usual, "code_needed": True, "code_set": code_usable(code),
                "optional": [{"label": label, "machine": live.says[name], "wake_s": budget.store.get(f"wake_s:{name}"),
                              "dollars": round(budget.machines[name][0] * budget.machines[name][1] / 3600, 2)}   # its idle window alone
                             for label, name in asked_for.items()],
                "code_short": bool(code) and not code_usable(code), "min_code_len": MIN_CODE_LEN, "locked_s": round(door.locked_s(now)),
                "budget_left": round(budget.left(now), 3), "budget_cap": budget.cap, "open": budget.allows(now),
                "about": about or {}, "wake_s": budget.store.get("wake_s"),
                "max_clip_s": MAX_CLIP_S, "keeps": "nothing: the recording is decoded in memory and discarded"}

    @app.post("/api/enter")
    def api_enter(x_access_code: str = Header(default="")):
        """The door alone, asked once before a recording is sent to several machines at once: a
        wrong code is then one wrong code, not one for each machine."""
        enter(x_access_code, clock())
        return {"ok": True}

    @app.post("/api/live")
    def api_live(audio: UploadFile = File(...), rungs: str = Form(""), voice_too: str = Form("1"),
                 x_access_code: str = Header(default="")):
        start = clock()
        enter(x_access_code, start)
        wanted = [r for r in rungs.split(",") if r] or usual
        unknown = [r for r in wanted if r not in live.offered]
        if unknown:
            raise HTTPException(400, f"no rung {', '.join(unknown)} here")
        try:
            clip = decode(audio.file.read(MAX_UPLOAD_BYTES + 1))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        carried = None
        if voice_too != "0":                                       # once a recording: the page asks it of one request
            try:
                carried = voice(clip)                              # here, not with the rungs: it is shown even if they fail
            except Exception as e:                                 # a measurement is not worth a turn
                print(f"the voice was not measured: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        mine = sorted({asked_for.get(label) for label in wanted}, key=str)      # the machines this request uses, in one order
        held = [busy[name] for name in mine]
        try:
            for lock in held:
                lock.acquire()
            try:
                out = live.turns(wanted, clip)
            finally:
                for lock in reversed(held):
                    lock.release()
        except Exception as e:
            print(f"the rungs did not answer: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            budget.charge(start, clock())                          # a machine that woke and failed was still up
            raise HTTPException(503, "the rungs did not answer; the measured part is still here") from None
        end = clock()
        woken = []
        if "used" in out:                                          # more than one machine: each charged for its own time
            dollars = 0.0
            for u in out["used"]:
                dollars += budget.charge(u["start"], u["end"], u["machine"])
                if u["cold"]:
                    key = f"wake_s:{u['machine']}" if u["machine"] else "wake_s"
                    budget.store[key] = round(max(0.0, u["end"] - u["start"] - u["busy_s"]), 1)
                    woken.append({"machine": u["says"] or machine, "wake_s": budget.store[key]})
        else:
            dollars = budget.charge(start, end)
            if out.get("cold"):                                    # what the wait held beside the turns themselves
                busy_s = sum(t.get("total_ms", 0.0) for t in out["turns"]) / 1000
                budget.store["wake_s"] = round(max(0.0, end - start - busy_s), 1)
                woken.append({"machine": out.get("machine") or machine, "wake_s": budget.store["wake_s"]})
        return JSONResponse({"machine": out.get("machine") or machine, "clip_s": round(len(clip) / SR, 2), "turns": out["turns"],
                             "voice": carried, "woken": woken,
                             "cold": bool(out.get("cold")), "wake_s": budget.store.get("wake_s") if out.get("cold") else None,
                             "load_s": out.get("load_s"), "charged": round(dollars, 4), "budget_left": round(budget.left(clock()), 3)})

    return app


def laptop():
    """The demo on the laptop's CPU, for development: uvicorn's --factory calls this."""
    from . import r2verbatim                                    # here, not at the top: it imports the model classes
    offered = {"R0": "r0", "R1": "r1", "R2": "r2"}
    if r2verbatim.available():
        offered["R2 verbatim"] = "r2-verbatim"
    return create_app(Live(offered), Budget({}, cap_dollars=1.0, dollars_per_hour=0.0),
                      root=os.environ.get("DEMO_ROOT", "."), machine="a laptop CPU (development)")
