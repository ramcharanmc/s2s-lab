"""
The demo's web app (FastAPI) around s2slab.demo: the page, the measured data and its clips, the
status, and the live turn. Needs requirements/demo.txt; nothing else in the library imports it.

    PYTHONPATH=src DEMO_ACCESS_CODE=... python -m uvicorn s2slab.demoweb:laptop --factory --port 8000

No `from __future__ import annotations` here: FastAPI reads the handlers' annotations as objects.
"""

import os
import pathlib
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .demo import CLIPS, CODE_ENV, MAX_CLIP_S, MAX_UPLOAD_BYTES, MEASURED, PAGE, SR, Budget, Live, code_ok, decode, measured


def create_app(live: Live, budget: Budget, root=".", machine: str = "this machine", code_env: str = CODE_ENV, clock=time.time,
               clips=CLIPS, rows=MEASURED, **panel):
    """The FastAPI app: the page, the measured data and its clips, the status, and the live turn."""
    root = pathlib.Path(root)
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
        return {"machine": machine, "rungs": list(live.offered), "code_needed": True, "code_set": bool(os.environ.get(code_env)),
                "budget_left": round(budget.left(now), 3), "budget_cap": budget.cap, "open": budget.allows(now),
                "max_clip_s": MAX_CLIP_S, "keeps": "nothing: the recording is decoded in memory and discarded"}

    @app.post("/api/live")
    async def api_live(audio: UploadFile = File(...), rungs: str = Form(""), x_access_code: str = Header(default="")):
        start = clock()
        if not code_ok(x_access_code, os.environ.get(code_env)):
            raise HTTPException(401, "the live part needs an access code")
        if not budget.allows(start):
            raise HTTPException(429, f"today's budget of ${budget.cap:.2f} is spent; the measured part is still here")
        wanted = [r for r in rungs.split(",") if r] or list(live.offered)
        unknown = [r for r in wanted if r not in live.offered]
        if unknown:
            raise HTTPException(400, f"no rung {', '.join(unknown)} here")
        try:
            clip = decode(await audio.read(MAX_UPLOAD_BYTES + 1))
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        turns = [live.turn(label, clip) for label in wanted]
        dollars = budget.charge(start, clock())
        return JSONResponse({"machine": machine, "clip_s": round(len(clip) / SR, 2), "turns": turns, "charged": round(dollars, 4),
                             "budget_left": round(budget.left(clock()), 3)})

    return app


def laptop():
    """The demo on the laptop's CPU, for development: uvicorn's --factory calls this."""
    from . import r2verbatim                                    # here, not at the top: it imports the model classes
    offered = {"R0": "r0", "R1": "r1", "R2": "r2"}
    if r2verbatim.available():
        offered["R2 verbatim"] = "r2-verbatim"
    return create_app(Live(offered), Budget({}, cap_dollars=1.0, dollars_per_hour=0.0),
                      root=os.environ.get("DEMO_ROOT", "."), machine="a laptop CPU (development)")
