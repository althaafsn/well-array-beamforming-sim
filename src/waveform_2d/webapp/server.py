"""
FastAPI server that streams the live wave field to the browser over a WebSocket.

The simulation runs server-side (pure Python/NumPy); each frame the float field
is mapped to fixed-scale RGBA bytes and pushed to the client, which paints them
onto a canvas. Control messages (drips, scene, parameters) flow back over the
same socket.

Run locally:
    uvicorn waveform_2d.webapp.server:app --reload
or:
    python -m waveform_2d.webapp.server
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .colormap import available
from .engine import SCENES, LiveSimulation

STATIC = Path(__file__).resolve().parent / "static"
FPS = int(os.environ.get("WAVE_FPS", "30"))
SUBSTEPS = int(os.environ.get("WAVE_SUBSTEPS", "6"))
GRID_W = int(os.environ.get("WAVE_GRID_W", "240"))
GRID_H = int(os.environ.get("WAVE_GRID_H", "240"))

app = FastAPI(title="Waveform 2D Live")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/meta")
async def meta() -> dict:
    return {"scenes": list(SCENES), "colormaps": available(),
            "width": GRID_W, "height": GRID_H, "fps": FPS}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    sim = LiveSimulation(width=GRID_W, height=GRID_H)
    await websocket.send_text(json.dumps({
        "type": "init", "width": sim.width, "height": sim.height,
        "scenes": list(SCENES), "colormaps": available(), "fps": FPS,
    }))

    async def receiver() -> None:
        while True:
            msg = json.loads(await websocket.receive_text())
            t = msg.get("type")
            if t == "drip":
                sim.add_drip(msg["x"], msg["y"])
            elif t == "oscillator":
                sim.add_oscillator(msg["x"], msg["y"])
            elif t == "scene":
                sim.set_scene(msg["name"])
            elif t == "param":
                sim.set_param(msg["key"], msg["value"])
            elif t == "reset":
                sim.reset()
            elif t == "clear":
                sim.clear_sources()

    async def sender() -> None:
        dt = 1.0 / max(1, FPS)
        while True:
            sim.step(substeps=SUBSTEPS)
            await websocket.send_bytes(sim.render())
            await asyncio.sleep(dt)

    try:
        await asyncio.gather(receiver(), sender())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    run()
