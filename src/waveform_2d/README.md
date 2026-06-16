# Waveform 2D — FDTD wave engine

Two-dimensional acoustic / elastic wave simulation for **ultrasonic NDT** (pulse-echo through fluid-filled pipe, steel walls, corrosion pits). Implements finite-difference time domain (FDTD) solvers and interactive browser demos.

Reference: Langtangen & Linge, *Finite Difference Methods for Wave Equations* (2016).

## Package layout

| Module | Purpose |
|--------|---------|
| [`main.py`](main.py) | Explicit leapfrog FDTD on displacement `u(x,y,t)` — book eq. (117)/(118) |
| [`acoustic_pml.py`](acoustic_pml.py) | Velocity–stress FDTD + split-field PML; pulse-echo A-scans |
| [`live_sim.py`](live_sim.py) | Interactive scenes, drawable geometry, RippleGL-style rendering |
| [`webapp/`](webapp/) | FastAPI WebSocket server + Pyodide static deploy |

## Quick start

From the repo root (after `pip install -e ".[web]"`):

```bash
# Book-faithful demo: point source + PNG frames
python -m waveform_2d.main

# Pulse-echo A-scan with PML (pipe cross-section)
python -m waveform_2d.acoustic_pml

# Live WebSocket visualizer → http://localhost:8000
waveform-2d-web
```

## Interactive demo (Pyodide + AWS)

Static browser demo (Python runs client-side via Pyodide):

```bash
./scripts/deploy_waveform_pyodide.sh
# → https://d1w9ll79m0xnql.cloudfront.net/
```

Features: draw blocks, circles, hollow pipe rings; ripple / oscillator sources; wall-clock simulation speed (not tied to FPS).

Source: [`webapp/pyodide/`](webapp/pyodide/) — `wave_engine.py` is a **self-contained browser bundle** (duplicates core logic for zero-import fetch). Server-side code lives in `live_sim.py`.

## Tests

```bash
pytest tests/test_waveform_2d.py -q
```

## Physics notes

- **Heterogeneous media**: water `c=1` (normalized), steel `c=4`; `dt` is recomputed on geometry change (CFL).
- **Walls**: Dirichlet `u=0` on drawn blocks/circles; steel annuli use variable velocity (reflection + transmission).
- **PML**: quadratic absorbing profile per Lorin et al.; used in `acoustic_pml` for pulse-echo ranging.

## Docker (WebSocket server)

```bash
docker build -f src/waveform_2d/webapp/Dockerfile -t waveform2d-web .
docker run -p 8000:8000 waveform2d-web
```

See [`webapp/README.md`](webapp/README.md) for ECS/ALB deployment notes.
