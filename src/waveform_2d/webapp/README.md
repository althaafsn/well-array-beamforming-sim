# Waveform 2D — live web visualizer

A RippleGL-style, real-time viewer for the 2D finite-difference wave engine. The
simulation runs **server-side in Python/NumPy**; each frame is mapped to a fixed
colour scale and streamed as raw RGBA over a WebSocket to a browser canvas, so
the animation is smooth and the colours are *consistent* across frames (no
per-frame auto-scaling).

Scenes:
- **tank** — open water; click to drop ripples, shift-click to plant an oscillator.
- **double_slit** — a reflecting barrier with two slits → interference fringes.
- **pipe** — water lumen + steel wall with a corrosion pit (the NDT use case).

## Run locally

```bash
pip install -e ".[web]"          # or: pip install -r requirements.txt
waveform-2d-web                  # -> http://localhost:8000
# alternatives:
uvicorn waveform_2d.webapp.server:app --reload
python -m waveform_2d.webapp.server
```

Tunables via environment variables: `WAVE_GRID_W`, `WAVE_GRID_H`, `WAVE_FPS`,
`WAVE_SUBSTEPS`, `HOST`, `PORT`.

## Docker

```bash
docker build -f src/waveform_2d/webapp/Dockerfile -t waveform2d-web .
docker run -p 8000:8000 waveform2d-web
```

## Deploy on AWS

The app is a single stateless container that speaks HTTP **and WebSockets**, so
any container service works — just make sure WebSockets are allowed.

**Option A — ECS Fargate behind an ALB (recommended; full WebSocket support).**
1. Push the image to ECR:
   ```bash
   aws ecr create-repository --repository-name waveform2d-web
   docker tag waveform2d-web:latest <acct>.dkr.ecr.<region>.amazonaws.com/waveform2d-web
   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
   docker push <acct>.dkr.ecr.<region>.amazonaws.com/waveform2d-web
   ```
2. Create an ECS Fargate service (container port 8000) behind an Application Load
   Balancer. ALB supports WebSockets out of the box; point the target group health
   check at `/healthz`. Raise the target-group idle timeout (e.g. 300s) so long-
   lived sockets aren't dropped.

**Option B — single EC2 instance.** Install Docker, `docker run -p 80:8000 ...`,
open the security group on 80/443. Put Nginx/Caddy in front for TLS (`wss://`).

**Note on App Runner / API Gateway:** plain App Runner and REST API Gateway do
not proxy raw WebSockets the way this app uses them — prefer ALB/ECS or EC2. (The
client also has an HTTP `/healthz` and `/api/meta` for probes.)

For many concurrent viewers, scale horizontally (each socket pins one CPU while
stepping); sessions are independent, so no shared state or sticky routing beyond
the socket itself is required.
