# Pyodide static deploy

Self-contained browser bundle for AWS S3 + CloudFront (no WebSocket server required).

| File | Role |
|------|------|
| `index.html` | UI, draw tools, wall-clock step loop |
| `wave_engine.py` | Fetched by Pyodide; duplicates `main.py` + `live_sim.py` logic in one file |

**Canonical server-side code:** [`../live_sim.py`](../live_sim.py) and [`../main.py`](../main.py).

When changing simulation logic, update **both** `live_sim.py` and `wave_engine.py`, then redeploy:

```bash
../../../scripts/deploy_waveform_pyodide.sh
```

Live demo: https://d1w9ll79m0xnql.cloudfront.net/
