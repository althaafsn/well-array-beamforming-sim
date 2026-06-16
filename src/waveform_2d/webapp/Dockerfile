# Lean image for the live 2D wave visualizer.
# Build context = repo root:
#   docker build -f src/waveform_2d/webapp/Dockerfile -t waveform2d-web .
#   docker run -p 8000:8000 waveform2d-web
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HOST=0.0.0.0 \
    PORT=8000

# The visualizer only needs the waveform_2d package + these runtime deps.
RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "matplotlib>=3.7" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27"

COPY src/waveform_2d /app/src/waveform_2d

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')" || exit 1

CMD ["python", "-m", "waveform_2d.webapp.server"]
