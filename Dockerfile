FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON=3.11

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY neural_echo/ ./neural_echo/
COPY services/ ./services/

# Baked outside /app/data on purpose: render.yaml mounts a persistent disk at
# /app/data, which would shadow/hide anything COPY'd there at build time
# (mounting an empty volume over a directory replaces its contents). Calibration
# data is static and image-baked; /app/data is for runtime state (job artifacts,
# generated candidates, model cache) that should persist across deploys instead.
COPY data/clip_library/calibration_bundle.npz ./calibration_baked/calibration_bundle.npz
COPY data/clip_library/raw/ ./calibration_baked/raw/
ENV CALIBRATION_BUNDLE_PATH=/app/calibration_baked/calibration_bundle.npz
ENV STUB_CLIPS_DIR=/app/calibration_baked/raw

ENV PORT=8000
EXPOSE 8000

# Render injects $PORT at runtime — shell form so it's expanded, not passed literally.
CMD /app/.venv/bin/uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT}
