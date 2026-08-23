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
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY neural_echo/ ./neural_echo/
COPY services/ ./services/

# Baked outside /app/data on purpose: render.yaml mounts a persistent disk at
# /app/data, which would shadow/hide anything COPY'd there at build time
# (mounting an empty volume over a directory replaces its contents). The stub
# clip library is static and image-baked; /app/data is for runtime state (job
# artifacts, generated candidates, model cache) that should persist across
# deploys instead.
COPY data/clip_library/raw/ ./calibration_baked/raw/
ENV STUB_CLIPS_DIR=/app/calibration_baked/raw

# Runpod pods should mount their persistent network volume at /app/data. Both
# model downloads and generated job artifacts then survive pod restarts.
ENV HF_HOME=/app/data/cache/huggingface
ENV TRIBE_CACHE_DIR=/app/data/cache/tribe

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=8s --start-period=600s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT}/health" > /dev/null || exit 1

# Shell wrapper expands Runpod's optional PORT override; exec preserves signals.
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT}"]
