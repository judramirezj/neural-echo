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
COPY data/clip_library/calibration_bundle.npz ./data/clip_library/calibration_bundle.npz
COPY data/clip_library/raw/ ./data/clip_library/raw/

ENV PORT=8000
EXPOSE 8000

# Render injects $PORT at runtime — shell form so it's expanded, not passed literally.
CMD /app/.venv/bin/uvicorn services.api.main:app --host 0.0.0.0 --port ${PORT}
