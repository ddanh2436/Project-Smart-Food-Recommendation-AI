# Hugging Face Spaces (Docker SDK) image for the VietNomNom AI service.
FROM python:3.11-slim

# --- OS packages ---
# libgl1 + libglib2.0-0 are required by opencv, which ultralytics imports.
# curl is only used by the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000; the model caches must be writable by that user.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence-transformers \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# Install dependencies first so a code-only change reuses the layer cache.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser
RUN mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME" "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR"

EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Single worker: each one would load its own copy of the models, and a free
# Space does not have the RAM for two.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
