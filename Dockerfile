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

# Download the models at build time so they live in an image layer.
#
# A Space's container filesystem is ephemeral and a free Space sleeps after
# about 48h idle, so without this every cold start re-downloads roughly 600MB
# before it can serve the first request -- slow enough to look broken, and
# liable to time out. Baking them in makes startup a model *load* rather than a
# download.
#
# Run as appuser, after the USER switch, so the files land in the cache
# directories the runtime actually reads.
#
# Kept non-fatal: if the Hub is unreachable during a build, the image still
# ships and the service falls back to downloading on first use, rather than
# failing the deploy outright.
RUN python -c "\
from transformers import pipeline; \
pipeline('sentiment-analysis', model='5CD-AI/Vietnamese-Sentiment-visobert')" \
    || echo 'WARN: sentiment model not pre-cached, will download at runtime'

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
    || echo 'WARN: embedding model not pre-cached, will download at runtime'

EXPOSE 7860

# start-period covers loading the (now pre-cached) models and the first read
# of the restaurant collection.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Single worker: each one would load its own copy of the models, and a free
# Space does not have the RAM for two.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
