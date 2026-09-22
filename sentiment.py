# -*- coding: utf-8 -*-
"""Vietnamese sentiment analysis, loaded lazily and batched.

Wraps the free ``5CD-AI/Vietnamese-Sentiment-visobert`` model. Two things the
original ``api.py`` got wrong are fixed here:

* the load was wrapped in a bare ``except: pass``, so a failure was invisible
  and every later call silently returned a neutral placeholder;
* reviews were classified one HTTP request at a time, which made the
  sentiment backfill over a few thousand reviews take many minutes. This
  module exposes a batch API.
"""

from __future__ import annotations

import logging
import threading

from config import settings

logger = logging.getLogger(__name__)

# The model emits LABEL_0/1/2; map to names the frontend already understands.
LABEL_ALIASES = {
    "LABEL_0": "NEG",
    "LABEL_1": "NEU",
    "LABEL_2": "POS",
    "NEGATIVE": "NEG",
    "NEUTRAL": "NEU",
    "POSITIVE": "POS",
    "NEG": "NEG",
    "NEU": "NEU",
    "POS": "POS",
}

# Longest review the model sees; visobert's window is 256 tokens.
MAX_CHARS = 1200


class SentimentAnalyzer:
    """Lazy, thread-safe wrapper around the sentiment pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline = None
        self._error: str | None = None
        self._attempted = False

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        """Load the model once. Safe to call repeatedly."""
        if not settings.enable_sentiment:
            self._error = "disabled by ENABLE_SENTIMENT"
            return
        with self._lock:
            if self._pipeline is not None or self._attempted:
                return
            self._attempted = True
            try:
                from transformers import pipeline

                # Use a GPU when there is one. The Space has none, so this
                # changes nothing there -- but the aspect index is a pass over
                # every review in the database, which is hours on a CPU and
                # minutes on the free GPU a Colab or Kaggle notebook gives you.
                # Without this the pipeline defaults to CPU even on a machine
                # with CUDA sitting idle.
                device = -1
                try:
                    import torch

                    if torch.cuda.is_available():
                        device = 0
                except ImportError:  # pragma: no cover
                    pass

                logger.info(
                    "Loading sentiment model %s on %s",
                    settings.sentiment_model,
                    "GPU" if device == 0 else "CPU",
                )
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=settings.sentiment_model,
                    truncation=True,
                    max_length=256,
                    device=device,
                )
                self._error = None
                logger.info("Sentiment model ready")
            except Exception as exc:  # noqa: BLE001 - reported via /health
                self._error = str(exc)
                self._pipeline = None
                logger.exception("Sentiment model failed to load: %s", exc)

    def analyze(self, text: str) -> dict:
        """Classify one review."""
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Classify many reviews in one forward pass.

        Always returns one result per input, in order, so a caller can zip it
        back onto its own list without length checks.
        """
        results: list[dict] = [
            {"label": "NEU", "score": 0.5, "available": False} for _ in texts
        ]
        if not texts:
            return results

        if self._pipeline is None and not self._attempted:
            self.load()
        if self._pipeline is None:
            return results

        cleaned = [(text or "").strip()[:MAX_CHARS] for text in texts]
        # Indices of inputs worth sending to the model.
        indices = [i for i, text in enumerate(cleaned) if text]
        if not indices:
            return results

        try:
            raw = self._pipeline([cleaned[i] for i in indices], batch_size=16)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment inference failed: %s", exc)
            return results

        for position, index in enumerate(indices):
            item = raw[position]
            if isinstance(item, list):  # some pipelines nest per-input lists
                item = max(item, key=lambda entry: entry.get("score", 0.0))
            label = LABEL_ALIASES.get(str(item.get("label", "")).upper(), "NEU")
            results[index] = {
                "label": label,
                "score": float(item.get("score", 0.5)),
                "available": True,
            }
        return results


analyzer = SentimentAnalyzer()
