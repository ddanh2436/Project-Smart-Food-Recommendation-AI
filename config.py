"""Runtime configuration for the VietNomNom AI service.

Every setting is read from the environment so the same code can run locally,
on Hugging Face Spaces, or anywhere else. See `.env.example` for the full list.

Everything here runs on free, self-hosted models — there is no paid API key
anywhere in this service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional: on Hugging Face the vars come from Settings.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, "").strip() or default
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    # --- Database ---
    mongo_uri: str = os.getenv("MONGO_URI", "").strip()
    db_name: str = os.getenv("DB_NAME", "VietNomNom").strip()
    collection_name: str = os.getenv("COLLECTION_NAME", "restaurants").strip()
    reviews_collection: str = os.getenv("REVIEWS_COLLECTION", "reviews").strip()

    # --- Server ---
    host: str = os.getenv("HOST", "0.0.0.0").strip()
    port: int = _int("PORT", 5000)
    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "*"))

    # --- Models (all free, downloaded from the Hugging Face Hub) ---
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "best.pt").strip()
    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "5CD-AI/Vietnamese-Sentiment-visobert"
    ).strip()
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()

    # --- Feature switches ---
    # Semantic search loads a ~120MB embedding model. Turn it off on a
    # RAM-constrained Space and search falls back to TF-IDF only.
    enable_semantic_search: bool = _flag("ENABLE_SEMANTIC_SEARCH", True)
    enable_sentiment: bool = _flag("ENABLE_SENTIMENT", True)
    enable_yolo: bool = _flag("ENABLE_YOLO", True)

    # --- Language-model parser (free tier; off when no key is set) ---
    # Only a query the rules could not read is sent, and only its text; the
    # model fills intent slots and never names a restaurant.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite").strip()
    llm_fallback_model: str = os.getenv(
        "LLM_FALLBACK_MODEL", "gemini-3.1-flash-lite"
    ).strip()
    # Requests per UTC day before the parser stops calling out, so a traffic
    # spike cannot run past the free quota.
    llm_daily_cap: int = _int("LLM_DAILY_CAP", 800)
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "6") or 6)
    # Queries read at or below this confidence go to the model.
    llm_confidence_gate: float = float(os.getenv("LLM_CONFIDENCE_GATE", "0.5") or 0.5)

    # --- Behaviour ---
    data_refresh_seconds: int = _int("DATA_REFRESH_SECONDS", 900)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 10)
    max_results: int = _int("MAX_RESULTS", 64)

    # --- Admin ---
    admin_token: str = os.getenv("ADMIN_TOKEN", "").strip()
    # Shared with the backend, which sends it as `x-internal-token`. When set,
    # every data endpoint requires it; see the middleware in api.py.
    internal_api_token: str = os.getenv("INTERNAL_API_TOKEN", "").strip()

    @property
    def yolo_weights(self) -> Path:
        """Absolute path to the YOLO weights, independent of the working dir."""
        candidate = Path(self.yolo_model_path)
        return candidate if candidate.is_absolute() else BASE_DIR / candidate

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
