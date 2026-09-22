# -*- coding: utf-8 -*-
"""In-memory restaurant store with a TF-IDF and an embedding index.

Fixes carried over from the original ``api.py``:

* ``MONGO_URI`` was hardcoded to the literal string ``"//"`` — it now comes
  from the environment and a missing value is reported instead of silently
  producing an empty database.
* ``tfidf_vectorizer`` was only assigned inside the non-empty branch, so an
  empty collection left it ``None`` and every ``/recommend`` call raised.
  The index is now always built, even over zero rows.
* the ``rating`` column was assumed to exist; a collection without
  ``diemTrungBinh`` made sorting raise ``KeyError``. Every column the ranker
  touches is now guaranteed present with a sensible default.
* data was read exactly once at startup, so newly crawled restaurants never
  appeared. The store now refreshes on a TTL.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import knowledge as kb
import text_utils as tu
from config import settings

logger = logging.getLogger(__name__)

# Mongo field name -> the name used everywhere in this service.
FIELD_RENAMES = {
    "tenQuan": "name",
    "diemTrungBinh": "rating",
    "giaCa": "price_text",
    "diaChi": "address",
    "gioMoCua": "opening_hours",
    "avatarUrl": "image_url",
    "urlGoc": "source_url",
    "diemKhongGian": "score_space",
    "diemViTri": "score_location",
    "diemChatLuong": "score_quality",
    "diemPhucVu": "score_service",
    "diemGiaCa": "score_price",
    # Computed and stored by the backend (rating-stats.service.ts). Reading
    # them here is what keeps the chatbot's ordering identical to the website's.
    "reviewCount": "review_count",
    "diemTrungBinhAdj": "rating_adjusted",
}

# Columns the ranker relies on, with the default used when absent.
REQUIRED_COLUMNS: dict[str, Any] = {
    "id": "",
    "name": "",
    "tags": "",
    "address": "",
    "opening_hours": "",
    "price_text": "",
    "image_url": "",
    "source_url": "",
    "rating": 0.0,
    "price": 0.0,
    "lat": 0.0,
    "lon": 0.0,
    "district": "",
    "city": "",
    "score_space": 0.0,
    "score_location": 0.0,
    "score_quality": 0.0,
    "score_service": 0.0,
    "score_price": 0.0,
    "review_count": 0.0,
    "rating_adjusted": 0.0,
}

NUMERIC_COLUMNS = [
    "rating", "price", "lat", "lon",
    "score_space", "score_location", "score_quality",
    "score_service", "score_price",
    "review_count", "rating_adjusted",
]
# --------------------------------------------------------------------------
# Rating shrinkage
#
# The raw rating is not comparable across restaurants, because the best-rated
# ones are also the least reviewed. Measured on the live data:
#
#     rating 9.5-10.0   488 places   median  1 review
#     rating 8.0-9.5   1415 places   median  2 reviews
#     rating 6.0-8.0   3012 places   median  6 reviews
#
# So a 10.0 is usually one person's opinion, while a 7.9 can rest on hundreds.
# Ranking on the raw number therefore promotes noise: a search for pho in Hanoi
# put a 10.0-rated fried-rice place above well-reviewed pho specialists.
#
# The fix is the standard Bayesian shrinkage toward the global mean,
#
#     adjusted = (C * m + n * R) / (C + n)
#
# and it is computed ONCE, by the backend, in rating-stats.service.ts. This
# module used to carry a second implementation of the same formula, with its
# own copy of the constants and its own global mean taken from its own
# snapshot. Two implementations of one ranking rule is a standing bug: the
# website ordered results by the backend's numbers while the assistant ordered
# them by its own, and a change to either one silently put the two out of step.
# The value is read from `diemTrungBinhAdj` now, and the review count from
# `reviewCount`, both already on the document.
#
# Only ranking uses the adjusted value; the raw rating is what gets displayed,
# so the UI never shows a number that contradicts the order.


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame({name: pd.Series(dtype="object")
                          for name in REQUIRED_COLUMNS})
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.Series(dtype="float64")
    return frame


def extract_district(address: str) -> str:
    """Find the district a stored address belongs to.

    Uses full-name, word-boundary patterns. The old version tested every
    alias as a bare substring, including two-letter ones, so "st" matched
    inside "Street" and mislabelled the row.
    """
    if not isinstance(address, str) or not address:
        return ""
    # Longest canonical name first so "Quận 12" is preferred over "Quận 1".
    for name in sorted(kb.DISTRICT_TO_CITY, key=len, reverse=True):
        if kb.ADDRESS_PATTERNS[name].search(address):
            return name
    return ""


def extract_city(address: str, district: str) -> str:
    """Resolve the city, preferring the district's known parent."""
    if district:
        city = kb.DISTRICT_TO_CITY.get(district)
        if city:
            return city
    if not isinstance(address, str) or not address:
        return ""
    for city, pattern in kb.CITY_PATTERNS.items():
        if pattern.search(address):
            return city
    return ""


def _tags_to_text(value) -> str:
    """Normalize the tags field, which may be a list or a string."""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


class RestaurantStore:
    """Thread-safe snapshot of the restaurant collection plus its indexes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._frame: pd.DataFrame = _empty_frame()
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._embedder = None
        self._embeddings: np.ndarray | None = None
        self._loaded_at: float = 0.0
        self._load_error: str | None = None
        self._embedding_error: str | None = None

    # ---------------------------------------------------------------- state
    @property
    def frame(self) -> pd.DataFrame:
        with self._lock:
            return self._frame

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return not self._frame.empty

    @property
    def semantic_ready(self) -> bool:
        with self._lock:
            return self._embeddings is not None and len(self._embeddings) > 0

    def status(self) -> dict:
        with self._lock:
            return {
                "restaurants": int(len(self._frame)),
                "loaded_at": self._loaded_at,
                "age_seconds": (
                    round(time.time() - self._loaded_at, 1)
                    if self._loaded_at else None
                ),
                "tfidf_ready": self._tfidf_matrix is not None,
                "semantic_ready": self.semantic_ready,
                "load_error": self._load_error,
                "embedding_error": self._embedding_error,
            }

    # ----------------------------------------------------------- refreshing
    def ensure_fresh(self) -> None:
        """Reload if the snapshot is older than the configured TTL."""
        with self._lock:
            age = time.time() - self._loaded_at
            stale = self._loaded_at == 0 or age > settings.data_refresh_seconds
        if stale:
            self.reload()

    def reload(self) -> dict:
        """Read the collection and rebuild both indexes."""
        frame, error = self._fetch()
        with self._lock:
            self._frame = frame
            self._load_error = error
            self._loaded_at = time.time()
            self._build_tfidf()
            self._build_embeddings()
        status = self.status()
        logger.info(
            "Store reloaded: %s restaurants (tfidf=%s, semantic=%s)",
            status["restaurants"], status["tfidf_ready"], status["semantic_ready"],
        )
        return status

    def _fetch(self) -> tuple[pd.DataFrame, str | None]:
        if not settings.mongo_uri:
            message = (
                "MONGO_URI is not set — the AI service cannot read restaurants. "
                "Set it in .env locally, or under Settings > Variables and "
                "secrets on Hugging Face Spaces."
            )
            logger.error(message)
            return _empty_frame(), message

        try:
            from pymongo import MongoClient

            client = MongoClient(
                settings.mongo_uri,
                serverSelectionTimeoutMS=10_000,
                connectTimeoutMS=10_000,
            )
            database = client[settings.db_name]
            documents = list(database[settings.collection_name].find({}))
            # No second pass over the reviews collection: `reviewCount` is
            # already on each restaurant document, maintained by the backend.
            client.close()
        except Exception as exc:  # noqa: BLE001 - surfaced via /health
            message = f"MongoDB read failed: {exc}"
            logger.exception(message)
            return _empty_frame(), message

        if not documents:
            logger.warning(
                "Collection %s.%s is empty",
                settings.db_name, settings.collection_name,
            )
            return _empty_frame(), None

        return self._normalize(documents), None

    def _normalize(self, documents: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(documents)
        frame["id"] = frame["_id"].astype(str)
        frame = frame.rename(columns=FIELD_RENAMES)

        # Guarantee every column the ranker reads, with a typed default.
        for column, default in REQUIRED_COLUMNS.items():
            if column not in frame.columns:
                frame[column] = default

        frame["tags"] = frame["tags"].apply(_tags_to_text).fillna("")
        frame["name"] = frame["name"].fillna("").astype(str)
        frame["address"] = frame["address"].fillna("").astype(str)
        frame["opening_hours"] = frame["opening_hours"].fillna("").astype(str)

        frame["lat"] = frame["lat"].apply(tu.clean_coordinate)
        frame["lon"] = frame["lon"].apply(tu.clean_coordinate)
        frame["price"] = frame["price_text"].apply(tu.clean_price)

        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        # `rating_adjusted` and `review_count` arrive on the documents; see the
        # shrinkage note above. If the backend has never run its rating pass the
        # column is all zeros, and ranking on that would order every restaurant
        # identically -- so fall back to the raw rating and say so in the log.
        if float(frame["rating_adjusted"].max() or 0.0) <= 0.0:
            logger.warning(
                "No diemTrungBinhAdj on any document: run the backend's rating "
                "pass (POST /restaurants/admin/rating-stats). Ranking falls "
                "back to the raw rating until then."
            )
            frame["rating_adjusted"] = frame["rating"]

        frame["district"] = frame["address"].apply(extract_district)
        frame["city"] = [
            extract_city(address, district)
            for address, district in zip(frame["address"], frame["district"])
        ]

        # Pre-folded text for cheap accent-insensitive matching later.
        frame["name_norm"] = frame["name"].apply(tu.normalize)
        frame["tags_norm"] = frame["tags"].apply(tu.normalize)
        frame["search_blob"] = (
            frame["name_norm"] + " " + frame["tags_norm"] + " "
            + frame["district"].apply(tu.normalize)
        ).str.strip()

        return frame.reset_index(drop=True)

    # --------------------------------------------------------------- TF-IDF
    def _build_tfidf(self) -> None:
        """Always build, even over zero rows, so callers never see ``None``."""
        corpus = (
            self._frame["search_blob"].tolist()
            if "search_blob" in self._frame.columns else []
        )
        try:
            # char_wb n-grams tolerate the misspellings and missing diacritics
            # that dominate real queries ("bun bo" vs "bún bò").
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
                sublinear_tf=True,
            )
            if corpus and any(text.strip() for text in corpus):
                self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
            else:
                self._vectorizer.fit([" "])
                self._tfidf_matrix = None
        except Exception as exc:  # noqa: BLE001
            logger.exception("TF-IDF build failed: %s", exc)
            self._vectorizer = None
            self._tfidf_matrix = None

    def tfidf_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every row."""
        with self._lock:
            if self._vectorizer is None or self._tfidf_matrix is None:
                return np.zeros(len(self._frame), dtype="float32")
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                vector = self._vectorizer.transform([tu.normalize(query)])
                return cosine_similarity(vector, self._tfidf_matrix).ravel()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TF-IDF scoring failed: %s", exc)
                return np.zeros(len(self._frame), dtype="float32")

    # ------------------------------------------------------------ embedding
    def _build_embeddings(self) -> None:
        if not settings.enable_semantic_search:
            self._embeddings = None
            self._embedding_error = "disabled by ENABLE_SEMANTIC_SEARCH"
            return
        if self._frame.empty:
            self._embeddings = None
            return

        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", settings.embedding_model)
                self._embedder = SentenceTransformer(settings.embedding_model)

            texts = [
                f"{name}. {tags}".strip(". ")
                for name, tags in zip(self._frame["name"], self._frame["tags"])
            ]
            self._embeddings = self._embedder.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self._embedding_error = None
        except Exception as exc:  # noqa: BLE001 - optional feature
            logger.warning("Semantic index unavailable, using TF-IDF only: %s", exc)
            self._embeddings = None
            self._embedding_error = str(exc)

    def semantic_scores(self, query: str) -> np.ndarray:
        """Cosine similarity in embedding space, or zeros when unavailable."""
        with self._lock:
            if self._embedder is None or self._embeddings is None:
                return np.zeros(len(self._frame), dtype="float32")
            try:
                vector = self._embedder.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")
                # Both sides are L2-normalized, so a dot product is the cosine.
                return (self._embeddings @ vector[0]).ravel()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Semantic scoring failed: %s", exc)
                return np.zeros(len(self._frame), dtype="float32")


store = RestaurantStore()
