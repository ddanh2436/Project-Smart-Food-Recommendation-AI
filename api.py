# -*- coding: utf-8 -*-
"""VietNomNom AI service.

FastAPI app exposing recommendation, chat, sentiment, review-insight and
food-photo endpoints. Every model it uses is free and runs locally — there is
no paid API dependency.

Response shapes for /recommend, /sentiment and /predict-food stay backwards
compatible with the existing NestJS backend; new fields are additive.

Run locally:      python api.py
Run in prod:      uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator

import chat as chat_engine
import aspect_index
import review_insights
from config import settings
from data_store import store
from intent import parse_intent
from search import row_to_payload, search
from sentiment import analyzer
import vision
from vision import detector, title_case

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("vietnomnom.ai")

SERVICE_VERSION = "2.0.0"


# ==========================================================================
# Schemas
# ==========================================================================
class RecommendRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    candidate_ids: list[str] | None = Field(default=None, max_length=5000)
    user_gps: list[float] | None = None
    city_filter: str | None = Field(default=None, max_length=100)
    limit: int = Field(default=64, ge=1, le=200)

    @field_validator("user_gps")
    @classmethod
    def _check_gps(cls, value: list[float] | None) -> list[float] | None:
        """Reject a malformed GPS pair instead of silently mis-ranking.

        The old code indexed ``user_gps[0]`` after only checking the length,
        so a payload like ``["abc", 1]`` raised deep inside the ranker.
        """
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("user_gps must be [latitude, longitude]")
        latitude, longitude = value
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise ValueError("user_gps out of range")
        return [float(latitude), float(longitude)]


class TasteScore(BaseModel):
    """One ranked restaurant. The first six fields are the original contract."""

    id: str
    name: str
    tags: str
    S_taste: float
    distance_km: float
    price: int
    # Additive fields, safe for older clients to ignore.
    address: str = ""
    district: str = ""
    city: str = ""
    rating: float = 0.0
    image_url: str = ""
    source_url: str = ""
    opening_hours: str = ""
    price_text: str = ""
    # Why this row is here, and what to know before going, as facts the client
    # words in its own language. Empty when the query asked for nothing that
    # could be evidenced.
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    cautions: list[dict[str, Any]] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    sort_by: str
    scores: list[TasteScore]
    total_matches: int = 0
    intent: dict[str, Any] | None = None
    relaxed_filters: list[str] = Field(default_factory=list)


class SentimentRequest(BaseModel):
    review: str = Field(..., max_length=5000)


class SentimentResponse(BaseModel):
    label: str
    score: float
    available: bool = True


class SentimentBatchRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=256)


class SentimentBatchResponse(BaseModel):
    results: list[SentimentResponse]


class ReviewItem(BaseModel):
    noiDung: str = Field(default="", max_length=5000)
    diemReview: float | None = None


class ReviewInsightRequest(BaseModel):
    reviews: list[ReviewItem] = Field(default_factory=list, max_length=500)
    lang: Literal["vi", "en"] = "vi"


class ChatMessage(BaseModel):
    role: Literal["user", "bot"]
    text: str = Field(default="", max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=500)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    user_gps: list[float] | None = None
    lang: Literal["vi", "en"] = "vi"
    limit: int = Field(default=5, ge=1, le=20)

    _check_gps = field_validator("user_gps")(
        RecommendRequest._check_gps.__func__  # reuse the same validation
    )


class ChatChip(BaseModel):
    """A quick reply. `label` is shown, `query` is what gets sent."""

    label: str
    query: str


class ChatResponse(BaseModel):
    reply: str
    results: list[TasteScore] = Field(default_factory=list)
    kind: str = "results"
    intent: dict[str, Any] | None = None
    total_matches: int = 0
    relaxed_filters: list[str] = Field(default_factory=list)
    # Which of dish / area / price the query left unset, and the chips that
    # fill them. Empty once the query is specific enough to stand on its own.
    slots_missing: list[str] = Field(default_factory=list)
    chips: list[ChatChip] = Field(default_factory=list)
    # Kept so any older client reading `reply_text` keeps working.
    reply_text: str = ""


class IntentRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    has_gps: bool = False


# ==========================================================================
# Lifespan
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the caches on boot.

    Replaces the deprecated ``@app.on_event("startup")`` hook. Model loads are
    individually guarded: one failing model must not stop the service, so the
    healthy endpoints stay usable and /health reports what is degraded.
    """
    logger.info("Starting VietNomNom AI service v%s", SERVICE_VERSION)
    started = time.time()

    store.reload()
    if settings.enable_sentiment:
        analyzer.load()
    if settings.enable_yolo:
        detector.load()

    logger.info("Startup finished in %.1fs", time.time() - started)
    logger.info("Status: %s", health_payload())
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="VietNomNom AI Service",
    version=SERVICE_VERSION,
    description="Recommendation, chat, sentiment and food recognition.",
    lifespan=lifespan,
)

# "*" cannot be combined with credentials, so only enable them for an explicit
# origin list. The old config sent both, which browsers reject outright.
allow_all = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not allow_all,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


def require_admin(x_admin_token: str = Header(default="")) -> None:
    """Guard destructive/expensive endpoints.

    Refuses when no token is configured, so an unset ADMIN_TOKEN fails closed
    rather than leaving the endpoint open to anyone.
    """
    if not settings.admin_token:
        raise HTTPException(503, "ADMIN_TOKEN is not configured on the server")
    if x_admin_token != settings.admin_token:
        raise HTTPException(401, "Invalid admin token")


# ==========================================================================
# Health
# ==========================================================================
def health_payload() -> dict:
    data = store.status()
    return {
        "status": "ok" if data["restaurants"] > 0 else "degraded",
        "version": SERVICE_VERSION,
        "data": data,
        "models": {
            "sentiment": {"ready": analyzer.ready, "error": analyzer.error},
            "yolo": {"ready": detector.ready, "error": detector.error},
            "semantic_search": {
                "ready": store.semantic_ready,
                "error": data["embedding_error"],
            },
        },
    }


@app.get("/")
async def root() -> dict:
    payload = health_payload()
    # Keep the original keys so existing uptime checks keep passing.
    payload["data_status"] = (
        "Active" if payload["data"]["restaurants"] > 0 else "Empty Data"
    )
    return payload


@app.get("/health")
async def health() -> dict:
    return health_payload()


# ==========================================================================
# Recommendation
# ==========================================================================
@app.post("/recommend", response_model=RecommendResponse)
async def handle_recommend(request: RecommendRequest) -> dict:
    result = search(
        request.query,
        user_gps=request.user_gps,
        city_filter=request.city_filter,
        limit=request.limit,
        candidate_ids=request.candidate_ids,
    )
    if not store.is_ready:
        raise HTTPException(
            503,
            store.status()["load_error"]
            or "Restaurant data is not loaded yet, please retry shortly.",
        )
    return {
        "sort_by": result.sort_by,
        "scores": [row_to_payload(row, result.intent) for _, row in result.rows.iterrows()],
        "total_matches": result.total_before_ranking,
        "intent": result.intent.to_dict(),
        "relaxed_filters": result.relaxed_filters,
    }


@app.post("/parse-intent")
async def handle_parse_intent(request: IntentRequest) -> dict:
    """Expose the intent parser on its own, for debugging and for the backend."""
    return parse_intent(request.query, has_gps=request.has_gps).to_dict()


# ==========================================================================
# Chat
# ==========================================================================
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest) -> dict:
    history = [
        chat_engine.ChatTurn(role=item.role, text=item.text)
        for item in request.history
    ]
    outcome = chat_engine.respond(
        request.message,
        history=history,
        user_gps=request.user_gps,
        lang=request.lang,
        limit=request.limit,
    )
    return {
        "reply": outcome["reply"],
        "reply_text": outcome["reply"],  # legacy key
        "results": outcome["results"],
        "kind": outcome["kind"],
        "intent": outcome.get("intent"),
        "total_matches": outcome.get("total_matches", 0),
        "relaxed_filters": outcome.get("relaxed_filters", []),
    }


# ==========================================================================
# Sentiment
# ==========================================================================
@app.post("/sentiment", response_model=SentimentResponse)
async def handle_sentiment(request: SentimentRequest) -> dict:
    return analyzer.analyze(request.review)


@app.post("/sentiment/batch", response_model=SentimentBatchResponse)
async def handle_sentiment_batch(request: SentimentBatchRequest) -> dict:
    """Classify many reviews in one pass.

    The backfill job used to issue one HTTP request per review; this turns a
    multi-minute migration into a handful of calls.
    """
    return {"results": analyzer.analyze_batch(request.reviews)}


@app.post("/review-insights")
async def handle_review_insights(request: ReviewInsightRequest) -> dict:
    """Aspect-level digest of a restaurant's reviews."""
    return review_insights.summarize_reviews(
        [item.model_dump() for item in request.reviews], lang=request.lang
    )


# ==========================================================================
# Food photo recognition
# ==========================================================================
@app.post("/predict-food")
async def predict_food(file: UploadFile = File(...)) -> dict:
    if not detector.ready:
        detector.load()
    if not detector.ready:
        raise HTTPException(503, detector.error or "Food model is unavailable")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, f"Expected an image, got {file.content_type}")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty upload")
    # Guard the upload size: an unbounded read is a trivial memory exhaustion
    # vector on a small Space.
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"Image larger than {settings.max_upload_mb}MB"
        )

    # Read below the naming floor so a weak detection can still say what kind
    # of food this looks like, instead of the dead end an empty list used to be.
    detections = detector.detect(
        contents, top_k=3, min_confidence=vision.WEAK_CONFIDENCE
    )
    verdict = vision.classify(detections)
    named = title_case(verdict["dish"]) if verdict["dish"] else None

    return {
        # Original contract: `food_name` is None unless a dish is actually
        # being asserted, which is the two top tiers.
        "food_name": named,
        "original_name": detections[0]["class_name"] if detections else None,
        "confidence": detections[0]["confidence"] if detections else 0.0,
        # How much to trust it: confident | uncertain | group | none. The
        # client words the reply, because it knows the interface language.
        "tier": verdict["tier"],
        "group": verdict["group"],
        "suggestions": verdict["suggestions"],
        # Lets the client offer "did you mean ...?"
        "detections": [
            {
                "food_name": title_case(item["dish"]),
                "original_name": item["class_name"],
                "confidence": item["confidence"],
            }
            for item in detections
            if item["confidence"] >= vision.MIN_CONFIDENCE
        ],
    }


@app.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...)) -> dict:
    """Recognise a dish and return matching restaurants in one round trip."""
    prediction = await predict_food(file)
    dish = prediction.get("food_name")

    # Nothing is searched unless a dish is actually being named. Searching on a
    # 0.18 guess would hand the user a confident list of the wrong restaurants,
    # which is worse than saying the photo was not clear enough.
    if not dish:
        return {
            "detected_food": None,
            "scores": [],
            "detections": [],
            "tier": prediction.get("tier", "none"),
            "group": prediction.get("group"),
            "suggestions": prediction.get("suggestions", []),
        }

    result = search(dish, limit=10)
    return {
        "detected_food": dish,
        "confidence": prediction.get("confidence"),
        "detections": prediction.get("detections", []),
        "tier": prediction.get("tier"),
        "group": prediction.get("group"),
        "suggestions": prediction.get("suggestions", []),
        "sort_by": result.sort_by,
        "scores": [row_to_payload(row, result.intent) for _, row in result.rows.iterrows()],
    }


# ==========================================================================
# Admin
# ==========================================================================
@app.post("/admin/aspect-index", dependencies=[Depends(require_admin)])
async def admin_aspect_index(limit: int = 200, force: bool = False) -> dict:
    """Precompute the per-aspect review verdicts used for aspect search.

    Batched on purpose: the full pass runs the sentiment model over every
    review and takes hours on a free CPU. Call it repeatedly until `remaining`
    reaches zero. Most-reviewed restaurants are indexed first, so stopping part
    way still covers the places search actually surfaces.
    """
    return aspect_index.build(limit=limit, force=force)


@app.get("/admin/aspect-index", dependencies=[Depends(require_admin)])
async def admin_aspect_index_status() -> dict:
    return aspect_index.status()


@app.post("/admin/reload", dependencies=[Depends(require_admin)])
async def admin_reload() -> dict:
    """Re-read the restaurant collection and rebuild the search indexes."""
    return store.reload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
