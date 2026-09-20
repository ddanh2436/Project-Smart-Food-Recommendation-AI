---
title: VietNomNom AI Service
emoji: 🍜
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# VietNomNom — AI Service

FastAPI service behind the VietNomNom food recommendation app. It handles
recommendation ranking, conversational search, Vietnamese sentiment analysis,
review summarisation, and dish recognition from photos.

**Every model runs locally and is free.** There is no paid API key anywhere in
this service.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Status summary (kept backwards compatible) |
| `GET` | `/health` | Per-model readiness and data freshness |
| `GET` | `/docs` | Interactive OpenAPI docs |
| `POST` | `/recommend` | Filter-then-rank restaurant search |
| `POST` | `/parse-intent` | Structured slots extracted from a query |
| `POST` | `/chat` | Multi-turn conversational search |
| `POST` | `/sentiment` | Sentiment of one review |
| `POST` | `/sentiment/batch` | Sentiment of up to 256 reviews in one pass |
| `POST` | `/review-insights` | Aspect-level digest of a restaurant's reviews |
| `POST` | `/predict-food` | Dish recognition from a photo |
| `POST` | `/search-by-image` | Recognise a dish *and* return matching places |
| `POST` | `/admin/reload` | Rebuild the data + search indexes (needs `x-admin-token`) |

## Models

| Component | Model | Size | Cost |
|---|---|---|---|
| Sentiment | `5CD-AI/Vietnamese-Sentiment-visobert` | ~500MB | free |
| Semantic search | `paraphrase-multilingual-MiniLM-L12-v2` | ~120MB | free |
| Dish recognition | `best.pt` (YOLO11m, fine-tuned in this repo) | ~39MB | free |
| Lexical search | TF-IDF char n-grams (scikit-learn) | — | free |

The trained classes are `Banh-Mi`, `Bot Chien`, `Bun`, `Goi-Cuon`, `Pho`
(see `data.yaml`).

## How search works

*Filter first, then rank* — the philosophy from `AI_Workflow.md`, now actually
implemented:

1. **Intent parsing** (`intent.py`) turns free text into slots: dish, area,
   price bounds, minimum rating, time of day, exclusions, party size and the
   ranking the user implied.
2. **Hard filters** (`search.py`) narrow the candidates. Any single filter that
   would empty the result set is dropped instead, and the response reports
   which ones in `relaxed_filters`, so an over-specific query degrades
   gracefully rather than returning nothing.
3. **Hybrid ranking** blends exact-name matching, dish-tag overlap, TF-IDF
   character n-grams (robust to missing diacritics: "bun bo" finds "bún bò")
   and multilingual embedding similarity, plus small rating and proximity
   priors.

## Configuration

Copy `.env.example` to `.env` locally. On Spaces, set these under
**Settings → Variables and secrets** (use *Secret* for `MONGO_URI` and
`ADMIN_TOKEN`).

| Variable | Required | Notes |
|---|---|---|
| `MONGO_URI` | yes | MongoDB Atlas connection string |
| `DB_NAME` | | defaults to `VietNomNom` |
| `PORT` | | `7860` on Spaces |
| `CORS_ORIGINS` | | comma-separated origins, or `*` |
| `ADMIN_TOKEN` | | required to call `/admin/reload` |
| `ENABLE_SEMANTIC_SEARCH` | | `false` saves ~500MB RAM; falls back to TF-IDF |
| `ENABLE_SENTIMENT` | | `false` disables review analysis |
| `ENABLE_YOLO` | | `false` disables photo recognition |
| `DATA_REFRESH_SECONDS` | | how often the restaurant cache reloads (900) |

A free Space has 16GB RAM and 2 vCPUs, which fits all three models. If you hit
memory limits, turn `ENABLE_SEMANTIC_SEARCH` off first — it is the largest
optional component and search still works without it.

## Running locally

```bash
cd AI
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env        # then fill in MONGO_URI
python api.py               # http://127.0.0.1:5000/docs
```

First boot downloads the sentiment and embedding models (~600MB) and is slow;
later boots read them from the cache.

## Module layout

| File | Responsibility |
|---|---|
| `api.py` | FastAPI app, routes, request/response schemas |
| `config.py` | Environment-driven settings |
| `knowledge.py` | Food and location dictionaries, address patterns |
| `text_utils.py` | Normalisation, word-boundary matching, geo, price parsing |
| `intent.py` | Query → structured slots |
| `data_store.py` | Mongo snapshot + TF-IDF and embedding indexes |
| `search.py` | Filter-then-rank pipeline |
| `chat.py` | Conversational layer with multi-turn context |
| `sentiment.py` | Batched Vietnamese sentiment |
| `review_insights.py` | Aspect-based review summarisation |
| `vision.py` | YOLO dish recognition |
