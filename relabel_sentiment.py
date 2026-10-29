# -*- coding: utf-8 -*-
"""Relabel every stored review with the current SENTIMENT_MODEL.

The backend's backfill only labels reviews that have no label yet, so swapping
in the 4-label model (kaggle/README.md, notebook 7) leaves the 26k crawled
reviews on the old 3-label verdicts. This rewrites them.

Resumable: each row records the model that labelled it (`aiSentimentModel`),
so an interrupted run picks up where it stopped, and a later model swap is the
same command. Minutes on a Kaggle/Colab GPU; around an hour on a laptop CPU.

    SENTIMENT_MODEL=Sirin2412/vietnomnom-sentiment-4c HF_TOKEN=... python relabel_sentiment.py
"""

from __future__ import annotations

from pymongo import MongoClient, UpdateOne

from config import settings
from sentiment import analyzer

BATCH = 256


def main() -> None:
    if not settings.mongo_uri:
        raise SystemExit("MONGO_URI is not set")
    analyzer.load()
    if not analyzer.ready:
        raise SystemExit(f"Sentiment model is not available: {analyzer.error}")

    reviews = MongoClient(settings.mongo_uri)[settings.db_name][settings.reviews_collection]
    todo = {
        "aiSentimentModel": {"$ne": settings.sentiment_model},
        "noiDung": {"$regex": r"\S"},  # blank text gets no verdict and would loop forever
    }
    total, done = reviews.count_documents(todo), 0
    print(f"{total} reviews to relabel with {settings.sentiment_model}")

    while batch := list(reviews.find(todo, {"noiDung": 1}).limit(BATCH)):
        results = analyzer.analyze_batch([r["noiDung"] for r in batch])
        if not all(r["available"] for r in results):
            raise SystemExit("Sentiment inference failed; this batch was not written")
        reviews.bulk_write([
            UpdateOne({"_id": doc["_id"]}, {"$set": {
                "aiSentimentLabel": result["label"],
                "aiSentimentScore": result["score"],
                "aiSentimentModel": settings.sentiment_model,
            }})
            for doc, result in zip(batch, results)
        ])
        done += len(batch)
        print(f"{done}/{total}")


if __name__ == "__main__":  # pragma: no cover - operational entry point
    main()
