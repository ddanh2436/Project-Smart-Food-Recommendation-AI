# -*- coding: utf-8 -*-
"""Precompute the per-aspect verdicts and store them on the restaurants.

`review_insights` answers "what did people say about this place" one restaurant
at a time, when somebody opens its page. That is the right shape for a detail
page and the wrong shape for search: the numbers it produces — how positive
people were about the food, the service, the parking — are exactly what would
let a query like "quán ăn sạch sẽ" or "quán có chỗ đậu xe" be answered from
evidence instead of from string matching against tags.

So the same computation is run ahead of time, in batches, and written back onto
each restaurant document:

    aspects: {
      hygiene: { positive_ratio: 0.86, mentions: 21, verdict: "positive" },
      parking: { positive_ratio: 0.18, mentions: 11, verdict: "negative" },
      ...
    }
    aspectsAt: <when this was computed>

Written to the restaurant rather than to a separate collection because the AI
service already loads every restaurant document into memory and nothing else
needs the values — one read instead of two.

Incremental by default: a restaurant is skipped when `aspectsAt` is newer than
its newest review, so a re-run only pays for what changed. This matters because
the model runs on a free CPU Space, where the full pass over 27k reviews is
tens of minutes, not seconds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import review_insights
from config import settings

logger = logging.getLogger(__name__)

# Below this many mentions there is no evidence worth ranking on — one person
# saying the parking was bad is not a fact about the restaurant.
MIN_MENTIONS = 3

# Results are written every this many restaurants rather than once at the end
# of a batch. The first run showed why: ordered by review count, the opening
# 400 took an hour, and an interruption anywhere in it would have thrown away
# the whole hour. Flushing often costs a few extra round trips and makes the
# job resumable at any point.
FLUSH_EVERY = 25


def _client():
    from pymongo import MongoClient

    return MongoClient(
        settings.mongo_uri,
        serverSelectionTimeoutMS=15_000,
        connectTimeoutMS=15_000,
    )


def _newest_review_at(reviews: list[dict]) -> datetime | None:
    stamps = [r.get("createdAt") for r in reviews if r.get("createdAt")]
    stamps = [s for s in stamps if isinstance(s, datetime)]
    return max(stamps) if stamps else None


def _needs_work(document: dict, newest: dict, force: bool) -> bool:
    """True when this restaurant has no aspects, or newer reviews than them."""
    if force or not document.get("aspectsAt"):
        return True
    latest = newest.get(document.get("urlGoc"))
    stored = document.get("aspectsNewestReview")
    if latest is None:
        return False
    if stored is None:
        return True
    # Compare naively: Mongo hands both back as datetimes from the same source.
    return latest > stored


def build(limit: int = 200, force: bool = False) -> dict:
    """Compute aspects for up to `limit` restaurants that need it.

    Returns a progress report so a caller can loop until `remaining` is zero,
    which is how this runs on a Space without any single request timing out.
    """
    if not settings.mongo_uri:
        return {"error": "MONGO_URI is not set", "processed": 0, "remaining": 0}

    client = _client()
    database = client[settings.db_name]
    restaurants = database[settings.collection_name]
    reviews_col = database[settings.reviews_collection]

    # One aggregation gives the newest review and the review count per
    # restaurant: the first decides staleness, the second decides order.
    newest: dict[str, object] = {}
    counts: dict[str, int] = {}
    for row in reviews_col.aggregate(
        [
            {
                "$group": {
                    "_id": "$urlGoc",
                    "max": {"$max": "$createdAt"},
                    "n": {"$sum": 1},
                }
            }
        ]
    ):
        if row.get("_id"):
            newest[str(row["_id"])] = row["max"]
            counts[str(row["_id"])] = int(row.get("n", 0))

    # A small projection over the whole collection, so the work queue is exact
    # rather than a query that cannot express "older than its newest review".
    candidates = [
        doc
        for doc in restaurants.find(
            {"urlGoc": {"$in": list(newest)}},
            {"urlGoc": 1, "aspectsAt": 1, "aspectsNewestReview": 1},
        )
        if _needs_work(doc, newest, force)
    ]
    # Most-reviewed first. A full pass takes hours on a free CPU, so the order
    # decides what a partial index covers -- and the places with the most
    # reviews are both the ones search surfaces most and the only ones whose
    # aspect ratios rest on enough evidence to rank on.
    candidates.sort(key=lambda d: counts.get(str(d.get("urlGoc")), 0), reverse=True)
    pending_total = len(candidates)
    batch = candidates[: max(1, int(limit))]

    processed = 0
    skipped = 0
    writes: list = []
    now = datetime.now(timezone.utc)

    from pymongo import UpdateOne

    def flush() -> None:
        if not writes:
            return
        restaurants.bulk_write(
            [UpdateOne(w["filter"], w["update"]) for w in writes], ordered=False
        )
        writes.clear()

    for document in batch:
        url = document.get("urlGoc")
        rows = list(
            reviews_col.find(
                {"urlGoc": url}, {"noiDung": 1, "diemReview": 1, "createdAt": 1}
            )
        )
        if not rows:
            skipped += 1
            continue

        digest = review_insights.summarize_reviews(rows, lang="vi")
        aspects = {
            item["key"]: {
                "positive_ratio": round(float(item["positive_ratio"]), 3),
                "mentions": int(item["mentions"]),
                "verdict": item["verdict"],
            }
            for item in digest.get("aspects", [])
            if int(item.get("mentions", 0)) >= MIN_MENTIONS
        }

        writes.append(
            {
                "filter": {"_id": document["_id"]},
                "update": {
                    "$set": {
                        "aspects": aspects,
                        "aspectsAt": now,
                        "aspectsReviewCount": len(rows),
                        # Lets the next run tell whether reviews have arrived
                        # since, without re-reading a single review body.
                        "aspectsNewestReview": _newest_review_at(rows),
                    }
                },
            }
        )
        processed += 1
        if len(writes) >= FLUSH_EVERY:
            flush()
            logger.info("Aspect index: %s/%s in this batch", processed, len(batch))

    flush()

    remaining = max(0, pending_total - processed - skipped)
    client.close()

    logger.info(
        "Aspect index: %s processed, %s skipped, %s remaining",
        processed, skipped, remaining,
    )
    return {
        "processed": processed,
        "skipped_no_reviews": skipped,
        "remaining": remaining,
        "min_mentions": MIN_MENTIONS,
    }


def status() -> dict:
    """How much of the collection has aspects, for /health and the CLI."""
    if not settings.mongo_uri:
        return {"error": "MONGO_URI is not set"}
    client = _client()
    database = client[settings.db_name]
    restaurants = database[settings.collection_name]
    total = restaurants.count_documents({})
    indexed = restaurants.count_documents({"aspectsAt": {"$exists": True}})
    with_reviews = len(
        database[settings.reviews_collection].distinct("urlGoc")
    )
    client.close()
    return {
        "restaurants": total,
        "indexed": indexed,
        "restaurants_with_reviews": with_reviews,
    }


if __name__ == "__main__":  # pragma: no cover - operational entry point
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="per batch")
    parser.add_argument("--force", action="store_true", help="recompute all")
    parser.add_argument(
        "--all", action="store_true", help="loop until nothing remains"
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(status())
    else:
        while True:
            report = build(limit=args.limit, force=args.force)
            print(report)
            if not args.all or report.get("remaining", 0) <= 0:
                break
            # After the first forced batch those documents are fresh again, so
            # the next pass picks up where this one stopped rather than looping.
            args.force = False
