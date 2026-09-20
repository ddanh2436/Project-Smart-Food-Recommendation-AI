# -*- coding: utf-8 -*-
"""Aspect-based review summarisation, without an LLM.

Produces the kind of digest a user actually wants on a restaurant page —
"food is great, service is slow, parking is awkward" — using only free,
local components:

1. split each review into clauses (Vietnamese reviews routinely praise and
   complain in the same sentence: "đồ ăn ngon nhưng phục vụ chậm");
2. assign clauses to aspects by keyword;
3. score each clause with the visobert sentiment model, batched;
4. aggregate per aspect and pick the most representative real quote.

Everything returned is extracted from the reviews themselves, so there is
nothing invented — the quotes are verbatim.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import text_utils as tu
from sentiment import analyzer

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Aspects and the words that signal them.
# --------------------------------------------------------------------------
ASPECTS: dict[str, dict] = {
    "food": {
        "label_vi": "Món ăn",
        "label_en": "Food",
        "icon": "🍜",
        "keywords": [
            "món ăn", "đồ ăn", "thức ăn", "hương vị", "khẩu vị",
            "ngon", "dở", "tệ", "nhạt", "mặn", "ngọt", "cay", "thơm",
            "tươi", "nguội", "dai", "mềm", "béo", "ngấy", "đậm đà", "nêm",
            "nước dùng", "nước lèo", "topping", "phần ăn", "suất", "chín",
            "food", "dish", "taste", "flavour", "flavor", "delicious",
        ],
    },
    "price": {
        "label_vi": "Giá cả",
        "label_en": "Price",
        "icon": "💰",
        "keywords": [
            "giá", "giá cả", "tiền", "đắt", "mắc", "rẻ", "hợp lý",
            "phải chăng", "chát", "xứng đáng", "đáng tiền", "hoá đơn",
            "hóa đơn", "nghìn", "price", "expensive", "cheap", "worth",
            "pricey", "value",
        ],
    },
    "service": {
        "label_vi": "Phục vụ",
        "label_en": "Service",
        "icon": "🧑‍🍳",
        "keywords": [
            "phục vụ", "nhân viên", "thái độ", "nhiệt tình", "chu đáo",
            "thân thiện", "lễ phép", "cọc", "chậm", "lâu", "nhanh",
            "chờ", "đợi", "order", "gọi món", "tính tiền", "chủ quán",
            "service", "staff", "waiter", "waitress", "friendly", "rude",
        ],
    },
    "space": {
        "label_vi": "Không gian",
        "label_en": "Ambience",
        "icon": "🪑",
        "keywords": [
            "không gian", "chỗ ngồi", "bàn", "ghế", "rộng", "chật",
            "thoáng", "máy lạnh", "điều hòa", "view", "trang trí", "decor",
            "ồn", "ồn ào", "yên tĩnh", "nhạc", "ánh sáng", "nóng nực",
            "ambience", "ambiance", "space", "seating", "decoration", "noisy",
        ],
    },
    "hygiene": {
        "label_vi": "Vệ sinh",
        "label_en": "Hygiene",
        "icon": "🧼",
        "keywords": [
            "vệ sinh", "nhà vệ sinh", "sạch", "sạch sẽ", "bẩn", "ruồi",
            "gián", "chuột", "kiến", "mùi", "hôi", "tanh", "bát", "chén",
            "đĩa", "khăn", "toilet", "clean", "dirty", "hygiene", "smell",
        ],
    },
    "parking": {
        "label_vi": "Chỗ đậu xe",
        "label_en": "Parking",
        "icon": "🛵",
        "keywords": [
            "đậu xe", "đỗ xe", "gửi xe", "giữ xe", "bãi xe", "vé xe",
            "chỗ để xe", "parking",
        ],
    },
}

# Precompiled word-boundary matchers per aspect.
#
# Two lessons are baked in here.
#
# 1. Match whole words. Plain substring matching filed a clause praising the
#    food under Ambience, because "ồn" (noisy) occurs inside "ngon" (tasty).
#
# 2. Match *with* diacritics. Accent folding is correct for place names — a
#    user really does type "Ha Noi" — but it destroys meaning in ordinary
#    Vietnamese, where minimal pairs are everywhere:
#        hôi (smelly)      vs hơi (slightly)
#        tiền (money)      vs tiện (convenient)
#        chỗ (seat)        vs chờ (to wait)
#    Folding made "Gửi xe hơi bất tiện" (awkward parking) read as a complaint
#    about smell *and* about price. So the primary matcher keeps diacritics,
#    and the folded matcher is a fallback used only for clauses written
#    without any diacritics at all.
def _build_matcher(keywords: list[str], folded: bool) -> re.Pattern[str]:
    prepare = tu.fold if folded else tu.normalize
    terms = sorted(
        {prepare(keyword).strip() for keyword in keywords if prepare(keyword).strip()},
        key=len,
        reverse=True,
    )
    return re.compile(
        "(?<!\\w)(?:" + "|".join(re.escape(term) for term in terms) + ")(?!\\w)"
    )


_ASPECT_MATCHERS: dict[str, re.Pattern[str]] = {
    aspect: _build_matcher(config["keywords"], folded=False)
    for aspect, config in ASPECTS.items()
}
_ASPECT_MATCHERS_FOLDED: dict[str, re.Pattern[str]] = {
    aspect: _build_matcher(config["keywords"], folded=True)
    for aspect, config in ASPECTS.items()
}

# Clause separators: Vietnamese reviews pivot on these.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.!?;\n]"
    r"|,\s*(?=nhưng|mà|tuy|song|có điều)"
    r"|\bnhưng\b|\btuy nhiên\b|\bmà\b|\bcó điều\b|\bngoài ra\b"
    r"|\bbut\b|\bhowever\b)",
    re.IGNORECASE,
)

MIN_CLAUSE_CHARS = 8
MAX_CLAUSE_CHARS = 220
MAX_QUOTES_PER_ASPECT = 2
# Below this many mentions an aspect is noise, not a signal.
MIN_MENTIONS = 2


def _split_clauses(text: str) -> list[str]:
    """Break a review into clauses worth scoring independently."""
    if not isinstance(text, str) or not text.strip():
        return []
    pieces = _CLAUSE_SPLIT_RE.split(text)
    clauses = []
    for piece in pieces:
        piece = re.sub(r"\s+", " ", (piece or "")).strip(" ,.:;-–—\"'")
        if MIN_CLAUSE_CHARS <= len(piece) <= MAX_CLAUSE_CHARS:
            clauses.append(piece)
        elif len(piece) > MAX_CLAUSE_CHARS:
            clauses.append(piece[:MAX_CLAUSE_CHARS].rsplit(" ", 1)[0])
    return clauses


def _aspects_of(clause: str) -> list[str]:
    """Which aspects a clause talks about, matched on whole words only."""
    normalized = tu.normalize(clause)
    hits = [
        aspect
        for aspect, matcher in _ASPECT_MATCHERS.items()
        if matcher.search(normalized)
    ]
    if hits:
        return hits
    # Nothing matched. If the clause carries no diacritics at all the author
    # typed without them, so retry accent-insensitively.
    folded = tu.fold(clause)
    if folded == normalized:
        return [
            aspect
            for aspect, matcher in _ASPECT_MATCHERS_FOLDED.items()
            if matcher.search(folded)
        ]
    return []


def _verdict(positive: int, negative: int) -> str:
    total = positive + negative
    if total == 0:
        return "mixed"
    ratio = positive / total
    if ratio >= 0.7:
        return "positive"
    if ratio <= 0.3:
        return "negative"
    return "mixed"


def summarize_reviews(reviews: list[dict], lang: str = "vi") -> dict:
    """Build an aspect-level digest from a list of reviews.

    ``reviews`` items need a ``noiDung`` (or ``content``) key; ``diemReview``
    (or ``rating``) is used when present.
    """
    texts: list[str] = []
    ratings: list[float] = []
    for review in reviews or []:
        content = review.get("noiDung") or review.get("content") or ""
        if isinstance(content, str) and content.strip():
            texts.append(content.strip())
        score = review.get("diemReview", review.get("rating"))
        try:
            if score is not None:
                ratings.append(float(score))
        except (TypeError, ValueError):
            pass

    if not texts:
        return {
            "available": False,
            "review_count": 0,
            "aspects": [],
            "summary": (
                "Chưa có đánh giá nào để phân tích."
                if lang == "vi"
                else "No reviews to analyse yet."
            ),
            "overall": {"positive": 0, "neutral": 0, "negative": 0},
            "average_rating": None,
        }

    # 1-2. clauses -> aspects
    clause_texts: list[str] = []
    clause_aspects: list[list[str]] = []
    for text in texts:
        for clause in _split_clauses(text):
            hits = _aspects_of(clause)
            if hits:
                clause_texts.append(clause)
                clause_aspects.append(hits)

    # 3. one batched forward pass over every clause plus every whole review
    whole_review_offset = len(clause_texts)
    batch = clause_texts + texts
    predictions = analyzer.analyze_batch(batch)
    clause_predictions = predictions[:whole_review_offset]
    review_predictions = predictions[whole_review_offset:]
    model_available = any(p.get("available") for p in predictions)

    # 4. aggregate
    buckets: dict[str, dict] = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0, "quotes": []}
    )
    for clause, hits, prediction in zip(
        clause_texts, clause_aspects, clause_predictions
    ):
        label = prediction["label"]
        for aspect in hits:
            bucket = buckets[aspect]
            if label == "POS":
                bucket["positive"] += 1
            elif label == "NEG":
                bucket["negative"] += 1
            else:
                bucket["neutral"] += 1
            bucket["quotes"].append(
                {"text": clause, "label": label, "score": prediction["score"]}
            )

    aspect_results = []
    for aspect, config in ASPECTS.items():
        bucket = buckets.get(aspect)
        if not bucket:
            continue
        mentions = bucket["positive"] + bucket["negative"] + bucket["neutral"]
        if mentions < MIN_MENTIONS:
            continue
        verdict = _verdict(bucket["positive"], bucket["negative"])
        # Quote the most confident clause matching the verdict, so the example
        # supports the headline rather than contradicting it.
        wanted = {"positive": "POS", "negative": "NEG"}.get(verdict)
        candidates = [
            quote for quote in bucket["quotes"]
            if wanted is None or quote["label"] == wanted
        ] or bucket["quotes"]
        candidates.sort(key=lambda quote: quote["score"], reverse=True)
        seen: set[str] = set()
        quotes = []
        for quote in candidates:
            key = tu.fold(quote["text"])[:60]
            if key in seen:
                continue
            seen.add(key)
            quotes.append(quote["text"])
            if len(quotes) >= MAX_QUOTES_PER_ASPECT:
                break

        positive_ratio = (
            bucket["positive"] / mentions if mentions else 0.0
        )
        aspect_results.append({
            "key": aspect,
            "label": config["label_vi" if lang == "vi" else "label_en"],
            "icon": config["icon"],
            "mentions": mentions,
            "positive": bucket["positive"],
            "neutral": bucket["neutral"],
            "negative": bucket["negative"],
            "positive_ratio": round(positive_ratio, 3),
            "verdict": verdict,
            "quotes": quotes,
        })

    # Most-discussed aspects first.
    aspect_results.sort(key=lambda item: item["mentions"], reverse=True)

    overall = {"positive": 0, "neutral": 0, "negative": 0}
    for prediction in review_predictions:
        if prediction["label"] == "POS":
            overall["positive"] += 1
        elif prediction["label"] == "NEG":
            overall["negative"] += 1
        else:
            overall["neutral"] += 1

    return {
        "available": model_available,
        "review_count": len(texts),
        "average_rating": (
            round(sum(ratings) / len(ratings), 2) if ratings else None
        ),
        "overall": overall,
        "aspects": aspect_results,
        "summary": _headline(overall, aspect_results, len(texts), lang),
    }


def _headline(overall: dict, aspects: list[dict], total: int, lang: str) -> str:
    """One sentence naming what people liked and what they did not."""
    positives = [a for a in aspects if a["verdict"] == "positive"]
    negatives = [a for a in aspects if a["verdict"] == "negative"]
    positive_share = overall["positive"] / total if total else 0.0

    def names(items: list[dict]) -> str:
        labels = [item["label"].lower() for item in items[:3]]
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + (" và " if lang == "vi" else " and ") + labels[-1]

    if lang == "vi":
        percent = round(positive_share * 100)
        parts = [f"{percent}% đánh giá là tích cực"]
        if positives:
            parts.append(f"thực khách khen {names(positives)}")
        if negatives:
            parts.append(f"nhưng còn phàn nàn về {names(negatives)}")
        if not positives and not negatives:
            parts.append("các ý kiến khá trái chiều")
        return ". ".join(
            [parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")]
        ) + "."

    percent = round(positive_share * 100)
    parts = [f"{percent}% of reviews are positive"]
    if positives:
        parts.append(f"diners praise {names(positives)}")
    if negatives:
        parts.append(f"but complain about {names(negatives)}")
    if not positives and not negatives:
        parts.append("opinions are mixed")
    return parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "") + "."
