# -*- coding: utf-8 -*-
"""A language model as a last-resort query parser.

The rules in ``intent.py`` and the concepts in ``knowledge.py`` read most
queries. Some they cannot: "chỗ nào ngồi lâu làm việc được", "món ăn tốt cho
người ốm". For those -- and only those, gated on the parser's own confidence --
the query text is sent to Gemini, which fills the same Intent slots.

What the model is allowed to do is deliberately narrow:

* It returns slots, never restaurants. The search, the filters and the ranking
  that follow are the same deterministic code every other query goes through,
  so nothing it says can put a place in front of a user that the data does not
  support.
* Its output is constrained twice. The request carries a JSON schema whose
  string fields are enums of values that exist in the data (dish tags,
  districts, aspects), and the reply is validated again here, because a
  schema is a request to the model and not a guarantee.
* It is optional. With no key, an exhausted daily cap, a timeout or an error,
  the rule-based intent is used unchanged and the user sees the same answer
  they would have before this module existed.

Only the query text is sent: no location, no account, no history.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone

import knowledge as kb
import text_utils as tu
from config import settings
from intent import Intent

logger = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ASPECTS = ["food", "price", "service", "space", "hygiene", "parking"]
TIME_TAGS = ["sáng", "trưa", "chiều", "tối", "ăn đêm"]
SORTS = ["relevance", "price", "rating", "distance"]

# Every value the model may use, taken from the knowledge base so the two
# can never drift apart.
DISHES = sorted(kb.DISH_TAGS - set(kb.ADJECTIVE_TAGS) - set(kb.TIME_TAGS))
ADJECTIVES = sorted(
    set(kb.ADJECTIVE_TAGS)
    | {adj for concept in kb.CONCEPTS for adj in concept.get("adjectives", [])}
)
DISTRICTS = sorted(kb.DISTRICT_TO_CITY)

PROMPT = """Bạn đọc câu hỏi tìm quán ăn (tiếng Việt hoặc tiếng Anh) và điền các trường tìm kiếm.

Quy tắc:
- Chỉ dùng giá trị có trong danh sách cho phép của từng trường. Không có giá trị phù hợp thì để trống.
- Không bao giờ nêu tên quán, điểm số hay địa chỉ.
- dishes: món người dùng muốn ăn, hoặc các món hợp với nhu cầu (ví dụ "tốt cho người ốm" -> cháo, súp). Tối đa 6.
- adjectives: đặc điểm quán mong muốn (dùng để xếp hạng, không lọc).
- aspects: khía cạnh cần được khen: food, price, service, space (không gian), hygiene, parking.
- exclude: món/đặc điểm người dùng KHÔNG muốn.
- price_max: giá tối đa (VND) nếu câu có ý về giá, ngược lại null.
- sort_by: "price" nếu ưu tiên rẻ, "rating" nếu ưu tiên ngon/nổi tiếng, "distance" nếu muốn gần, còn lại "relevance".
- clarify: true nếu câu không phải tìm quán ăn, hoặc quá mơ hồ để điền trường nào.
"""


def _schema() -> dict:
    def enum_list(values: list[str]) -> dict:
        return {"type": "ARRAY", "items": {"type": "STRING", "enum": values}}

    return {
        "type": "OBJECT",
        "properties": {
            "dishes": enum_list(DISHES),
            "adjectives": enum_list(ADJECTIVES),
            "aspects": enum_list(ASPECTS),
            "exclude": enum_list(DISHES + ADJECTIVES),
            "districts": enum_list(DISTRICTS),
            "time_tags": enum_list(TIME_TAGS),
            "price_max": {"type": "NUMBER", "nullable": True},
            "sort_by": {"type": "STRING", "enum": SORTS},
            "clarify": {"type": "BOOLEAN"},
        },
        "required": ["dishes", "adjectives", "aspects", "clarify"],
    }


class _Budget:
    """A per-UTC-day request counter, so traffic cannot exceed the free tier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._count = 0

    def take(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= settings.llm_daily_cap:
                return False
            self._count += 1
            return True

    @property
    def used(self) -> int:
        return self._count


_budget = _Budget()
# The same question asked twice gets the same reading without a second call.
_cache: "OrderedDict[str, dict | None]" = OrderedDict()
_CACHE_SIZE = 1000
_cache_lock = threading.Lock()
# After a failure (quota, outage) stop calling for a while instead of adding
# a timeout's worth of latency to every query.
_paused_until = 0.0
_PAUSE_SECONDS = 60


def enabled() -> bool:
    return bool(settings.gemini_api_key)


def status() -> dict:
    return {
        "ready": enabled(),
        "enabled": enabled(),
        "model": settings.llm_model if enabled() else None,
        "calls_today": _budget.used,
        "daily_cap": settings.llm_daily_cap,
        "paused": time.time() < _paused_until,
    }


def _call(model: str, query: str) -> dict:
    body = {
        "systemInstruction": {"parts": [{"text": PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": query[:300]}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": _schema(),
            "maxOutputTokens": 400,
        },
    }
    request = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=settings.llm_timeout_seconds) as response:
        payload = json.load(response)
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def _ask(query: str) -> dict | None:
    """The model's reading of ``query``, or None when it could not be had."""
    global _paused_until
    key = tu.normalize(query)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    if time.time() < _paused_until or not _budget.take():
        return None

    # Each model has its own free quota, so a 429 on the first is a reason
    # to try the second, not to give up.
    result: dict | None = None
    for model in (settings.llm_model, settings.llm_fallback_model):
        if not model:
            continue
        try:
            result = _call(model, query)
            break
        except urllib.error.HTTPError as error:
            logger.warning("LLM parser %s: HTTP %s", model, error.code)
        except Exception as error:  # noqa: BLE001 - never breaks a search
            logger.warning("LLM parser %s failed: %s", model, error)

    if result is None:
        # Both failed: pause, and do not cache the miss, so the same question
        # is read properly once the quota or the service is back.
        _paused_until = time.time() + _PAUSE_SECONDS
        return None

    with _cache_lock:
        _cache[key] = result
        if len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return result


def _clean(values, allowed: list[str], limit: int = 6) -> list[str]:
    """Keep only allowed values, deduplicated, in order."""
    allowed_set = set(allowed)
    out: list[str] = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, str) and value in allowed_set and value not in out:
            out.append(value)
    return out[:limit]


def refine(intent: Intent, query: str) -> Intent:
    """Fill ``intent`` from the model when the rules could not read the query.

    Adds to what the rules found rather than replacing it: a district or a
    price the rules did read stays as read.
    """
    if not enabled() or intent.confidence > settings.llm_confidence_gate:
        return intent
    reading = _ask(query)
    if not isinstance(reading, dict):
        return intent

    if reading.get("clarify") and not any(
        reading.get(field) for field in ("dishes", "adjectives", "aspects", "districts")
    ):
        intent.uncertainties.append("llm:clarify")
        return intent

    def add(target: list[str], values: list[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)

    add(intent.dishes, _clean(reading.get("dishes"), DISHES))
    add(intent.adjectives, _clean(reading.get("adjectives"), ADJECTIVES))
    add(intent.aspects, _clean(reading.get("aspects"), ASPECTS))
    add(intent.exclude, _clean(reading.get("exclude"), DISHES + ADJECTIVES))
    add(intent.time_tags, _clean(reading.get("time_tags"), TIME_TAGS, 2))
    for district in _clean(reading.get("districts"), DISTRICTS, 4):
        add(intent.districts, [district])
        city = kb.DISTRICT_TO_CITY.get(district)
        if city:
            add(intent.cities, [city])

    price = reading.get("price_max")
    if (
        intent.price_max is None
        and isinstance(price, (int, float))
        and 5_000 <= price <= 5_000_000
    ):
        intent.price_max = float(price)
    sort = reading.get("sort_by")
    if intent.sort_by == "relevance" and sort in SORTS:
        intent.sort_by = sort

    understood = intent.dishes or intent.adjectives or intent.aspects or intent.districts
    if understood:
        # The leftover words were the model's to read; keeping them as free
        # text would also keep treating the query as a restaurant's name.
        intent.name_like = False
        intent.free_text = ""
        intent.confidence = max(intent.confidence, 0.75)
        intent.uncertainties.append("llm_parsed")
    return intent
