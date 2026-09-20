# -*- coding: utf-8 -*-
"""Intent extraction for a natural-language restaurant query.

This is the "semantic brain" that ``AI_Workflow.md`` described but never
shipped. It runs entirely offline — no API key, no per-request cost — and
turns a free-text query into the structured slots the ranker needs:

    "bún bò huế ngon ở quận 1 dưới 100k, không cay"
    -> dishes=['bún bò huế'] districts=['Quận 1'] sort_by='rating'
       price_max=100000 exclude=['cay']

Improvements over the original inline parsing in ``api.py``:

* word-boundary matching, so "phở, quận 1" finds both "phở" and "Quận 1"
  (the old ``f" {tag} " in query`` form missed anything next to punctuation);
* negation — "không cay", "trừ hải sản", "no pork" become exclusions instead
  of being silently treated as things the user *wants*;
* explicit price constraints — "dưới 100k", "từ 50k đến 200k", "tầm 2 trăm";
* a minimum-rating constraint — "trên 8 điểm", "8.5 trở lên";
* open-now and party-size hints;
* the leftover free text is kept, so semantic search can still use whatever
  the rules did not understand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import knowledge as kb
import text_utils as tu

# --------------------------------------------------------------------------
# Sort intent. Checked longest-phrase-first within each group.
# --------------------------------------------------------------------------
SORT_DISTANCE_CUES = [
    "gần đây nhất", "gần nhất", "gần đây", "quanh đây", "xung quanh",
    "cạnh đây", "gần tôi", "gần", "nearby", "closest", "near me",
]
SORT_PRICE_CUES = [
    "rẻ nhất", "giá rẻ", "bình dân", "tiết kiệm", "sinh viên", "rẻ",
    "cheapest", "budget",
]
SORT_RATING_CUES = [
    "ngon nhất", "tốt nhất", "đỉnh nhất", "nổi tiếng nhất", "review tốt",
    "đánh giá cao", "được đánh giá", "nổi tiếng", "chất lượng nhất",
    "ngon", "tốt", "best", "top rated", "highest rated",
]

# --------------------------------------------------------------------------
# Negation. A cue applies to the phrase that follows it, within a window,
# because Vietnamese negation is prefixed: "không ăn được hải sản".
# --------------------------------------------------------------------------
NEGATION_CUES = [
    "không thích", "không ăn được", "không ăn", "không muốn", "không có",
    "chẳng thích", "đừng", "ngoại trừ", "ngoài trừ", "trừ", "không",
    "no ", "without ", "except ", "not ",
]
NEGATION_WINDOW = 28  # characters after the cue that the negation covers

# --------------------------------------------------------------------------
# Price constraints.
# --------------------------------------------------------------------------
_AMOUNT = r"(\d+(?:[.,]\d+)*)\s*(k|nghìn|ngàn|tr|triệu|đồng|đ|vnd)?"

_PRICE_RANGE_RE = re.compile(
    rf"(?:từ|khoảng|tầm|between)?\s*{_AMOUNT}\s*(?:-|–|đến|tới|to)\s*{_AMOUNT}",
    re.IGNORECASE,
)
_PRICE_MAX_RE = re.compile(
    rf"(?:dưới|ít hơn|không quá|tối đa|under|below|less than|max)\s*{_AMOUNT}",
    re.IGNORECASE,
)
_PRICE_MIN_RE = re.compile(
    rf"(?:trên|hơn|từ|ít nhất|above|over|at least|min)\s*{_AMOUNT}",
    re.IGNORECASE,
)
_PRICE_ABOUT_RE = re.compile(
    rf"(?:khoảng|tầm|cỡ|độ|around|about)\s*{_AMOUNT}",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# Rating and party size.
# --------------------------------------------------------------------------
_RATING_RE = re.compile(
    r"(?:trên|hơn|từ|above|over)?\s*(\d(?:[.,]\d)?)\s*"
    r"(?:điểm|sao|\/10|trở lên|\+)",
    re.IGNORECASE,
)
_PARTY_RE = re.compile(
    r"(\d+)\s*(?:người|khách|bạn|đứa|người ăn|pax|people|persons?)",
    re.IGNORECASE,
)
_OPEN_NOW_CUES = [
    "đang mở", "mở cửa", "giờ này", "bây giờ", "còn mở", "open now",
    "đang bán",
]

# Filler words that carry no search signal. Stripped from the leftover text so
# ``free_text`` holds only genuinely unrecognised content (usually a name).
STOPWORDS = [
    # Vietnamese
    "quán", "tiệm", "nhà hàng", "hàng", "chỗ", "địa điểm", "khu vực", "khu",
    "ăn", "uống", "tìm", "kiếm", "gợi ý", "giới thiệu", "cho tôi", "cho mình",
    "giúp tôi", "giúp mình", "muốn", "cần", "thích", "hãy", "làm ơn",
    "tôi", "mình", "tớ", "em", "anh", "chị", "bạn", "cho", "với", "và",
    "ở", "tại", "vùng", "gần", "đồ", "món", "loại", "kiểu",
    "gì", "nào", "đó", "này", "thì", "là", "có", "được", "hơi", "khá",
    "giá", "mức giá", "điểm", "sao", "người", "phần", "suất",
    "nhất", "rất", "lắm", "quá", "hơn", "trên", "dưới", "khoảng", "tầm",
    # English
    "restaurant", "restaurants", "place", "places", "food", "eat", "find",
    "search", "want", "need", "like", "looking", "for", "some", "any",
    "give", "show", "me", "please", "the", "a", "an", "in", "at", "on",
    "near", "with", "and", "or", "of", "to", "is", "are", "best", "good",
]


def _to_vnd(number: str, unit: str | None) -> float:
    digits = number.replace(".", "").replace(",", "")
    try:
        amount = float(digits)
    except ValueError:
        return 0.0
    unit = (unit or "").lower()
    if unit in {"k", "nghìn", "ngàn"}:
        return amount * 1_000
    if unit in {"tr", "triệu"}:
        return amount * 1_000_000
    # No unit: a small bare number in a food query means thousands.
    return amount * 1_000 if amount < 1_000 else amount


@dataclass
class Intent:
    """Structured form of a user query."""

    raw_query: str = ""
    normalized_query: str = ""
    # What the user wants to eat.
    dishes: list[str] = field(default_factory=list)
    # Preferences that rank but never filter ("ngon", "sang trọng").
    adjectives: list[str] = field(default_factory=list)
    # Hard time filters ("ăn đêm", "sáng").
    time_tags: list[str] = field(default_factory=list)
    # Things the user explicitly does not want.
    exclude: list[str] = field(default_factory=list)
    # Places.
    districts: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    # Constraints.
    price_min: float | None = None
    price_max: float | None = None
    min_rating: float | None = None
    party_size: int | None = None
    open_now: bool = False
    # How to order results: relevance | distance | price | rating.
    sort_by: str = "relevance"
    # Whatever the rules did not consume, for semantic search to use.
    free_text: str = ""
    # True when the query looks like a restaurant name rather than a dish.
    name_like: bool = False

    @property
    def has_constraints(self) -> bool:
        return bool(
            self.dishes
            or self.time_tags
            or self.districts
            or self.cities
            or self.price_min
            or self.price_max
            or self.min_rating
            or self.exclude
        )

    @property
    def search_text(self) -> str:
        """The text used for TF-IDF / embedding similarity."""
        parts = self.dishes + self.adjectives + self.time_tags
        if self.free_text:
            parts.append(self.free_text)
        return " ".join(parts).strip() or self.normalized_query

    def to_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "dishes": self.dishes,
            "adjectives": self.adjectives,
            "time_tags": self.time_tags,
            "exclude": self.exclude,
            "districts": self.districts,
            "cities": self.cities,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "min_rating": self.min_rating,
            "party_size": self.party_size,
            "open_now": self.open_now,
            "sort_by": self.sort_by,
            "free_text": self.free_text,
            "name_like": self.name_like,
        }


def _negated_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges that a negation cue applies to."""
    spans: list[tuple[int, int]] = []
    for cue in NEGATION_CUES:
        for match in re.finditer(rf"(?<!\w){re.escape(cue)}", text, re.IGNORECASE):
            start = match.end()
            spans.append((start, min(len(text), start + NEGATION_WINDOW)))
    return spans


def _is_negated(text: str, phrase: str, spans: list[tuple[int, int]]) -> bool:
    if not spans:
        return False
    for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I):
        for start, end in spans:
            if start <= match.start() < end:
                return True
    return False


def _detect_sort(text: str, has_gps: bool) -> str:
    """Pick a ranking criterion from explicit cues in the query."""
    for cue in sorted(SORT_DISTANCE_CUES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text, re.I):
            # "gần" without a location is only meaningful with coordinates.
            return "distance" if has_gps else "relevance"
    for cue in sorted(SORT_PRICE_CUES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text, re.I):
            return "price"
    for cue in sorted(SORT_RATING_CUES, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text, re.I):
            return "rating"
    return "relevance"


def _parse_price(text: str) -> tuple[float | None, float | None, str]:
    """Extract (min, max) price bounds in VND, plus ``text`` minus the match.

    Returning the remainder matters: without it the matched words ("dưới
    100k") leak into ``free_text`` and pollute the similarity search.
    """
    match = _PRICE_RANGE_RE.search(text)
    if match:
        low = _to_vnd(match.group(1), match.group(2))
        high = _to_vnd(match.group(3), match.group(4))
        if low and high:
            return (min(low, high), max(low, high), _cut(text, match))

    match = _PRICE_MAX_RE.search(text)
    if match:
        value = _to_vnd(match.group(1), match.group(2))
        if value:
            return (None, value, _cut(text, match))

    match = _PRICE_ABOUT_RE.search(text)
    if match:
        target = _to_vnd(match.group(1), match.group(2))
        if target:
            # "khoảng 100k" -> accept +/- 40%.
            return (target * 0.6, target * 1.4, _cut(text, match))

    match = _PRICE_MIN_RE.search(text)
    if match:
        value = _to_vnd(match.group(1), match.group(2))
        if value:
            return (value, None, _cut(text, match))

    return (None, None, text)


def _parse_rating(text: str) -> tuple[float | None, str]:
    """Extract a minimum rating, plus ``text`` minus the matched phrase.

    Runs *before* price parsing: "trên 8 điểm" is a rating floor, but
    ``_PRICE_MIN_RE`` would otherwise read the same "trên 8" as 8,000 VND.
    """
    for match in _RATING_RE.finditer(text):
        try:
            value = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        if 0 < value <= 10:
            return (value, _cut(text, match))
    return (None, text)


def _cut(text: str, match: re.Match[str]) -> str:
    """Remove a matched span from ``text``."""
    return f"{text[: match.start()]} {text[match.end():]}"


def parse_intent(query: str, has_gps: bool = False) -> Intent:
    """Turn a free-text query into an :class:`Intent`.

    Order matters: English is translated first, then synonyms are folded to
    canonical tags, then the longest phrases are consumed before shorter ones
    so "bún bò huế" is never shredded into "bún" + "bò".
    """
    intent = Intent(raw_query=query or "")
    normalized = tu.normalize(query)
    if not normalized:
        return intent

    # 1. English -> Vietnamese, then colloquial -> canonical.
    translated = tu.replace_phrases(normalized, kb.EN_VI_MAPPING, kb.EN_VI_SORTED)
    translated = tu.replace_phrases(
        translated, kb.TAG_SYNONYMS, kb.TAG_SYNONYMS_SORTED
    )
    intent.normalized_query = translated

    # 2. Numeric constraints, read before any phrase is consumed. Each parser
    #    returns the text minus its own match, so "trên 8 điểm" is claimed by
    #    the rating rule and never re-read as a price, and none of the matched
    #    wording survives into ``free_text``.
    stripped = translated
    intent.min_rating, stripped = _parse_rating(stripped)
    intent.price_min, intent.price_max, stripped = _parse_price(stripped)
    party = _PARTY_RE.search(stripped)
    if party:
        try:
            size = int(party.group(1))
            intent.party_size = size if 1 <= size <= 100 else None
        except ValueError:
            pass
        stripped = _cut(stripped, party)
    intent.open_now = any(
        re.search(rf"(?<!\w){re.escape(cue)}", translated, re.I)
        for cue in _OPEN_NOW_CUES
    )

    # 3. Sort criterion, before the cue words get stripped out.
    intent.sort_by = _detect_sort(translated, has_gps)

    # 4. Negation spans, computed on the full text for correct positions.
    negations = _negated_spans(translated)

    # 5. Places. Longest alias first so "quận 12" beats "quận 1".
    remaining = stripped
    for alias in kb.LOCATION_ALIASES_SORTED:
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, remaining, re.IGNORECASE):
            canonical = kb.LOCATION_ALIASES[alias]
            if canonical in kb.MAJOR_CITIES:
                if canonical not in intent.cities:
                    intent.cities.append(canonical)
            elif canonical not in intent.districts:
                intent.districts.append(canonical)
                # A district implies its city, which widens a too-narrow filter.
                city = kb.DISTRICT_TO_CITY.get(canonical)
                if city and city not in intent.cities:
                    intent.cities.append(city)
            remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)

    # 6. Tags. Longest first, consuming each match.
    for tag in kb.CANDIDATE_TAGS_SORTED:
        pattern = rf"(?<!\w){re.escape(tag)}(?!\w)"
        if not re.search(pattern, remaining, re.IGNORECASE):
            continue
        remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)

        if _is_negated(translated, tag, negations):
            if tag not in intent.exclude:
                intent.exclude.append(tag)
        elif tag in kb.TIME_TAGS:
            if tag not in intent.time_tags:
                intent.time_tags.append(tag)
        elif tag in kb.ADJECTIVE_TAGS:
            if tag not in intent.adjectives:
                intent.adjectives.append(tag)
        elif tag not in intent.dishes:
            intent.dishes.append(tag)

    # 7. Leftover words, minus the constraint phrasing we already consumed.
    noise = (
        SORT_DISTANCE_CUES + SORT_PRICE_CUES + SORT_RATING_CUES
        + NEGATION_CUES + _OPEN_NOW_CUES
        + STOPWORDS
    )
    leftover = tu.remove_phrases(remaining, sorted(noise, key=len, reverse=True))
    leftover = re.sub(r"[^\w\s]", " ", leftover)
    # Drop leftover amounts ("100k", "200 nghìn") and bare numbers.
    leftover = re.sub(
        r"(?<!\w)\d+(?:[.,]\d+)*\s*(?:k|nghìn|ngàn|tr|triệu|đ|vnd)?(?!\w)",
        " ",
        leftover,
        flags=re.IGNORECASE,
    )
    intent.free_text = re.sub(r"\s+", " ", leftover).strip()

    # 8. Two or more unrecognised words usually means a proper name
    #    ("Phở Thìn Lò Đúc", "Cục Gạch Quán") rather than a category, so the
    #    ranker should boost exact name matches for this query.
    intent.name_like = len(intent.free_text.split()) >= 2

    return intent
