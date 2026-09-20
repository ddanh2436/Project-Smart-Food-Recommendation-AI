# -*- coding: utf-8 -*-
"""Small, dependency-free text and geo helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable

from knowledge import strip_accents

__all__ = [
    "normalize",
    "fold",
    "find_phrases",
    "replace_phrases",
    "remove_phrases",
    "haversine_km",
    "clean_coordinate",
    "clean_price",
    "format_price",
]

_WHITESPACE_RE = re.compile(r"\s+")
# Keep Vietnamese letters, digits and spaces; turn everything else into a space
# so "bún bò,Q1!" tokenizes the same as "bún bò Q1".
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, NFC-normalize and collapse whitespace.

    NFC matters for Vietnamese: the same word can arrive pre-composed ("ộ")
    or decomposed ("o" + two combining marks), and the two never compare
    equal without this step.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def fold(text: str) -> str:
    """Normalize *and* drop diacritics, for accent-insensitive matching."""
    return strip_accents(normalize(text))


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    """Word-boundary matcher for a possibly multi-word phrase.

    The old code tested ``f" {phrase} " in f" {query} "``, which missed any
    phrase sitting next to punctuation ("phở, quận 1" never matched "phở").
    """
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


_REGEX_CACHE: dict[str, re.Pattern[str]] = {}


def _cached_regex(phrase: str) -> re.Pattern[str]:
    pattern = _REGEX_CACHE.get(phrase)
    if pattern is None:
        pattern = _phrase_regex(phrase)
        _REGEX_CACHE[phrase] = pattern
    return pattern


def find_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    """Return the phrases present in ``text``, consuming each match.

    ``phrases`` must be ordered longest-first so that "bún bò huế" is found
    and removed before "bún bò" gets a chance to match the same words.
    """
    haystack = text
    found: list[str] = []
    for phrase in phrases:
        if _cached_regex(phrase).search(haystack):
            found.append(phrase)
            haystack = _cached_regex(phrase).sub(" ", haystack)
    return found


def replace_phrases(text: str, mapping: dict[str, str], order: Iterable[str]) -> str:
    """Replace each phrase in ``order`` with ``mapping[phrase]``, once."""
    result = text
    for phrase in order:
        replacement = mapping[phrase]
        result = _cached_regex(phrase).sub(f" {replacement} ", result)
    return _WHITESPACE_RE.sub(" ", result).strip()


def remove_phrases(text: str, phrases: Iterable[str]) -> str:
    """Strip every given phrase from ``text``."""
    result = text
    for phrase in phrases:
        result = _cached_regex(phrase).sub(" ", result)
    return _WHITESPACE_RE.sub(" ", result).strip()


# --------------------------------------------------------------------------
# Geo
# --------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
UNKNOWN_DISTANCE = 9999.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km, or ``UNKNOWN_DISTANCE`` if unusable.

    Returns a large sentinel rather than raising so a single bad row cannot
    break a whole result page.
    """
    try:
        lat1, lon1, lat2, lon2 = (float(v) for v in (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError):
        return UNKNOWN_DISTANCE

    # (0, 0) is in the Atlantic; for Vietnamese data it always means "missing".
    if not all(map(math.isfinite, (lat1, lon1, lat2, lon2))):
        return UNKNOWN_DISTANCE
    if (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0):
        return UNKNOWN_DISTANCE
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return UNKNOWN_DISTANCE
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return UNKNOWN_DISTANCE

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def clean_coordinate(value) -> float:
    """Parse a latitude/longitude that may use a comma decimal separator."""
    try:
        if isinstance(value, (int, float)):
            return float(value) if math.isfinite(float(value)) else 0.0
        if isinstance(value, str):
            text = value.strip().replace(",", ".")
            return float(text) if text else 0.0
    except (TypeError, ValueError):
        pass
    return 0.0


# "100.000 - 200.000đ", "50k - 100k", "Dưới 100.000đ"
_PRICE_TOKEN_RE = re.compile(r"(\d+(?:[.,]\d+)*)\s*(k|nghìn|ngàn|tr|triệu)?", re.I)


def _price_token_to_vnd(number: str, unit: str | None) -> float:
    digits = number.replace(".", "").replace(",", "")
    try:
        amount = float(digits)
    except ValueError:
        return 0.0
    unit = (unit or "").lower()
    if unit in {"k", "nghìn", "ngàn"}:
        amount *= 1_000
    elif unit in {"tr", "triệu"}:
        amount *= 1_000_000
    return amount


def clean_price(value) -> float:
    """Best-effort average price in VND from a free-text price field.

    The stored values look like "100.000 - 200.000đ". The old implementation
    concatenated every digit in the string, turning that into the nonsense
    number 100000200000. This parses each amount and averages the range.
    """
    if isinstance(value, (int, float)):
        try:
            return float(value) if math.isfinite(float(value)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if not isinstance(value, str) or not value.strip():
        return 0.0

    amounts = [
        _price_token_to_vnd(match.group(1), match.group(2))
        for match in _PRICE_TOKEN_RE.finditer(value)
    ]
    amounts = [a for a in amounts if a > 0]
    if not amounts:
        return 0.0
    # A bare "50" almost certainly means 50k in this dataset.
    amounts = [a * 1_000 if a < 1_000 else a for a in amounts]
    return sum(amounts) / len(amounts)


def format_price(vnd: float) -> str:
    """Render a VND amount the way Vietnamese menus do."""
    if not vnd or vnd <= 0:
        return "Đang cập nhật"
    if vnd >= 1_000_000:
        return f"{vnd / 1_000_000:.1f} triệu đ".replace(".0 ", " ")
    return f"{int(round(vnd / 1_000))}k"
