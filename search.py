# -*- coding: utf-8 -*-
"""Filter-then-rank recommendation engine.

Implements the philosophy stated in ``AI_Workflow.md`` — *filter first, then
rank* — with a hybrid relevance score.

Pipeline
--------
1. **Hard filters** (a row either qualifies or it does not): location, time
   tags, explicit exclusions, price bounds, minimum rating, open-now.
   Each filter is skipped rather than applied if it would empty the result
   set, so a query is never answered with nothing just because one
   over-specific constraint failed.
2. **Relevance** = weighted blend of
   * exact / prefix name match (dominant, for "Phở Thìn Lò Đúc"),
   * dish-tag overlap,
   * TF-IDF char-ngram similarity (robust to missing diacritics),
   * embedding cosine similarity (true synonyms: "đồ biển" ~ "hải sản"),
   * small quality and proximity priors -- the quality prior uses the
     review-count-shrunk rating, not the raw one, so a 10.0 backed by a single
     review cannot outrank a well-reviewed 8.
3. **Ordering** by the criterion the intent asked for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import knowledge as kb
import text_utils as tu
from config import settings
from data_store import store
from intent import Intent, parse_intent

logger = logging.getLogger(__name__)

# Relevance weights. Name match dominates so that searching a restaurant by
# name always surfaces it, whatever its rating.
W_NAME_EXACT = 6.0
W_NAME_PREFIX = 3.0
W_NAME_PARTIAL = 1.8
W_DISH_TAG = 1.2
W_DISH_IN_NAME = 0.9
W_TFIDF = 1.0
W_SEMANTIC = 1.5
W_ADJECTIVE = 0.35
W_RATING_PRIOR = 0.25
W_PROXIMITY_PRIOR = 0.30

# Rating weight used once the dish filter has already matched.
#
# At that point every surviving row genuinely serves the dish, so lexical
# similarity has nothing left to contribute and merely favours short names:
# a search for "phở" led with a 5.0-rated "Phở Cuốn Hà Nội" ahead of 350 other
# pho places, purely because its name is mostly the word "phở". With the
# category settled, quality is what the user is actually choosing on.
#
# Deliberately NOT applied to name-like queries. Raising this weight globally
# was tried and broke name search outright -- "Phở Thìn Lò Đúc" started
# returning "Phở Cần Thơ" -- so it is gated on `intent.name_like` being false.
W_RATING_PRIOR_CATEGORICAL = 3.0

# Rows farther than this are dropped when the user asked for "near me".
NEARBY_RADIUS_KM = 20.0

# Minimum query-match evidence (priors excluded) for a result to count as a
# real hit. char-ngram TF-IDF gives every row a small non-zero score, so the
# floor has to sit above that noise rather than at zero.
#
# Calibrated against the real 5,700-row dataset rather than guessed. Measured
# peak content score per query:
#     real      "bún bò huế" 8.62 · "phở" 5.72 · "bánh mì" 5.58
#               "cà phê trứng" 3.13 · "Phở Thìn Lò Đúc" 2.80 · worst 1.33
#     nonsense  "zzzqqq" 0.28 · "qwertyuiop" 0.16 · "asdfgh" 0.14
# 0.8 sits ~3x above the nonsense ceiling and ~1.6x below the weakest real
# query. The original 0.12 was inside the noise band, so a nonsense query
# returned the whole database ranked by rating.
MIN_CONTENT_SCORE = 0.8


@dataclass
class SearchResult:
    intent: Intent
    rows: pd.DataFrame
    total_before_ranking: int
    relaxed_filters: list[str]

    @property
    def sort_by(self) -> str:
        return self.intent.sort_by


def _distance_column(frame: pd.DataFrame, gps: list[float] | None) -> pd.Series:
    """Per-row distance from the user, or the unknown sentinel."""
    if not gps or len(gps) != 2:
        return pd.Series(
            [tu.UNKNOWN_DISTANCE] * len(frame), index=frame.index, dtype="float64"
        )
    user_lat, user_lon = gps[0], gps[1]
    # Vectorized haversine: the original applied a Python function per row,
    # which dominated request time on a few thousand restaurants.
    lat = np.radians(frame["lat"].to_numpy(dtype="float64"))
    lon = np.radians(frame["lon"].to_numpy(dtype="float64"))
    lat0 = np.radians(float(user_lat))
    lon0 = np.radians(float(user_lon))
    d_lat = lat - lat0
    d_lon = lon - lon0
    a = (
        np.sin(d_lat / 2) ** 2
        + np.cos(lat0) * np.cos(lat) * np.sin(d_lon / 2) ** 2
    )
    distance = tu.EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Rows with no usable coordinates must not look like they are next door.
    missing = (
        (frame["lat"].to_numpy() == 0) & (frame["lon"].to_numpy() == 0)
    ) | ~np.isfinite(distance)
    distance = np.where(missing, tu.UNKNOWN_DISTANCE, distance)
    return pd.Series(distance, index=frame.index, dtype="float64")


def _apply_filter(
    frame: pd.DataFrame,
    mask: pd.Series,
    label: str,
    relaxed: list[str],
    minimum: int = 1,
) -> pd.DataFrame:
    """Apply ``mask`` unless doing so would leave fewer than ``minimum`` rows.

    Dropping an over-specific constraint beats returning an empty page: a
    user asking for "bún bò dưới 30k ở Quận 1" would rather see the cheapest
    bún bò in Quận 1 than nothing at all. Anything relaxed is reported back
    so the UI can say so.
    """
    filtered = frame[mask]
    if len(filtered) >= minimum:
        return filtered
    relaxed.append(label)
    return frame


def _location_mask(frame: pd.DataFrame, intent: Intent) -> pd.Series | None:
    """District match preferred; fall back to the whole city."""
    if intent.districts:
        mask = frame["district"].isin(intent.districts)
        if mask.any():
            return mask
    if intent.cities:
        return frame["city"].isin(intent.cities)
    if intent.districts:
        return frame["district"].isin(intent.districts)
    return None


def _tag_contains(frame: pd.DataFrame, needles: list[str]) -> pd.Series:
    """True where any needle appears in the row's tags or name."""
    if not needles:
        return pd.Series(True, index=frame.index)
    blob = frame["tags_norm"] + " " + frame["name_norm"]
    mask = pd.Series(False, index=frame.index)
    for needle in needles:
        mask |= blob.str.contains(tu.normalize(needle), regex=False, na=False)
    return mask


def _broaden(dish: str) -> list[str]:
    """Progressively more general forms of a dish name, most specific first.

    "cà phê trứng" -> ["cà phê trứng", "cà phê"]

    Used so a narrow dish that nothing matches falls back to its category
    instead of being abandoned. Only prefixes that are themselves known tags
    are offered, so "bún bò" is a valid fallback for "bún bò huế" but
    "hột vịt" is not invented as a fallback for "hột vịt lộn".
    """
    words = dish.split()
    forms = [dish]
    for cut in range(len(words) - 1, 0, -1):
        candidate = " ".join(words[:cut])
        if candidate in kb.DISH_TAGS:
            forms.append(candidate)
    return forms


def _apply_dish_filter(
    frame: pd.DataFrame, intent: Intent, relaxed: list[str]
) -> pd.DataFrame:
    """Keep rows serving the requested dish, broadening before giving up.

    For "cà phê trứng ở Hoàn Kiếm" there is no egg coffee in Hoàn Kiếm, so the
    plain filter emptied the set, relaxed itself, and the query was answered
    with a pizza place — the highest-rated row in the district. Now it retries
    with "cà phê" first; only if even that finds nothing is the filter recorded
    as relaxed, and the caller then applies the relevance floor so a
    dish-less answer comes back empty rather than arbitrary.
    """
    ladders = [_broaden(dish) for dish in intent.dishes]
    # Walk broadening levels together. zip() would truncate to the shortest
    # ladder, so a two-dish query where only one dish can broaden would never
    # reach the broadened level at all; clamping each ladder to its most
    # general form instead keeps every level reachable.
    depth = max(len(ladder) for ladder in ladders)
    for level in range(depth):
        attempt = [ladder[min(level, len(ladder) - 1)] for ladder in ladders]
        matches = frame[_tag_contains(frame, attempt)]
        if not matches.empty:
            if attempt != intent.dishes:
                relaxed.append("dish_broadened")
            return matches

    relaxed.append("dish")
    return frame


def _is_open_now(hours: str) -> bool:
    """Parse "07:00 - 22:00 | 06:00-11:00" style strings.

    Unknown hours count as open: the original returned ``False``, which
    silently hid every restaurant whose hours had not been crawled.
    """
    if not isinstance(hours, str) or not hours.strip():
        return True
    from datetime import datetime

    now = datetime.now()
    minutes_now = now.hour * 60 + now.minute
    for window in hours.replace("|", ",").split(","):
        parts = [part.strip() for part in window.split("-")]
        if len(parts) != 2:
            continue
        try:
            start_h, start_m = (int(v) for v in parts[0].split(":")[:2])
            end_h, end_m = (int(v) for v in parts[1].split(":")[:2])
        except (ValueError, IndexError):
            continue
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if start <= end:
            if start <= minutes_now <= end:
                return True
        else:  # window crosses midnight
            if minutes_now >= start or minutes_now <= end:
                return True
    return False


def _relevance(
    frame: pd.DataFrame, intent: Intent, dish_matched: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Blended relevance for every row.

    Returns ``(total, content)``. ``content`` excludes the rating and
    proximity priors, so the caller can tell "this row actually matches the
    query" apart from "this row is merely popular" — a distinction the ranker
    needs to answer a nonsense query with nothing instead of with everything.
    """
    size = len(frame)
    if size == 0:
        return np.zeros(0, dtype="float32"), np.zeros(0, dtype="float32")

    scores = np.zeros(size, dtype="float32")
    positions = frame.index.to_numpy()

    # --- lexical + semantic similarity, computed over the whole store then
    #     projected onto the surviving rows (cheaper than re-vectorizing).
    query_text = intent.search_text
    if query_text:
        full_frame_size = len(store.frame)
        tfidf = store.tfidf_scores(query_text)
        if len(tfidf) == full_frame_size:
            scores += W_TFIDF * tfidf[positions].astype("float32")
        semantic = store.semantic_scores(query_text)
        if len(semantic) == full_frame_size:
            scores += W_SEMANTIC * semantic[positions].astype("float32")

    # --- name matching, the strongest signal for a named search
    name_query = tu.normalize(intent.raw_query)
    if len(name_query) >= 3:
        names = frame["name_norm"].to_numpy()
        folded_query = tu.fold(name_query)
        for i, name in enumerate(names):
            if not name:
                continue
            if name == name_query:
                scores[i] += W_NAME_EXACT
            elif name.startswith(name_query) or name_query.startswith(name):
                scores[i] += W_NAME_PREFIX
            elif name_query in name or tu.fold(name).find(folded_query) >= 0:
                scores[i] += W_NAME_PARTIAL

    # --- dish tags
    if intent.dishes:
        tags = frame["tags_norm"].to_numpy()
        names = frame["name_norm"].to_numpy()
        for dish in intent.dishes:
            needle = tu.normalize(dish)
            for i in range(size):
                if needle and needle in tags[i]:
                    scores[i] += W_DISH_TAG
                if needle and needle in names[i]:
                    scores[i] += W_DISH_IN_NAME

    # --- soft preferences
    if intent.adjectives:
        tags = frame["tags_norm"].to_numpy()
        for adjective in intent.adjectives:
            needle = tu.normalize(adjective)
            for i in range(size):
                if needle and needle in tags[i]:
                    scores[i] += W_ADJECTIVE

    # Everything above is evidence that the row matches the query itself.
    content = scores.copy()

    # --- quality prior, normalized to 0..1.
    # Uses the shrunk rating the backend computed (see the shrinkage note in
    # data_store): the raw value would let a 10.0 backed by a single review
    # outrank a well-reviewed 7.9.
    #
    # Weight depends on whether the category is already settled. Once the dish
    # filter has matched, every candidate is relevant and quality decides;
    # otherwise the prior stays small so it cannot outweigh a name match.
    rating_column = (
        "rating_adjusted" if "rating_adjusted" in frame.columns else "rating"
    )
    rating = frame[rating_column].to_numpy(dtype="float32")
    rating_weight = (
        W_RATING_PRIOR_CATEGORICAL
        if (dish_matched and not intent.name_like)
        else W_RATING_PRIOR
    )
    scores += rating_weight * np.clip(rating / 10.0, 0.0, 1.0)

    # --- proximity prior, only when a distance is actually known
    if "distance_km" in frame.columns:
        distance = frame["distance_km"].to_numpy(dtype="float32")
        known = distance < tu.UNKNOWN_DISTANCE
        prior = np.zeros(size, dtype="float32")
        prior[known] = 1.0 / (1.0 + distance[known] / 5.0)
        scores += W_PROXIMITY_PRIOR * prior

    return scores, content


def _order(frame: pd.DataFrame, intent: Intent) -> pd.DataFrame:
    """Sort by the criterion the query asked for, with sensible tie-breaks."""
    if intent.sort_by == "distance":
        return frame.sort_values(
            ["distance_km", "relevance"], ascending=[True, False]
        )
    if intent.sort_by == "price":
        # Rows with an unknown price go last rather than first.
        frame = frame.assign(
            _price_key=frame["price"].replace(0, np.nan)
        )
        return frame.sort_values(
            ["_price_key", "rating"], ascending=[True, False], na_position="last"
        ).drop(columns="_price_key")
    if intent.sort_by == "rating":
        # "ngon nhất" should mean reliably well rated, so order by the shrunk
        # rating and break ties on the raw one.
        key = "rating_adjusted" if "rating_adjusted" in frame.columns else "rating"
        return frame.sort_values(
            [key, "rating", "relevance"], ascending=[False, False, False]
        )
    return frame.sort_values(
        ["relevance", "rating"], ascending=[False, False]
    )


def search(
    query: str,
    user_gps: list[float] | None = None,
    city_filter: str | None = None,
    limit: int | None = None,
    candidate_ids: list[str] | None = None,
) -> SearchResult:
    """Run the full filter-then-rank pipeline for ``query``."""
    store.ensure_fresh()
    frame = store.frame
    has_gps = bool(user_gps and len(user_gps) == 2)
    intent = parse_intent(query, has_gps=has_gps)

    if frame.empty:
        return SearchResult(intent, frame, 0, [])

    working = frame.copy()
    working["distance_km"] = _distance_column(working, user_gps if has_gps else None)
    relaxed: list[str] = []

    # An explicit candidate list from the caller wins over everything.
    if candidate_ids:
        subset = working[working["id"].isin(candidate_ids)]
        if not subset.empty:
            working = subset

    # --- 1. hard filters -------------------------------------------------
    location_mask = _location_mask(working, intent)
    if location_mask is not None:
        working = _apply_filter(working, location_mask, "location", relaxed)
    elif city_filter:
        canonical = kb.CITY_KEY_ALIASES.get(
            tu.normalize(city_filter).replace(" ", ""), city_filter
        )
        pattern = kb.CITY_PATTERNS.get(canonical)
        mask = (
            working["address"].str.contains(pattern, na=False)
            if pattern is not None
            else working["city"].str.contains(canonical, case=False, na=False)
        )
        working = _apply_filter(working, mask, "city_filter", relaxed)
    elif intent.sort_by == "distance" and has_gps:
        working = _apply_filter(
            working, working["distance_km"] <= NEARBY_RADIUS_KM, "radius", relaxed
        )

    if intent.time_tags:
        working = _apply_filter(
            working, _tag_contains(working, intent.time_tags), "time", relaxed
        )

    if intent.exclude:
        working = _apply_filter(
            working, ~_tag_contains(working, intent.exclude), "exclude", relaxed
        )

    # The dish is the primary intent, so it is filtered *before* the secondary
    # budget and rating constraints. Filtering price first was actively wrong:
    # for "hải sản dưới 50k" the only seafood place was dropped for being over
    # budget, the dish filter then had nothing left to keep and relaxed itself,
    # and the query answered with a coffee shop. This order instead keeps the
    # seafood and relaxes the budget, which is what the user meant.
    if intent.dishes:
        working = _apply_dish_filter(working, intent, relaxed)
    # A clean dish match means the candidate set is already on-topic.
    dish_matched = bool(intent.dishes) and "dish" not in relaxed

    if intent.price_max:
        # Keep rows with no known price: absence of data is not a violation.
        mask = (working["price"] <= intent.price_max) | (working["price"] <= 0)
        working = _apply_filter(working, mask, "price_max", relaxed)
    if intent.price_min:
        mask = (working["price"] >= intent.price_min) | (working["price"] <= 0)
        working = _apply_filter(working, mask, "price_min", relaxed)

    if intent.min_rating:
        working = _apply_filter(
            working, working["rating"] >= intent.min_rating, "min_rating", relaxed
        )

    if intent.open_now:
        mask = working["opening_hours"].apply(_is_open_now)
        working = _apply_filter(working, mask, "open_now", relaxed)

    total_before_ranking = len(working)
    if working.empty:
        return SearchResult(intent, working, 0, relaxed)

    # --- 2. rank ---------------------------------------------------------
    total_scores, content_scores = _relevance(working, intent, dish_matched)
    working = working.assign(relevance=total_scores)

    # Two cases where a result set has to be proven relevant rather than just
    # ordered. In both, answering with whatever scored highest on the rating
    # prior looks like a broken search, so it is better to return nothing and
    # let the UI say "not found".
    #
    #  1. Nothing in the query was recognised and no text matches -- a nonsense
    #     query such as "zzzqqq".
    #  2. The dish filter was relaxed entirely, so the candidates are merely
    #     the right *place*, with no evidence of the right food.
    #
    # An explicit ranking counts as expressed intent even with no dish named:
    # "đồ ăn gần đây" asks to browse by distance and must not be rejected for
    # weak text similarity, whereas "zzzqqq" expresses nothing at all.
    dish_unmatched = "dish" in relaxed
    expressed_intent = intent.has_constraints or intent.sort_by != "relevance"
    if query and query.strip() and (not expressed_intent or dish_unmatched):
        if content_scores.max(initial=0.0) < MIN_CONTENT_SCORE:
            return SearchResult(intent, working.iloc[0:0], 0, relaxed)
        if dish_unmatched:
            # Some rows do carry query evidence: keep only those, so a search
            # for coffee never answers with the district's best pizza.
            keep = content_scores >= MIN_CONTENT_SCORE
            working = working[keep]
            if working.empty:
                return SearchResult(intent, working, 0, relaxed)

    working = _order(working, intent)

    # --- 3. truncate -----------------------------------------------------
    cap = limit or settings.max_results
    return SearchResult(
        intent, working.head(cap), total_before_ranking, relaxed
    )


def row_to_payload(row: Any) -> dict:
    """Serialize one result row for the API response."""
    distance = float(row.get("distance_km", tu.UNKNOWN_DISTANCE))
    return {
        "id": str(row.get("id", "")),
        "name": str(row.get("name", "")),
        "tags": str(row.get("tags", "")),
        "address": str(row.get("address", "")),
        "district": str(row.get("district", "")),
        "city": str(row.get("city", "")),
        "rating": round(float(row.get("rating", 0.0) or 0.0), 2),
        "rating_adjusted": round(float(row.get("rating_adjusted", 0.0) or 0.0), 2),
        "review_count": int(row.get("review_count", 0) or 0),
        "image_url": str(row.get("image_url", "") or ""),
        "source_url": str(row.get("source_url", "") or ""),
        "opening_hours": str(row.get("opening_hours", "") or ""),
        "price_text": str(row.get("price_text", "") or ""),
        "S_taste": round(float(row.get("relevance", 0.0) or 0.0), 4),
        "distance_km": round(distance, 2),
        "price": int(row.get("price", 0) or 0),
    }
