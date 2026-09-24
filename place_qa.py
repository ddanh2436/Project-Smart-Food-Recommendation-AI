# -*- coding: utf-8 -*-
"""Questions about one restaurant, and references back to earlier results.

Two things a conversation does that a search box does not:

* It asks about a place: "quán thứ hai có chỗ đậu xe không", "Phở Thìn mở
  cửa mấy giờ". The answer is in the record -- the aspect verdicts drawn from
  its reviews, its scores, its tags, its hours and price -- so it is read from
  there, and a question the record does not cover is answered as such rather
  than guessed.
* It points back: "quán đó", "cái đầu tiên", "giống quán thứ hai nhưng yên
  tĩnh hơn". The client sends the ids of the places each answer showed, and
  an ordinal or a demonstrative picks one of them.

Nothing here writes a sentence that the data does not back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import text_utils as tu
from data_store import store

ORDINALS: list[tuple[str, int]] = [
    ("đầu tiên", 0), ("thứ nhất", 0), ("số 1", 0), ("first", 0),
    ("thứ hai", 1), ("thứ 2", 1), ("số 2", 1), ("second", 1),
    ("thứ ba", 2), ("thứ 3", 2), ("số 3", 2), ("third", 2),
    ("thứ tư", 3), ("thứ 4", 3), ("số 4", 3), ("fourth", 3),
    ("thứ năm", 4), ("thứ 5", 4), ("số 5", 4), ("fifth", 4),
    ("cuối cùng", -1), ("cuối", -1), ("last", -1),
]
_PLACE_NOUN = r"(?:quán|cái|chỗ|nhà hàng|địa điểm|tiệm|place|one)"
DEMONSTRATIVES = [
    "quán đó", "quán này", "quán kia", "quán ấy", "quán vừa rồi", "quán trên",
    "chỗ đó", "chỗ này", "nhà hàng đó", "nhà hàng này", "tiệm đó",
    "that place", "this place", "that one",
]
SIMILAR_CUES = ["giống", "tương tự", "kiểu như", "như quán", "similar", "like the"]

# Question topics and the phrases that raise them. Diacritics kept, as in
# intent.ASPECT_CUES, because folding merges "chỗ" with "chờ".
TOPICS: dict[str, list[str]] = {
    "parking": ["đậu xe", "để xe", "gửi xe", "đỗ xe", "đỗ ô tô", "bãi xe", "parking"],
    "hygiene": ["sạch", "vệ sinh", "clean", "hygiene"],
    "service": ["phục vụ", "nhân viên", "thái độ", "service", "staff"],
    "space": ["không gian", "view", "ồn", "yên tĩnh", "chật", "rộng", "ambience", "quiet", "noisy"],
    "price": ["giá", "bao nhiêu tiền", "đắt", "rẻ", "price", "expensive", "cheap"],
    "food": ["ngon", "đồ ăn", "món ăn", "chất lượng", "tasty", "food"],
    "hours": ["mấy giờ", "giờ mở", "mở cửa", "đóng cửa", "còn mở", "open", "close", "hours"],
    "address": ["ở đâu", "địa chỉ", "chỗ nào", "where", "address"],
    "aircon": ["máy lạnh", "điều hòa", "air con", "aircon"],
    "good_for": ["hẹn hò", "gia đình", "trẻ em", "nhóm", "tiếp khách", "date", "family", "kids"],
}
QUESTION_CUES = ["không", "chưa", "thế nào", "ra sao", "bao nhiêu", "mấy", "ở đâu", "nào", "?",
                 "how", "what", "is it", "does it", "are they"]

ASPECT_TOPICS = {"parking", "hygiene", "service", "space", "price", "food"}
SCORE_COLUMNS = {
    "food": "score_quality", "price": "score_price",
    "service": "score_service", "space": "score_space",
}
GOOD_FOR_TAGS = ["Hẹn hò", "Gia đình", "Trẻ em", "Nhóm hội", "Tụ tập", "Tiếp khách",
                 "Nhậu", "Cơm văn phòng", "Lãng mạn"]
MIN_MENTIONS = 3


@dataclass
class Reference:
    row: dict
    index: int
    similar: bool = False
    # The message with the reference phrase removed, for a "similar" search.
    remainder: str = ""


def _has(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE))


def _row(restaurant_id: str) -> dict | None:
    frame = store.frame
    if frame.empty or not restaurant_id:
        return None
    match = frame[frame["id"] == restaurant_id]
    return None if match.empty else match.iloc[0].to_dict()


def _last_ids(history) -> list[str]:
    for turn in reversed(history or []):
        ids = getattr(turn, "ids", None) or []
        if turn.role == "bot" and ids:
            return list(ids)
    return []


def resolve(text: str, history) -> Reference | None:
    """The earlier result ``text`` points at, if any."""
    ids = _last_ids(history)
    if not ids:
        return None
    normalized = tu.normalize(text)
    index: int | None = None
    matched = ""
    for phrase, position in ORDINALS:
        # The noun is required: bare "thứ hai" is Monday and "cuối" starts
        # "cuối tuần".
        pattern = rf"(?<!\w){_PLACE_NOUN}\s+{re.escape(phrase)}(?!\w)"
        found = re.search(pattern, normalized, re.IGNORECASE)
        if found:
            index, matched = position, found.group(0)
            break
    if index is None:
        for phrase in DEMONSTRATIVES:
            if _has(normalized, phrase):
                index, matched = 0, phrase
                break
    if index is None:
        # A name from the last answer: "Bà Yum thì sao", matching the part of
        # the name before " - " (the rest is usually the address).
        folded = tu.fold(normalized)
        for position, restaurant_id in enumerate(ids):
            row = _row(restaurant_id)
            head = tu.fold(str((row or {}).get("name", "")).split(" - ")[0]).strip()
            if row and len(head) >= 4 and head in folded:
                index, matched = position, head
                break
        if index is None:
            return None
    try:
        restaurant_id = ids[index]
    except IndexError:
        return None
    row = _row(restaurant_id)
    if row is None:
        return None
    remainder = re.sub(re.escape(matched), " ", normalized, flags=re.IGNORECASE)
    for cue in SIMILAR_CUES:
        remainder = re.sub(rf"(?<!\w){re.escape(cue)}(?!\w)", " ", remainder, flags=re.IGNORECASE)
    return Reference(
        row=row,
        index=index if index >= 0 else len(ids) - 1,
        # "quán thứ hai nhưng yên tĩnh hơn" asks for places like it, changed,
        # not about it.
        similar=any(_has(normalized, cue) for cue in SIMILAR_CUES)
        or (_has(normalized, "hơn") and not is_question(normalized))
        or _has(normalized, "nhưng"),
        remainder=re.sub(r"\s+", " ", remainder).strip(),
    )


def topics_of(text: str) -> list[str]:
    normalized = tu.normalize(text)
    return [topic for topic, cues in TOPICS.items() if any(_has(normalized, c) for c in cues)]


def is_question(text: str) -> bool:
    normalized = tu.normalize(text)
    return "?" in normalized or any(_has(normalized, cue) for cue in QUESTION_CUES if cue != "?")


def strip_question(text: str) -> str:
    """``text`` without the question and topic wording, leaving a name."""
    out = tu.normalize(text or "")
    cues = [c for cues in TOPICS.values() for c in cues] + QUESTION_CUES
    for cue in sorted(cues, key=len, reverse=True):
        if cue != "?":
            out = re.sub(rf"(?<!\w){re.escape(cue)}(?!\w)", " ", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out.replace("?", " ")).strip()


def find_by_name(free_text: str) -> dict | None:
    """The best-rated restaurant whose name contains every word of ``free_text``."""
    words = tu.fold(strip_question(free_text)).split()
    frame = store.frame
    if len(words) < 2 or frame.empty:
        return None
    names = frame["name_norm"].map(tu.fold)
    mask = None
    for word in words:
        hit = names.str.contains(word, regex=False, na=False)
        mask = hit if mask is None else (mask & hit)
    matches = frame[mask]
    if matches.empty:
        return None
    column = "rating_adjusted" if "rating_adjusted" in matches.columns else "rating"
    return matches.sort_values(column, ascending=False).iloc[0].to_dict()


def similar_query(row: dict) -> str:
    """A search for places like ``row``: its main dish, in its district."""
    from search import _DISH_TAGS_FOLDED  # the dish list the ranker reads

    folded = tu.fold(str(row.get("tags", "")))
    dish = next((original for needle, original in _DISH_TAGS_FOLDED if needle and needle in folded), "")
    district = str(row.get("district", "") or "")
    return " ".join(part for part in (dish, district) if part)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
LABELS = {
    "vi": {
        "food": "món ăn", "price": "giá cả", "service": "phục vụ", "space": "không gian",
        "hygiene": "vệ sinh", "parking": "chỗ để xe",
    },
    "en": {
        "food": "the food", "price": "the prices", "service": "the service",
        "space": "the space", "hygiene": "cleanliness", "parking": "parking",
    },
}


def _aspect_line(row: dict, topic: str, lang: str) -> str:
    label = LABELS[lang][topic]
    mentions = int(float(row.get(f"aspect_{topic}_n", 0) or 0))
    ratio = float(row.get(f"aspect_{topic}", 0.0) or 0.0)
    score = float(row.get(SCORE_COLUMNS.get(topic, ""), 0.0) or 0.0) if topic in SCORE_COLUMNS else 0.0
    score_text = (
        (f" Điểm {label}: {score:.1f}/10." if lang == "vi" else f" Score: {score:.1f}/10.")
        if score > 0 else ""
    )
    if mentions < MIN_MENTIONS:
        if lang == "vi":
            return f"Chưa đủ đánh giá nhắc tới {label} để nói chắc.{score_text}"
        return f"Not enough reviews mention {label} to say.{score_text}"
    percent = round(ratio * 100)
    if lang == "vi":
        verdict = "được khen" if ratio >= 0.7 else "hay bị chê" if ratio <= 0.3 else "ý kiến trái chiều"
        return f"Về {label}: {verdict} — {percent}% trong {mentions} lượt nhắc là tích cực.{score_text}"
    verdict = "praised" if ratio >= 0.7 else "often criticised" if ratio <= 0.3 else "mixed"
    return f"{label.capitalize()}: {verdict} — {percent}% of {mentions} mentions positive.{score_text}"


def answer(row: dict, topics: list[str], lang: str = "vi") -> str:
    """A reply about ``row`` covering ``topics``, from the record alone."""
    lang = "en" if lang == "en" else "vi"
    name = str(row.get("name", ""))
    tags = str(row.get("tags", ""))
    lines: list[str] = []
    for topic in topics or ["food", "hours"]:
        if topic in ASPECT_TOPICS:
            line = _aspect_line(row, topic, lang)
            if topic == "price" and row.get("price_text"):
                line += (f" Giá khoảng {row['price_text']}." if lang == "vi"
                         else f" Around {row['price_text']}.")
            lines.append(line)
        elif topic == "hours":
            hours = str(row.get("opening_hours", "") or "")
            lines.append(
                (f"Giờ mở cửa: {hours}." if hours else "Quán chưa công bố giờ mở cửa.")
                if lang == "vi" else (f"Open: {hours}." if hours else "Opening hours are not listed.")
            )
        elif topic == "address":
            lines.append((f"Địa chỉ: {row.get('address', '')}." if lang == "vi"
                          else f"Address: {row.get('address', '')}."))
        elif topic == "aircon":
            has_ac = "máy lạnh" in tu.normalize(tags)
            lines.append(
                ("Theo Foody, quán có máy lạnh." if has_ac else "Foody không ghi quán có máy lạnh.")
                if lang == "vi" else
                ("Foody lists air conditioning." if has_ac else "Foody does not list air conditioning.")
            )
        elif topic == "good_for":
            fits = [t for t in GOOD_FOR_TAGS if tu.normalize(t) in tu.normalize(tags)]
            if lang == "vi":
                lines.append(f"Theo Foody, quán hợp cho: {', '.join(fits)}." if fits
                             else "Foody chưa ghi quán hợp cho dịp nào.")
            else:
                lines.append(f"Foody lists it as good for: {', '.join(fits)}." if fits
                             else "Foody does not list what occasions it suits.")
    head = f"**{name}** — " if lines else f"**{name}**"
    return head + " ".join(lines)
