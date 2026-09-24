# -*- coding: utf-8 -*-
"""Conversational layer over the recommendation engine.

Free and deterministic: replies are composed from templates filled with facts
taken from the retrieved rows, so the bot can describe *why* it picked a place
without a language model and without ever inventing a detail.

What it adds over the original one-shot bot:

* **Multi-turn context.** "phở ở quận 1" then "rẻ hơn đi" keeps the dish and
  the district and only changes the ranking.
* **Follow-up understanding.** Cheaper / closer / better / show-me-more are
  recognised as refinements rather than new searches.
* **Clarifying questions.** A bare "tôi đói" gets asked what they feel like
  instead of a random result list.
* **Grounded descriptions.** The reply names the top result with its real
  rating, price band and distance.
* **Small talk.** Greetings and thanks no longer run a restaurant search.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import text_utils as tu
from intent import Intent, parse_intent
from search import SearchResult, row_to_payload, search

# --------------------------------------------------------------------------
# Conversational cues
# --------------------------------------------------------------------------
GREETING_CUES = [
    "xin chào", "chào bạn", "chào", "hello", "hi", "hey", "alo", "helo",
]
THANKS_CUES = [
    "cảm ơn", "cám ơn", "thanks", "thank you", "thank", "tks", "ok cảm ơn",
]
HELP_CUES = [
    "giúp gì", "giúp được gì", "giúp được những gì", "làm được gì",
    "làm được những gì", "làm gì được", "có thể làm gì", "biết làm gì",
    "bạn là ai", "bạn tên gì", "hướng dẫn", "cách dùng", "chức năng",
    "what can you do", "what do you do", "who are you", "how to use", "help",
]
HUNGRY_CUES = [
    "đói", "ăn gì", "gợi ý", "tùy bạn", "gì cũng được", "không biết",
    "recommend", "suggest", "anything", "hungry",
]

# Refinements: each maps to the ranking it implies.
REFINEMENTS: dict[str, list[str]] = {
    "price": ["rẻ hơn", "rẻ hơn đi", "bớt đắt", "giá thấp hơn", "cheaper"],
    "distance": ["gần hơn", "gần hơn đi", "đỡ xa", "closer"],
    "rating": [
        "ngon hơn", "tốt hơn", "chất lượng hơn", "đánh giá cao hơn", "better",
    ],
}
MORE_CUES = [
    "còn gì khác", "còn quán nào", "khác đi", "thêm đi", "xem thêm",
    "cái khác", "quán khác", "more", "others", "something else",
]

# ---------------------------------------------------------------------------
# Slot filling
#
# "quán ở Quận 1" is answerable, but it is answerable with four hundred places.
# The useful reply is the list plus the one question that would shorten it, and
# the cheapest way to ask is a row of chips the user can tap.
#
# A chip carries a full query rather than a bare word, composed from the slots
# already known, so tapping "Phở" after "quán ở Quận 1" sends "Phở ở Quận 1"
# and not "Phở" on its own. The value stays Vietnamese in both languages -- it
# is what the index is built from -- while the label is translated.
# ---------------------------------------------------------------------------

DISH_CHIPS = [
    ("Phở", "Pho"),
    ("Bún bò", "Bun bo"),
    ("Cơm tấm", "Com tam"),
    ("Bánh mì", "Banh mi"),
    ("Lẩu", "Hot pot"),
    ("Cà phê", "Coffee"),
]

PRICE_CHIPS = [
    ("dưới 50k", "under 50k"),
    ("dưới 100k", "under 100k"),
    ("dưới 200k", "under 200k"),
]

NEARBY_CHIP = ("gần đây", "near me")

# Below this many candidates the list is already short enough to read, so the
# chips would be noise rather than help.
CHIP_THRESHOLD = 25


def _missing_slots(intent: Intent) -> list[str]:
    """Which of the three slots that most shorten a result list are unset."""
    missing = []
    if not intent.dishes:
        missing.append("dish")
    if not (intent.districts or intent.cities or intent.wants_nearby):
        missing.append("area")
    if intent.price_min is None and intent.price_max is None:
        missing.append("price")
    return missing


def _compose(intent: Intent, dish: str = "", price: str = "", area: str = "") -> str:
    """A natural query from the known slots plus one new term.

    Written out rather than appending the term to the raw message, so the chip
    the user taps reads as a sentence they might have typed: "Phở ở Quận 1",
    not "quán ở Quận 1 Phở".
    """
    subject = dish or (intent.dishes[0] if intent.dishes else "quán ăn")
    parts = [subject]

    where = area or (
        intent.districts[0]
        if intent.districts
        else intent.cities[0] if intent.cities else ""
    )
    if where:
        parts.append(where if where == NEARBY_CHIP[0] else f"ở {where}")
    elif intent.wants_nearby:
        parts.append(NEARBY_CHIP[0])

    if price:
        parts.append(price)
    elif intent.price_max:
        parts.append(f"dưới {int(intent.price_max // 1000)}k")

    return " ".join(parts)


# Six fits one scrollable row without the user having to drag it.
MAX_CHIPS = 6


def _slot_chips(intent: Intent, lang: str, missing: list[str]) -> list[dict]:
    """Quick replies for the most useful missing slots."""
    pick = 0 if lang == "vi" else 1
    chips: list[dict] = []

    # "gần đây" narrows harder than any dish does, so its place is reserved
    # before the dishes are laid out rather than left to whatever fits last.
    wants_nearby_chip = "area" in missing
    room = MAX_CHIPS - (1 if wants_nearby_chip else 0)

    if "dish" in missing:
        chips += [
            {"label": pair[pick], "query": _compose(intent, dish=pair[0])}
            for pair in DISH_CHIPS[:room]
        ]
    elif "price" in missing:
        chips += [
            {"label": pair[pick], "query": _compose(intent, price=pair[0])}
            for pair in PRICE_CHIPS[:room]
        ]

    if wants_nearby_chip:
        chips.append(
            {"label": NEARBY_CHIP[pick], "query": _compose(intent, area=NEARBY_CHIP[0])}
        )

    return chips[:MAX_CHIPS]


# Vague queries where a clarifying question beats a random list.
CLARIFY_QUESTIONS = {
    "vi": [
        "Bạn đang muốn ăn món nước (phở, bún, hủ tiếu) hay món khô (cơm, bánh mì) nhỉ? 🍜",
        "Cho mình biết bạn thích món gì, hoặc đang ở khu nào để mình tìm cho sát nhé! 📍",
        "Bạn muốn ăn mặn, ăn chay, hay uống cà phê / trà sữa? ☕",
    ],
    "en": [
        "Are you after a noodle soup (pho, bun) or something dry (rice, banh mi)? 🍜",
        "Tell me a dish you like, or which area you're in, and I'll narrow it down! 📍",
        "Savoury food, vegetarian, or a coffee / bubble tea? ☕",
    ],
}

# When nothing was found *and* the query was not understood, the honest
# answer is "I did not follow", naming the words, not "there is no such
# place" -- which blames the data for what was a misreading.
UNCLEAR_REPLIES = {
    "vi": "Mình chưa hiểu \"{words}\" lắm 🤔. Bạn nói thử tên món, khu vực hoặc mức giá nhé — hoặc chọn nhanh bên dưới:",
    "en": "I didn't quite follow \"{words}\" 🤔. Try a dish, an area or a budget — or pick one below:",
}

# At or below this parser confidence an empty result is a misreading.
UNCLEAR_CONFIDENCE = 0.5

GREETING_REPLIES = {
    "vi": [
        "Chào bạn! 👋 Hôm nay bạn muốn ăn gì? Nhắn tên món, khu vực, hoặc gửi ảnh món ăn nhé!",
        "VietNomNom xin chào! 🍜 Bạn thử hỏi mình kiểu \"bún bò ngon ở quận 1 dưới 100k\" xem.",
        "Hello! 🥘 Bạn đang đói gì nào? Mình tìm quán theo món, giá, khoảng cách đều được.",
    ],
    "en": [
        "Hi there! 👋 What are you craving? Send a dish, an area, or even a photo.",
        "VietNomNom here! 🍜 Try something like \"cheap banh mi near me\".",
        "Hello! 🥘 I can search by dish, price or distance — what are you after?",
    ],
}

THANKS_REPLIES = {
    "vi": [
        "Không có gì! Chúc bạn ăn ngon nha 😋",
        "Rất vui được giúp bạn! Cần tìm quán nào nữa cứ nhắn mình nhé 🍽️",
    ],
    "en": [
        "You're welcome — enjoy your meal! 😋",
        "Happy to help! Ping me any time you're hungry 🍽️",
    ],
}

HELP_REPLIES = {
    "vi": (
        "Mình là trợ lý ẩm thực VietNomNom 🤖 Mình có thể:\n"
        "• Tìm quán theo món: \"bún bò huế ngon\"\n"
        "• Theo khu vực: \"cơm tấm ở Bình Thạnh\"\n"
        "• Theo giá: \"lẩu dưới 200k\"\n"
        "• Theo khoảng cách: \"cà phê gần đây\"\n"
        "• Loại trừ: \"hải sản nhưng không cay\"\n"
        "• Nhận diện món qua ảnh — bấm nút hình ảnh nhé!"
    ),
    "en": (
        "I'm the VietNomNom food assistant 🤖 I can:\n"
        "• find places by dish: \"bun bo hue\"\n"
        "• by area: \"com tam in Binh Thanh\"\n"
        "• by price: \"hotpot under 200k\"\n"
        "• by distance: \"coffee near me\"\n"
        "• with exclusions: \"seafood but not spicy\"\n"
        "• identify a dish from a photo — tap the image button!"
    ),
}

# Shown when the user asks for somewhere nearby but sent no coordinates.
NEED_LOCATION_REPLIES = {
    "vi": (
        "Mình chưa biết bạn đang ở đâu nên chưa tính được quán nào gần "
        "nhất 📍 Bạn bật quyền vị trí giúp mình nhé, hoặc nói rõ khu vực "
        "— ví dụ “phở ở Quận 1”. Trong lúc đó, đây là những quán được "
        "đánh giá cao nhất:"
    ),
    "en": (
        "I do not know where you are, so I cannot work out what is "
        "closest 📍 Allow location access, or name an area — for example "
        "“pho in District 1”. In the meantime, here are the highest-rated "
        "ones:"
    ),
}

NOT_FOUND_REPLIES = {
    "vi": [
        "Hic, mình chưa tìm thấy quán nào khớp với \"{query}\". Bạn thử từ khóa ngắn hơn xem sao? 🍜",
        "Chưa có kết quả cho \"{query}\" trong dữ liệu của mình. Thử \"phở\", \"cơm tấm\" hoặc \"bánh mì\" nhé!",
        "Ca này khó 😅 Mình không thấy quán nào cho \"{query}\". Kiểm tra lại chính tả giúp mình nha.",
    ],
    "en": [
        "I couldn't find anything matching \"{query}\". Try a shorter keyword? 🍜",
        "No results for \"{query}\" yet. Try \"pho\", \"com tam\" or \"banh mi\"!",
        "Tough one 😅 nothing found for \"{query}\" — maybe check the spelling?",
    ],
}

SORT_PHRASES = {
    "vi": {
        "price": "sắp theo giá từ thấp đến cao",
        "distance": "sắp theo khoảng cách gần nhất",
        "rating": "sắp theo điểm đánh giá cao nhất",
        "relevance": "sắp theo độ phù hợp",
    },
    "en": {
        "price": "sorted cheapest first",
        "distance": "sorted nearest first",
        "rating": "sorted by highest rating",
        "relevance": "sorted by best match",
    },
}

RELAXED_NOTES = {
    "vi": {
        "price_max": "mình đã nới mức giá một chút vì không có quán nào đúng ngưỡng đó",
        "price_min": "mình đã nới mức giá một chút",
        "min_rating": "mình đã hạ yêu cầu điểm đánh giá để có kết quả",
        "location": "mình đã mở rộng ra ngoài khu vực bạn nói",
        "radius": "mình đã tìm xa hơn một chút",
        "dish": "mình chưa có đúng món đó nên gợi ý các quán gần nghĩa",
        "dish_broadened": "mình chưa có đúng món đó nên mở rộng sang nhóm món tương tự",
        "time": "mình đã bỏ qua điều kiện giờ mở cửa",
        "exclude": "mình chưa lọc được hết yêu cầu loại trừ",
        "open_now": "mình đã bỏ qua điều kiện đang mở cửa",
    },
    "en": {
        "price_max": "I loosened the price cap since nothing matched exactly",
        "price_min": "I loosened the price range",
        "min_rating": "I lowered the rating requirement to find results",
        "location": "I looked beyond the area you mentioned",
        "radius": "I searched a bit further out",
        "dish": "I don't have that exact dish, so these are the closest matches",
        "dish_broadened": "I widened the search to the closest dish category",
        "time": "I ignored the opening-hours filter",
        "exclude": "I couldn't fully apply your exclusions",
        "open_now": "I ignored the open-now filter",
    },
}


@dataclass
class ChatTurn:
    role: str  # "user" | "bot"
    text: str


def _matches_any(text: str, cues: list[str]) -> bool:
    for cue in sorted(cues, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text, re.IGNORECASE):
            return True
    return False


def _detect_refinement(text: str) -> str | None:
    for sort_by, cues in REFINEMENTS.items():
        if _matches_any(text, cues):
            return sort_by
    return None


def _last_substantive_query(history: list[ChatTurn]) -> str:
    """The most recent user turn that actually named something to search for.

    Taking simply the last user turn was wrong for a chain of follow-ups:
    after "phở ở Hà Nội" -> "rẻ hơn đi" -> "còn gì khác", the last turn is
    itself a refinement, so the bot ended up searching for the literal word
    "đi". Walking back to the last turn with real constraints keeps the
    original subject alive across any number of refinements.
    """
    for turn in reversed(history or []):
        if turn.role != "user" or not turn.text.strip():
            continue
        text = tu.normalize(turn.text)
        if _detect_refinement(text) or _matches_any(text, MORE_CUES):
            continue  # a refinement, not a subject
        if parse_intent(turn.text).has_constraints:
            return turn.text
    return ""


def _merge_intents(previous: Intent, current: Intent) -> Intent:
    """Carry unstated slots forward from the previous turn.

    A follow-up like "rẻ hơn" only supplies a ranking, so the dish, area and
    constraints from the previous turn have to survive or the bot starts over.
    """
    merged = Intent(
        raw_query=current.raw_query,
        normalized_query=current.normalized_query or previous.normalized_query,
        dishes=current.dishes or previous.dishes,
        adjectives=current.adjectives or previous.adjectives,
        time_tags=current.time_tags or previous.time_tags,
        exclude=list({*previous.exclude, *current.exclude}),
        districts=current.districts or previous.districts,
        cities=current.cities or previous.cities,
        price_min=current.price_min if current.price_min is not None else previous.price_min,
        price_max=current.price_max if current.price_max is not None else previous.price_max,
        min_rating=current.min_rating if current.min_rating is not None else previous.min_rating,
        party_size=current.party_size or previous.party_size,
        open_now=current.open_now or previous.open_now,
        sort_by=current.sort_by if current.sort_by != "relevance" else previous.sort_by,
        free_text=current.free_text or previous.free_text,
        # Carried forward like the other slots. Without these a follow-up such
        # as "rẻ hơn đi" dropped the aspect preference the user had already
        # stated, so "quán sạch sẽ ở Quận 1" silently stopped caring about
        # hygiene one turn later.
        aspects=current.aspects or previous.aspects,
        aspect_avoid=list({*previous.aspect_avoid, *current.aspect_avoid}),
    )
    merged.name_like = len(merged.free_text.split()) >= 2
    return merged


def _describe_top(row: dict, lang: str) -> str:
    """One grounded sentence about the best hit, using only real fields."""
    name = row.get("name") or ""
    rating = row.get("rating") or 0
    distance = row.get("distance_km")
    price = row.get("price_text") or ""
    district = row.get("district") or ""

    bits: list[str] = []
    if rating:
        bits.append(
            f"{rating:.1f}/10 điểm" if lang == "vi" else f"rated {rating:.1f}/10"
        )
    if district:
        bits.append(f"ở {district}" if lang == "vi" else f"in {district}")
    if distance is not None and 0 < distance < 100:
        bits.append(
            f"cách bạn ~{distance:.1f}km" if lang == "vi"
            else f"~{distance:.1f}km away"
        )
    if price and price.lower() not in {"đang cập nhật", "n/a"}:
        bits.append(f"giá {price}" if lang == "vi" else f"around {price}")

    if not bits:
        return ""
    detail = ", ".join(bits)
    if lang == "vi":
        return f'Nổi bật nhất là **{name}** — {detail}.'
    return f'The standout is **{name}** — {detail}.'


def _success_reply(result: SearchResult, payloads: list[dict], lang: str) -> str:
    # Report how many places actually matched, not how many fit on this page.
    # Saying "found 3" when 87 matched and only 3 were shown is simply wrong.
    total = result.total_before_ranking or len(payloads)
    shown = len(payloads)
    intent = result.intent
    subject = (
        ", ".join(intent.dishes) if intent.dishes
        else (intent.free_text or (intent.raw_query or "").strip())
    )
    where = intent.districts[0] if intent.districts else (
        intent.cities[0] if intent.cities else ""
    )

    if lang == "vi":
        head = f"Mình tìm được {total} quán"
        if subject:
            head += f' cho "{subject}"'
        if where:
            head += f" ở {where}"
        if total > shown:
            head += f", đây là {shown} quán phù hợp nhất"
        head += f", {SORT_PHRASES['vi'][intent.sort_by]}."
    else:
        head = f"I found {total} place{'s' if total != 1 else ''}"
        if subject:
            head += f' for "{subject}"'
        if where:
            head += f" in {where}"
        if total > shown:
            head += f", here are the top {shown}"
        head += f", {SORT_PHRASES['en'][intent.sort_by]}."

    parts = [head]

    highlight = _describe_top(payloads[0], lang) if payloads else ""
    if highlight:
        parts.append(highlight)

    notes = [
        RELAXED_NOTES[lang][key]
        for key in result.relaxed_filters
        if key in RELAXED_NOTES[lang]
    ]
    if notes:
        prefix = "Lưu ý: " if lang == "vi" else "Note: "
        parts.append(prefix + "; ".join(notes[:2]) + ".")

    if intent.exclude:
        excluded = ", ".join(intent.exclude)
        parts.append(
            f"Đã loại các quán có {excluded}." if lang == "vi"
            else f"Excluded places with {excluded}."
        )

    return " ".join(parts)


def respond(
    message: str,
    history: list[ChatTurn] | None = None,
    user_gps: list[float] | None = None,
    lang: str = "vi",
    limit: int = 5,
) -> dict:
    """Produce a reply plus the restaurant cards that back it up."""
    lang = "vi" if lang not in {"vi", "en"} else lang
    history = history or []
    text = tu.normalize(message)

    # --- small talk, handled before any search ---------------------------
    if not text:
        return {
            "reply": random.choice(GREETING_REPLIES[lang]),
            "results": [],
            "intent": None,
            "kind": "greeting",
        }
    if _matches_any(text, THANKS_CUES) and len(text.split()) <= 4:
        return {
            "reply": random.choice(THANKS_REPLIES[lang]),
            "results": [],
            "intent": None,
            "kind": "thanks",
        }
    if _matches_any(text, HELP_CUES):
        return {
            "reply": HELP_REPLIES[lang],
            "results": [],
            "intent": None,
            "kind": "help",
        }
    if _matches_any(text, GREETING_CUES) and len(text.split()) <= 3:
        return {
            "reply": random.choice(GREETING_REPLIES[lang]),
            "results": [],
            "intent": None,
            "kind": "greeting",
        }

    has_gps = bool(user_gps and len(user_gps) == 2)
    intent = parse_intent(message, has_gps=has_gps)

    # --- follow-up refinement --------------------------------------------
    refinement = _detect_refinement(text)
    wants_more = _matches_any(text, MORE_CUES)
    previous_query = _last_substantive_query(history)

    if (refinement or wants_more) and previous_query:
        # A refinement carries no subject of its own, so search the previous
        # subject again and only change how the results are ordered. Using the
        # refinement's own words as the query made the bot search for "đi".
        previous_intent = parse_intent(previous_query, has_gps=has_gps)
        intent = _merge_intents(previous_intent, intent)
        if refinement:
            intent.sort_by = refinement
        effective_query = previous_query
    else:
        effective_query = message

    # --- too vague to search ---------------------------------------------
    if not intent.has_constraints and _matches_any(text, HUNGRY_CUES):
        missing = _missing_slots(intent)
        return {
            "reply": random.choice(CLARIFY_QUESTIONS[lang]),
            "results": [],
            "intent": intent.to_dict(),
            "kind": "clarify",
            "slots_missing": missing,
            "chips": _slot_chips(intent, lang, missing),
        }

    # --- search -----------------------------------------------------------
    # "còn gì khác" should show the *next* places, not repeat the same ones, so
    # page forward by however many times the user has already asked for more.
    page = _more_count(history) if wants_more else 0
    offset = page * limit
    result = search(
        effective_query, user_gps=user_gps, limit=max(limit + offset, 10)
    )

    # A "near me" question with no coordinates cannot be ordered by distance;
    # ordering by quality is the most useful honest substitute.
    if result.intent.wants_nearby and not has_gps and not result.rows.empty:
        from search import _order  # internal helper, deliberately reused

        result.intent.sort_by = "rating"
        result = SearchResult(
            result.intent,
            _order(result.rows, result.intent),
            result.total_before_ranking,
            result.relaxed_filters,
        )

    if refinement:
        # Re-rank the already-filtered set by what the follow-up asked for.
        result.intent.sort_by = refinement
        from search import _order  # internal helper, deliberately reused

        result = SearchResult(
            result.intent,
            _order(result.rows, result.intent),
            result.total_before_ranking,
            result.relaxed_filters,
        )

    rows = result.rows.iloc[offset : offset + limit]
    if rows.empty and offset:
        # Ran off the end of the list: wrap to the start and say so.
        rows = result.rows.head(limit)
        exhausted = True
    else:
        exhausted = False

    payloads = [row_to_payload(row, result.intent) for _, row in rows.iterrows()]

    if not payloads and result.intent.confidence <= UNCLEAR_CONFIDENCE:
        missing = _missing_slots(result.intent)
        words = result.intent.free_text or (message or "").strip()
        return {
            "reply": UNCLEAR_REPLIES[lang].format(words=words[:40]),
            "results": [],
            "intent": result.intent.to_dict(),
            "kind": "clarify",
            "slots_missing": missing,
            "chips": _slot_chips(result.intent, lang, missing),
        }

    if not payloads:
        template = random.choice(NOT_FOUND_REPLIES[lang])
        return {
            "reply": template.format(query=(message or "").strip()[:60]),
            "results": [],
            "intent": result.intent.to_dict(),
            "kind": "not_found",
        }

    # The user asked for somewhere nearby but we have no coordinates.
    #
    # The proximity request used to be dropped silently and the answer came
    # back ranked by relevance, so "quán phở gần đây" listed places in Hà Nội
    # and Đà Lạt as though they were nearby. Say what is missing instead, and
    # fall back to quality rather than pretending the order means proximity.
    if result.intent.wants_nearby and not has_gps:
        return {
            "reply": NEED_LOCATION_REPLIES[lang],
            "results": payloads,
            "intent": result.intent.to_dict(),
            "kind": "need_location",
            "total_matches": result.total_before_ranking,
            "relaxed_filters": result.relaxed_filters,
        }

    reply = _success_reply(result, payloads, lang)
    if exhausted:
        reply += (
            " (Mình đã hết gợi ý mới nên quay lại từ đầu nhé!)"
            if lang == "vi"
            else " (That's everything I have, so I've looped back to the top.)"
        )

    # A broad query still gets its results; the chips sit under them so the
    # user can narrow without being made to answer a question first.
    missing = _missing_slots(result.intent)
    chips = (
        _slot_chips(result.intent, lang, missing)
        if missing and result.total_before_ranking > CHIP_THRESHOLD
        else []
    )

    return {
        "reply": reply,
        "results": payloads,
        "intent": result.intent.to_dict(),
        "kind": "results",
        "total_matches": result.total_before_ranking,
        "relaxed_filters": result.relaxed_filters,
        "slots_missing": missing,
        "chips": chips,
    }


def _more_count(history: list[ChatTurn]) -> int:
    """How many times the user has already asked for more, since the last
    substantive query. Used to page forward through the result list."""
    count = 0
    for turn in reversed(history or []):
        if turn.role != "user" or not turn.text.strip():
            continue
        text = tu.normalize(turn.text)
        if _matches_any(text, MORE_CUES):
            count += 1
            continue
        if _detect_refinement(text):
            continue
        break  # reached the subject turn
    return count + 1
