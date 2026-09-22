# -*- coding: utf-8 -*-
"""Food photo recognition with the fine-tuned YOLO model.

Changes from the original inline implementation in ``api.py``:

* weights are loaded from an absolute path, so the service no longer depends
  on the working directory it happens to be started from;
* the model loads lazily and records its error instead of failing silently;
* all detections are returned (ranked), not just the single best box, so the
  caller can fall back to the second guess;
* ``.conf`` tensors are converted to floats before sorting rather than being
  compared as tensors;
* the class-name to Vietnamese-dish mapping is exact-match first and only
  then fuzzy, so "Bun" can no longer be resolved by an unrelated substring.
"""

from __future__ import annotations

import io
import logging
import threading

import knowledge as kb
import text_utils as tu
from config import settings

logger = logging.getLogger(__name__)

# The trained classes, from data.yaml:
#   ['Banh-Mi', 'Bot Chien', 'Bun', 'Goi-Cuon', 'Pho']
CLASS_TO_DISH: dict[str, str] = {
    "banh-mi": "bánh mì",
    "banh mi": "bánh mì",
    "banhmi": "bánh mì",
    "bot chien": "bột chiên",
    "bot-chien": "bột chiên",
    "bun": "bún",
    "goi-cuon": "gỏi cuốn",
    "goi cuon": "gỏi cuốn",
    "pho": "phở",
}

# Confidence tiers.
#
# The model knows five dishes. Anything a user photographs outside that set --
# com tam, hu tieu, lau, banh xeo, che -- lands somewhere between a weak guess
# and nothing at all, and the old behaviour for both was the same dead end: an
# empty response and "khong nhan dien duoc mon an". That is the worst possible
# answer, because the user cannot tell whether the photo was bad, the dish is
# unsupported, or the service is broken.
#
# So the reply is graded instead:
#
#   >= 0.55   confident  name the dish and search for it
#   >= 0.25   uncertain  name it, say it is a guess, offer alternatives
#   >= 0.10   group      do not name a dish; say which kind of food it looks
#                        like and offer the dishes in that group
#   below     none       offer the popular dishes and ask for a typed name
#
# The 0.25 floor is unchanged -- it is still what decides whether a dish gets
# named -- and the 0.10 tier only reads detections that were already being
# computed and thrown away.
CONFIDENT_CONFIDENCE = 0.55
MIN_CONFIDENCE = 0.25
WEAK_CONFIDENCE = 0.10

# The five trained classes split cleanly into two kinds of meal, which is a
# useful thing to say even when the dish itself is not certain.
DISH_GROUPS: dict[str, str] = {
    "phở": "soup",
    "bún": "soup",
    "bánh mì": "dry",
    "bột chiên": "dry",
    "gỏi cuốn": "dry",
}

# Offered when the model cannot name anything. Vietnamese values, because they
# are what the search index is built from; the client translates the labels.
POPULAR_DISHES = ["Phở", "Bún bò", "Cơm tấm", "Bánh mì", "Lẩu", "Cà phê"]

GROUP_DISHES = {
    "soup": ["Phở", "Bún bò", "Hủ tiếu", "Bún riêu"],
    "dry": ["Bánh mì", "Cơm tấm", "Gỏi cuốn", "Bột chiên"],
}


class FoodDetector:
    """Lazy, thread-safe wrapper around the YOLO model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._error: str | None = None
        self._attempted = False

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        if not settings.enable_yolo:
            self._error = "disabled by ENABLE_YOLO"
            return
        with self._lock:
            if self._model is not None or self._attempted:
                return
            self._attempted = True
            weights = settings.yolo_weights
            if not weights.exists():
                self._error = (
                    f"YOLO weights not found at {weights}. Commit best.pt or "
                    f"set YOLO_MODEL_PATH."
                )
                logger.error(self._error)
                return
            try:
                from ultralytics import YOLO

                logger.info("Loading YOLO weights from %s", weights)
                self._model = YOLO(str(weights))
                self._error = None
                logger.info("YOLO model ready")
            except Exception as exc:  # noqa: BLE001 - reported via /health
                self._error = str(exc)
                self._model = None
                logger.exception("YOLO failed to load: %s", exc)

    def detect(
        self,
        image_bytes: bytes,
        top_k: int = 3,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> list[dict]:
        """Return detections ranked by confidence, best first.

        `min_confidence` is a parameter so the caller can look below the floor
        that decides whether a dish gets named -- a 0.18 detection is not worth
        asserting, but it is worth saying "this looks like a noodle soup".
        """
        if self._model is None and not self._attempted:
            self.load()
        if self._model is None:
            return []

        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            # YOLO wants RGB; phone photos are often RGBA or CMYK.
            if image.mode != "RGB":
                image = image.convert("RGB")
            results = self._model(image, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO inference failed: %s", exc)
            return []

        detections: list[dict] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            for box in boxes:
                try:
                    # Convert tensors to plain Python before comparing/sorting.
                    confidence = float(box.conf.item() if hasattr(box.conf, "item")
                                       else box.conf)
                    class_id = int(box.cls.item() if hasattr(box.cls, "item")
                                   else box.cls)
                except (AttributeError, TypeError, ValueError):
                    continue
                raw_name = str(names.get(class_id, class_id))
                detections.append({
                    "class_name": raw_name,
                    "confidence": round(confidence, 4),
                    "dish": resolve_dish(raw_name),
                })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        # Keep the best detection per dish so three boxes of the same bowl of
        # pho do not fill the whole list.
        unique: list[dict] = []
        seen: set[str] = set()
        for detection in detections:
            if detection["confidence"] < min_confidence:
                continue
            if detection["dish"] in seen:
                continue
            seen.add(detection["dish"])
            unique.append(detection)
            if len(unique) >= top_k:
                break
        return unique


def classify(detections: list[dict]) -> dict:
    """Grade a detection list into one of the four tiers.

    Returns the tier, the dish when there is one, the kind of food it looks
    like, and the dishes worth offering next. The wording is left to the
    client, which knows the language the interface is in.
    """
    best = detections[0] if detections else None

    if best and best["confidence"] >= CONFIDENT_CONFIDENCE:
        group = DISH_GROUPS.get(best["dish"])
        return {
            "tier": "confident",
            "dish": best["dish"],
            "group": group,
            "suggestions": [],
        }

    if best and best["confidence"] >= MIN_CONFIDENCE:
        group = DISH_GROUPS.get(best["dish"])
        # Alternatives first, then the rest of its group: if the guess is
        # wrong the neighbouring dish is the likeliest correction.
        others = [title_case(d["dish"]) for d in detections[1:]]
        pool = others + GROUP_DISHES.get(group or "", [])
        return {
            "tier": "uncertain",
            "dish": best["dish"],
            "group": group,
            "suggestions": _dedupe(pool, exclude=title_case(best["dish"]))[:4],
        }

    if best and best["confidence"] >= WEAK_CONFIDENCE:
        group = DISH_GROUPS.get(best["dish"])
        return {
            "tier": "group",
            "dish": None,
            "group": group,
            "suggestions": _dedupe(GROUP_DISHES.get(group or "", POPULAR_DISHES))[:4],
        }

    return {
        "tier": "none",
        "dish": None,
        "group": None,
        "suggestions": list(POPULAR_DISHES),
    }


def _dedupe(items: list[str], exclude: str = "") -> list[str]:
    seen, out = set(), []
    for item in items:
        key = tu.fold(item)
        if key in seen or (exclude and key == tu.fold(exclude)):
            continue
        seen.add(key)
        out.append(item)
    return out


def resolve_dish(class_name: str) -> str:
    """Map a YOLO class name to a Vietnamese dish name.

    Exact match first. The original fell straight into a substring loop, where
    the first dictionary entry whose key merely *overlapped* the class name
    won — so an unrelated mapping could claim the detection.
    """
    cleaned = tu.normalize(str(class_name).replace("_", " ").replace("-", " "))
    if not cleaned:
        return ""

    # 1. exact, against the trained class list
    for candidate in (cleaned, cleaned.replace(" ", "-"), cleaned.replace(" ", "")):
        if candidate in CLASS_TO_DISH:
            return CLASS_TO_DISH[candidate]

    # 2. exact, against the general English-Vietnamese dictionary
    if cleaned in kb.EN_VI_MAPPING:
        return kb.EN_VI_MAPPING[cleaned]

    # 3. whole-word containment, longest key first so "bun bo hue" beats "bun"
    for key in kb.EN_VI_SORTED:
        if len(key) < 3:
            continue
        import re

        if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", cleaned):
            return kb.EN_VI_MAPPING[key]

    # 4. give up and return the cleaned class name
    return cleaned


def title_case(dish: str) -> str:
    """Capitalise a Vietnamese dish name for display."""
    return " ".join(word.capitalize() for word in dish.split()) if dish else ""


detector = FoodDetector()
