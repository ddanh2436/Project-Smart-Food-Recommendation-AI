# -*- coding: utf-8 -*-
"""Write sentiment_to_grade.json: reviews to label by hand, the gold test set
for the 4-label sentiment model (kaggle/README.md, notebook 7).

Measured against the reviewers' own scores, the current 3-label model is right
at the ends (>= 8: 96% positive, <= 4: 92% negative) and unreliable in between:
it has no "mixed" label, so "ngon nhưng giá chát" is forced to one side. The
sample is weighted to that middle band, where the errors are.

One review per line, no IDs; the notebook drops these texts from training.
Refuses to overwrite a sheet that already has labels.

    python sentiment_sheet.py            # needs MONGO_URI
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pymongo import MongoClient

from config import settings

SHEET = Path(__file__).resolve().parent / "sentiment_to_grade.json"
PER_BAND = {"high": 120, "middle": 200, "low": 80}  # 400 in all
HOW_TO = (
    "label: POS = khen là chính; NEG = chê là chính; MIX = vừa khen vừa chê rõ ràng "
    "(\"ngon nhưng giá chát\"); NEU = kể lại, hỏi, không bày tỏ ý kiến. Chấm theo "
    "NỘI DUNG, không theo điểm: điểm chỉ để tham khảo. null = chưa chấm."
)


def band(score) -> str | None:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    return "high" if score >= 8 else "low" if score <= 4 else "middle"


def main() -> None:
    if SHEET.exists():
        done = sum(r["label"] is not None for r in json.loads(SHEET.read_text(encoding="utf-8"))["reviews"])
        if done:
            raise SystemExit(f"{SHEET.name} already has {done} labels; delete it to resample.")

    reviews = MongoClient(settings.mongo_uri)[settings.db_name][settings.reviews_collection]
    pools: dict[str, list] = {b: [] for b in PER_BAND}
    seen: set[str] = set()
    for row in reviews.find({}, {"noiDung": 1, "diemReview": 1}):
        text = " ".join((row.get("noiDung") or "").split())
        b = band(row.get("diemReview"))
        if b and len(text) >= 20 and text not in seen:
            seen.add(text)
            pools[b].append({"score": float(row["diemReview"]), "text": text, "label": None})

    rng = random.Random(2026)  # fixed, so a rerun picks the same reviews
    picked = [r for b, n in PER_BAND.items() for r in rng.sample(pools[b], min(n, len(pools[b])))]
    rng.shuffle(picked)  # bands mixed, so the score order does not prime the grader
    lines = ",\n  ".join(json.dumps(r, ensure_ascii=False) for r in picked)
    SHEET.write_text(
        "{" + f'"cach_cham": {json.dumps(HOW_TO, ensure_ascii=False)},\n"reviews": [\n  {lines}\n]}}\n',
        encoding="utf-8",
    )
    print(f"{len(picked)} reviews -> {SHEET.name}")


if __name__ == "__main__":
    main()
