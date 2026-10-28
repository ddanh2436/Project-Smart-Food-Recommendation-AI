# -*- coding: utf-8 -*-
"""Build search_qrels.json: per query, which restaurant IDs are relevant.

benchmark.py used to ask "does a name or tag in the top 5 contain one of these
words". That cannot tell a better ranking from a worse one -- any hit passes --
and it is tied to how results are worded rather than to which places they are.
A qrels file names the places, so a new embedding model or reranker is scored
on the same frozen judgments as the old one.

Grades, judged against EVERY restaurant in the store, not just the ones the
current search returns (otherwise a new model that finds a relevant place the
old one missed would be scored as wrong):

    name query ("Phở Thìn Lò Đúc"): 3 when every expected word is in the name
    otherwise, for rows whose district/city/price fit the case's intent (and,
    for a case with user_gps and max_km, that lie within max_km):
      +1 an expected word is in the name or tags
      +1 one is in the NAME (a specialist, not a place that also serves it)
      +1 two or more different expected words match
    0  everything else (not stored; a complete query treats missing as 0)

Words match whole and with their diacritics: folded, "dê" became "de" and
matched three thousand names.

Queries the rules cannot grade ("trời mưa nên ăn gì") get the current top 10
as a pool with grade null. benchmark.py skips a query until it has at least
one grade, and reports how much of each top 10 was judged.

Those are graded by hand in qrels_to_grade.json: query, name, the facts a
grader needs (district, price, score, hours, distance, review shares) and the
grade -- no IDs. A rebuild reads the grades back into search_qrels.json, which
stays the machine file: chains share names, so the benchmark needs the IDs.

Grades already in either file are kept, so hand edits survive a rebuild. To
regrade a query from changed rules, delete its entry and rebuild.

    python build_qrels.py            # needs MONGO_URI, like the service
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

import text_utils as tu
from data_store import store
from search import _distance_column, search

BASE = Path(__file__).resolve().parent
CASES = BASE / "search_eval.json"
QRELS = BASE / "search_qrels.json"
SHEET = BASE / "qrels_to_grade.json"
POOL_DEPTH = 10
HOW_TO = (
    "grade: 3 = đúng y yêu cầu, 2 = hợp, 1 = hơi liên quan, 0 = không liên quan "
    "hoặc vi phạm yêu cầu; null = chưa chấm. Chấm xong chạy: python build_qrels.py"
)
ASPECT_LABELS = {
    "food": "món ăn", "price": "giá", "service": "phục vụ",
    "space": "không gian", "hygiene": "vệ sinh", "parking": "đậu xe",
}


def qrel_key(item: dict) -> tuple:
    """The same words asked from a known position are a different question."""
    return item["query"], tuple(item.get("user_gps") or ())


def _word(text: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(tu.normalize(text))}(?!\w)")


def grade_rows(case: dict) -> dict[str, int]:
    frame = store.frame
    names = frame["name_norm"]
    tags = frame["tags_norm"]
    intent = case.get("intent") or {}

    if case.get("expect_top1"):
        words = [_word(w) for w in case["expect_top1"]]
        hit = names.map(lambda n: all(w.search(n) for w in words))
        return {i: 3 for i in frame.loc[hit, "id"]}

    needles = [_word(n) for n in case.get("expect_any") or []]
    fits = pd.Series(True, index=frame.index)
    if intent.get("districts"):
        fits &= frame["district"].isin(intent["districts"])
    if intent.get("cities"):
        fits &= frame["city"].isin(intent["cities"])
    if case.get("user_gps") and case.get("max_km"):
        # A Cần Thơ lẩu mắm is not an answer to "lẩu mắm" asked in Quận 1.
        fits &= _distance_column(frame, case["user_gps"]) <= case["max_km"]
    if intent.get("price_max"):
        # Unknown price is not a violation, as in search.py.
        fits &= (frame["price"] <= intent["price_max"]) | (frame["price"] <= 0)

    grades: dict[str, int] = {}
    for rid, name, tag, ok in zip(frame["id"], names, tags, fits):
        if not ok:
            continue
        in_name = [n for n in needles if n.search(name)]
        matched = {n.pattern for n in needles if n.search(name) or n.search(tag)}
        if matched:
            grades[rid] = 1 + bool(in_name) + (len(matched) >= 2)
    return grades


def describe(row: pd.Series, km: float | None) -> str:
    """What a grader needs to judge one place, on one line."""
    parts = [row["district"] or row["city"], row["price_text"]]
    if row["rating"]:
        parts.append(f"{row['rating']:.1f}/10")
    parts.append(row["opening_hours"])
    if km is not None and km == km:  # NaN: no coordinates
        parts.append(f"cách {km:.1f} km")
    shares = [
        f"{label} {row[f'aspect_{key}']:.0%}"
        for key, label in ASPECT_LABELS.items()
        if row[f"aspect_{key}_n"] >= 3
    ]
    if shares:
        parts.append("review khen: " + ", ".join(shares))
    parts.append(row["tags"][:120])
    return " · ".join(p for p in parts if p)


def labels_for(ids: list[str], names: dict[str, str]) -> dict[str, str]:
    """ID -> the name a grader sees, made unique within one query."""
    out: dict[str, str] = {}
    for rid in ids:
        label, n = names.get(rid, "?"), 2
        while label in out.values():
            label, n = f"{names.get(rid, '?')} ({n})", n + 1
        out[rid] = label
    return out


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    store.reload()
    if store.frame.empty:
        raise SystemExit(f"No restaurants loaded: {store.status()['load_error']}")
    names = dict(zip(store.frame["id"], store.frame["name"]))
    rows = store.frame.set_index("id")

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    old = {}
    if QRELS.exists():
        old = {qrel_key(q): q for q in json.loads(QRELS.read_text(encoding="utf-8"))["queries"]}
    sheet = {}
    if SHEET.exists():
        sheet = {
            qrel_key(q): {p["name"]: p["grade"] for p in q["places"]}
            for q in json.loads(SHEET.read_text(encoding="utf-8"))["queries"]
        }

    out = []
    to_grade = []
    seen: set[tuple] = set()
    for case in cases:
        if case.get("expect_empty") or case.get("min_distinct_dishes"):
            continue
        query = case["query"]
        key = qrel_key(case)
        if key in seen:  # judgments are per query and position; first case wins
            continue
        seen.add(key)
        complete = bool(case.get("expect_top1") or case.get("expect_any"))
        if complete:
            grades: dict[str, int | None] = grade_rows(case)
        else:
            result = search(query, user_gps=case.get("user_gps"), limit=POOL_DEPTH)
            grades = {rid: None for rid in result.rows["id"]}
        # Hand edits win over the rules, and the sheet is the newest edit.
        for item in (old.get(key) or {}).get("judgments", []):
            grades[item["id"]] = item["grade"]
        if not complete:
            labels = labels_for(sorted(grades), names)
            for rid, label in labels.items():
                if sheet.get(key, {}).get(label) is not None:
                    grades[rid] = sheet[key][label]
        judgments = sorted(
            ({"id": rid, "grade": g} for rid, g in grades.items()),
            key=lambda j: (-(j["grade"] if j["grade"] is not None else -1), j["id"]),
        )
        entry = {"query": query, "complete": complete, "judgments": judgments}
        if case.get("user_gps"):
            entry["user_gps"] = case["user_gps"]
        out.append(entry)
        if not complete and grades:
            km = (
                pd.Series(_distance_column(store.frame, case["user_gps"]).values, index=store.frame["id"])
                if case.get("user_gps") else {}
            )
            to_grade.append({
                **{k: entry[k] for k in ("query", "user_gps") if k in entry},
                "places": [
                    {
                        "name": labels[rid],
                        "info": describe(rows.loc[rid], km.get(rid)) if rid in rows.index else "",
                        "grade": grades[rid],
                    }
                    for rid in sorted(grades, key=labels.get)
                ],
            })

    # One query per line: diffable, and small enough to commit.
    lines = ",\n".join(json.dumps(q, ensure_ascii=False) for q in out)
    QRELS.write_text('{"queries": [\n' + lines + "\n]}\n", encoding="utf-8")
    # One place per line, so a grade is edited where its name is read.
    blocks = []
    for q in to_grade:
        head = json.dumps({k: v for k, v in q.items() if k != "places"}, ensure_ascii=False)[:-1]
        places = ",\n    ".join(json.dumps(p, ensure_ascii=False) for p in q["places"])
        blocks.append(f'{head}, "places": [\n    {places}\n  ]}}')
    SHEET.write_text(
        "{" + f'"cach_cham": {json.dumps(HOW_TO, ensure_ascii=False)},\n"queries": [\n  '
        + ",\n  ".join(blocks) + "\n]}\n",
        encoding="utf-8",
    )
    todo = sum(1 for q in out for j in q["judgments"] if j["grade"] is None)
    print(f"{len(out)} queries -> {QRELS.name}; {todo} await a grade in {SHEET.name}")


if __name__ == "__main__":
    main()
