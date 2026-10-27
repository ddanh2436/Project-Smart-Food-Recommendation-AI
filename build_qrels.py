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
as a pool with grade null. Fill those in by hand; benchmark.py skips a query
until it has at least one grade, and reports how much of each top 10 was judged.

Grades already in the file are kept, so hand edits survive a rebuild. To
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
POOL_DEPTH = 10


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


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    store.reload()
    if store.frame.empty:
        raise SystemExit(f"No restaurants loaded: {store.status()['load_error']}")
    names = dict(zip(store.frame["id"], store.frame["name"]))

    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    old = {}
    if QRELS.exists():
        old = {qrel_key(q): q for q in json.loads(QRELS.read_text(encoding="utf-8"))["queries"]}

    out = []
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
        # Hand edits win over the rules.
        for item in (old.get(key) or {}).get("judgments", []):
            grades[item["id"]] = item["grade"]
        # Names only where a person has to grade by hand; on complete queries
        # they tripled the file for nothing the benchmark reads.
        judgments = sorted(
            (
                {"id": rid, "grade": g, **({} if complete else {"name": names.get(rid, "?")})}
                for rid, g in grades.items()
            ),
            key=lambda j: (-(j["grade"] if j["grade"] is not None else -1), j["id"]),
        )
        entry = {"query": query, "complete": complete, "judgments": judgments}
        if case.get("user_gps"):
            entry["user_gps"] = case["user_gps"]
        out.append(entry)

    # One query per line: diffable, and small enough to commit.
    lines = ",\n".join(json.dumps(q, ensure_ascii=False) for q in out)
    QRELS.write_text('{"queries": [\n' + lines + "\n]}\n", encoding="utf-8")
    todo = sum(1 for q in out for j in q["judgments"] if j["grade"] is None)
    print(f"{len(out)} queries -> {QRELS.name}; {todo} pooled judgments await a grade")


if __name__ == "__main__":
    main()
