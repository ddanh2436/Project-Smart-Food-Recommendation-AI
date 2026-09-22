# -*- coding: utf-8 -*-
"""Score the search against search_eval.json, and compare runs.

`evaluate_search.py` answers one question — did anything relevant show up in
the top five — which is the right first question and not enough on its own.
Two changes in this codebase passed that check while breaking something else:
raising the rating weight globally kept the pass rate up while turning a search
for "Phở Thìn Lò Đúc" into "Phở Cần Thơ", and a relevance floor set inside the
TF-IDF noise band let nonsense queries return the whole database.

So this reports several numbers rather than one, and — the part that actually
prevents that class of regression — writes them to a file so the next run can
be diffed against the last:

    python benchmark.py --save baseline.json     # before a change
    python benchmark.py --against baseline.json  # after it

A metric that moved the wrong way is printed with its delta. Nothing here
decides whether a change is good; it makes the trade visible instead of
letting one number hide it.

Requires the service running (AI_EVAL_URL, default http://127.0.0.1:5000).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
API = os.getenv("AI_EVAL_URL", "http://127.0.0.1:5000").rstrip("/")

GREEN, RED, YELLOW, BLUE, DIM, END = (
    "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[90m", "\033[0m",
)


def haystack(item: dict) -> str:
    return f"{item.get('name', '')} {item.get('tags', '')}".lower()


def matches(item: dict, needles: list[str]) -> bool:
    text = haystack(item)
    return any(needle.lower() in text for needle in needles)


def check_intent(actual: dict, expected: dict) -> list[str]:
    """Field-by-field, returning the mismatches rather than a bare bool."""
    problems = []
    for field, want in expected.items():
        got = actual.get(field)
        if isinstance(want, list):
            # Order does not matter, and extra values are allowed: the case
            # asserts what must be understood, not what must be absent.
            missing = [v for v in want if v not in (got or [])]
            if want == [] and got:
                problems.append(f"{field}: expected empty, got {got}")
            elif missing:
                problems.append(f"{field}: missing {missing} (got {got})")
        elif got != want:
            problems.append(f"{field}: expected {want!r}, got {got!r}")
    return problems


def distinct_dishes(items: list[dict]) -> int:
    """How many different dishes a result list offers, by first tag word."""
    seen = set()
    for item in items:
        tags = str(item.get("tags", "")).lower()
        for word in ("bún chả", "bún bò", "bún đậu", "bún mắm", "bún thái",
                     "bún riêu", "phở", "cơm tấm", "hủ tiếu", "bánh mì",
                     "cháo", "mì", "xôi", "bún"):
            if word in tags:
                seen.add(word)
                break
    return len(seen)


def run(cases: list[dict], top_k: int = 5) -> dict:
    results = {
        "cases": len(cases),
        "top1": 0, "top5": 0, "scored": 0,
        "zero_results": 0, "relaxed": 0,
        "intent_checked": 0, "intent_ok": 0,
        "errors": 0,
    }
    latencies: list[float] = []
    failures: list[str] = []

    for case in cases:
        query = case["query"]
        started = time.time()
        try:
            response = requests.post(
                f"{API}/recommend", json={"query": query, "limit": top_k},
                timeout=60,
            )
            latencies.append((time.time() - started) * 1000)
            if response.status_code != 200:
                results["errors"] += 1
                failures.append(f"{query}: HTTP {response.status_code}")
                continue
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            results["errors"] += 1
            failures.append(f"{query}: {exc}")
            continue

        items = payload.get("scores", [])[:top_k]
        # A case that asserts emptiness is not a zero-result problem; counting
        # it as one made the rate report a success as a failure.
        if not items and not case.get("expect_empty"):
            results["zero_results"] += 1
        if payload.get("relaxed_filters"):
            results["relaxed"] += 1

        if case.get("intent"):
            results["intent_checked"] += 1
            problems = check_intent(payload.get("intent") or {}, case["intent"])
            if problems:
                failures.append(f"{query}: intent — {'; '.join(problems)}")
            else:
                results["intent_ok"] += 1

        # A case that asserts emptiness is scored on emptiness, not on hits.
        if case.get("expect_empty"):
            results["scored"] += 1
            if items:
                failures.append(
                    f"{query}: expected no results, got {len(items)} "
                    f"(top: {items[0].get('name', '')[:40]})"
                )
            else:
                results["top1"] += 1
                results["top5"] += 1
            continue

        if case.get("min_distinct_dishes"):
            found = distinct_dishes(items)
            results["scored"] += 1
            if found >= case["min_distinct_dishes"]:
                results["top1"] += 1
                results["top5"] += 1
            else:
                failures.append(
                    f"{query}: only {found} distinct dishes in top {top_k}, "
                    f"wanted {case['min_distinct_dishes']}"
                )
            continue

        needles = case.get("expect_top1") or case.get("expect_any")
        if not needles:
            continue
        results["scored"] += 1
        if items and matches(items[0], needles):
            results["top1"] += 1
            results["top5"] += 1
        elif any(matches(item, needles) for item in items):
            results["top5"] += 1
            if case.get("expect_top1"):
                failures.append(
                    f"{query}: expected {needles} at rank 1, got "
                    f"{items[0].get('name', '')[:40]}"
                )
        else:
            failures.append(
                f"{query}: no match for {needles} in top {top_k}"
                + (f" (top: {items[0].get('name', '')[:40]})" if items else "")
            )

    answerable = max(
        len([c for c in cases if not c.get("expect_empty")]), 1
    )
    scored = max(results["scored"], 1)
    checked = max(results["intent_checked"], 1)
    results["metrics"] = {
        "top1_accuracy": round(results["top1"] / scored * 100, 2),
        "top5_accuracy": round(results["top5"] / scored * 100, 2),
        "intent_accuracy": round(results["intent_ok"] / checked * 100, 2),
        "zero_result_rate": round(results["zero_results"] / answerable * 100, 2),
        "relaxation_rate": round(results["relaxed"] / len(cases) * 100, 2),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(
            sorted(latencies)[int(len(latencies) * 0.95) - 1], 1
        ) if len(latencies) >= 2 else 0,
        "errors": results["errors"],
    }
    results["failures"] = failures
    return results


# Which direction is an improvement, so a delta can be coloured honestly.
HIGHER_IS_BETTER = {
    "top1_accuracy": True,
    "top5_accuracy": True,
    "intent_accuracy": True,
    "zero_result_rate": False,
    "relaxation_rate": False,
    "latency_p50_ms": False,
    "latency_p95_ms": False,
    "errors": False,
}


def report(current: dict, baseline: dict | None) -> int:
    print(f"\n{BLUE}SEARCH BENCHMARK{END}  ({current['cases']} cases, {API})")
    print("=" * 62)

    regressed = 0
    for name, value in current["metrics"].items():
        line = f"  {name:<20} {value:>9}"
        if baseline:
            was = baseline.get("metrics", {}).get(name)
            if was is not None and was != value:
                delta = round(value - was, 2)
                better = (delta > 0) == HIGHER_IS_BETTER[name]
                colour = GREEN if better else RED
                if not better:
                    regressed += 1
                line += f"   {colour}{delta:+}{END}  {DIM}(was {was}){END}"
            elif was is not None:
                line += f"   {DIM}unchanged{END}"
        print(line)

    if current["failures"]:
        print(f"\n{YELLOW}Failing cases ({len(current['failures'])}):{END}")
        for failure in current["failures"]:
            print(f"  - {failure}")

    if baseline:
        print()
        if regressed:
            print(f"{RED}{regressed} metric(s) moved the wrong way.{END} "
                  "Decide whether the trade is worth it — this does not.")
        else:
            print(f"{GREEN}No metric regressed against the baseline.{END}")
    print()
    return regressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(BASE / "search_eval.json"))
    parser.add_argument("--save", help="write this run to a file as a baseline")
    parser.add_argument("--against", help="compare with a saved baseline")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    current = run(cases, top_k=args.top_k)

    baseline = None
    if args.against:
        path = Path(args.against)
        if path.exists():
            baseline = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"{YELLOW}No baseline at {path}, reporting absolute numbers.{END}")

    regressed = report(current, baseline)

    if args.save:
        Path(args.save).write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Baseline written to {args.save}\n")

    # Non-zero on a regression so this can gate a script, but never on a plain
    # run: a first run has nothing to regress against.
    return 1 if (baseline and regressed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
