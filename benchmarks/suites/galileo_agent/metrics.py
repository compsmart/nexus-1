from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchItem, CaseResult
from benchmarks.suites.common import contains_expected, keyword_match


def judge_case(item: BatchItem, answer: str) -> bool:
    if item.keywords:
        return keyword_match(answer, item.keywords)
    return contains_expected(answer, item.expected)


def compute_metrics(
    items: List[BatchItem],
    case_results: List[CaseResult],
    category_weights: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    by_baseline: Dict[str, Dict[str, float]] = {}
    if not case_results:
        return by_baseline
    baseline_ids = sorted(case_results[0].per_baseline.keys())
    categories = sorted({i.category for i in items})

    for baseline_id in baseline_ids:
        counts = {c: {"n": 0.0, "ok": 0.0} for c in categories}
        for row in case_results:
            cat = row.category
            ok = bool(row.per_baseline[baseline_id]["correct"])
            counts[cat]["n"] += 1.0
            counts[cat]["ok"] += 1.0 if ok else 0.0

        metrics = {}
        total_n = sum(v["n"] for v in counts.values()) or 1.0
        total_ok = sum(v["ok"] for v in counts.values())
        metrics["exact_match"] = total_ok / total_n
        for cat in categories:
            denom = counts[cat]["n"] or 1.0
            metrics[f"{cat}_accuracy"] = counts[cat]["ok"] / denom
        weighted = 0.0
        weight_sum = 0.0
        for cat in categories:
            w = float(category_weights.get(cat, 0.0))
            weighted += metrics[f"{cat}_accuracy"] * w
            weight_sum += w
        metrics["composite_score"] = weighted / (weight_sum or 1.0)
        by_baseline[baseline_id] = metrics
    return by_baseline

