from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchItem, CaseResult
from benchmarks.suites.common import contains_expected, keyword_match


def judge_case(item: BatchItem, answer: str) -> bool:
    return keyword_match(answer, item.keywords) or contains_expected(answer, item.expected)


def compute_metrics(items: List[BatchItem], case_results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    by_baseline: Dict[str, Dict[str, float]] = {}
    if not case_results:
        return by_baseline
    baseline_ids = sorted(case_results[0].per_baseline.keys())
    for baseline_id in baseline_ids:
        n = float(len(case_results)) or 1.0
        ok = sum(1.0 for row in case_results if row.per_baseline[baseline_id]["correct"])
        by_baseline[baseline_id] = {"exact_match": ok / n, "logic_accuracy": ok / n}
    return by_baseline

