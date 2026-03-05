from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchItem, CaseResult
from benchmarks.suites.common import keyword_match


def judge_case(item: BatchItem, answer: str) -> bool:
    return keyword_match(answer, item.keywords)


def compute_metrics(items: List[BatchItem], case_results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    by_baseline: Dict[str, Dict[str, float]] = {}
    if not case_results:
        return by_baseline
    baseline_ids = sorted(case_results[0].per_baseline.keys())
    cats = sorted({i.category for i in items})
    for baseline_id in baseline_ids:
        n = float(len(case_results)) or 1.0
        ok = sum(1.0 for row in case_results if row.per_baseline[baseline_id]["correct"])
        metrics = {"transfer_accuracy": ok / n}
        for cat in cats:
            cat_rows = [r for r in case_results if r.category == cat]
            cn = float(len(cat_rows)) or 1.0
            cok = sum(1.0 for r in cat_rows if r.per_baseline[baseline_id]["correct"])
            metrics[f"{cat}_accuracy"] = cok / cn
        by_baseline[baseline_id] = metrics
    return by_baseline

