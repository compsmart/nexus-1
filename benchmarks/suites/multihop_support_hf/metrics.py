from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchItem, CaseResult


def _get_flag(row: CaseResult, baseline_id: str, key: str) -> bool:
    meta = (row.per_baseline.get(baseline_id, {}) or {}).get("metadata") or {}
    return bool(meta.get(key, False))


def compute_metrics(items: List[BatchItem], case_results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    by_baseline: Dict[str, Dict[str, float]] = {}
    if not case_results:
        return by_baseline

    baseline_ids = sorted(case_results[0].per_baseline.keys())
    datasets = sorted({i.dataset for i in items if i.dataset})

    for baseline_id in baseline_ids:
        n = float(len(case_results)) or 1.0
        hop1_ok = 0.0
        hop2_ok = 0.0
        any_ok = 0.0
        by_ds = {d: {"n": 0.0, "any": 0.0} for d in datasets}

        for row in case_results:
            if _get_flag(row, baseline_id, "hop1_hit"):
                hop1_ok += 1.0
            if _get_flag(row, baseline_id, "hop2_hit"):
                hop2_ok += 1.0
            if _get_flag(row, baseline_id, "any_hit"):
                any_ok += 1.0
                if row.dataset in by_ds:
                    by_ds[row.dataset]["any"] += 1.0
            if row.dataset in by_ds:
                by_ds[row.dataset]["n"] += 1.0

        metrics = {
            "hop1_answer_doc_hit_rate": hop1_ok / n,
            "hop2_answer_doc_hit_rate": hop2_ok / n,
            "any_answer_doc_hit_rate": any_ok / n,
        }
        for d in datasets:
            dn = by_ds[d]["n"] or 1.0
            metrics[f"{d}_any_answer_doc_hit_rate"] = by_ds[d]["any"] / dn

        by_baseline[baseline_id] = metrics

    return by_baseline
