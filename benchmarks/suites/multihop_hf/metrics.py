from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from benchmarks.core.types import BatchItem, CaseResult


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def exact_match(answer: str, expected: str) -> bool:
    return _norm(answer) == _norm(expected)


def token_f1(answer: str, expected: str) -> float:
    a = _norm(answer).split()
    e = _norm(expected).split()
    if not a and not e:
        return 1.0
    if not a or not e:
        return 0.0
    ca = Counter(a)
    ce = Counter(e)
    common = sum((ca & ce).values())
    if common == 0:
        return 0.0
    p = common / len(a)
    r = common / len(e)
    return 2 * p * r / (p + r)


def judge_case(item: BatchItem, answer: str) -> bool:
    return exact_match(answer, item.expected) or _norm(item.expected) in _norm(answer)


def compute_metrics(items: List[BatchItem], case_results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    by_baseline: Dict[str, Dict[str, float]] = {}
    if not case_results:
        return by_baseline
    baseline_ids = sorted(case_results[0].per_baseline.keys())
    datasets = sorted({i.dataset for i in items})

    for baseline_id in baseline_ids:
        n = float(len(case_results)) or 1.0
        ok = 0.0
        f1_sum = 0.0
        by_ds = {d: {"n": 0.0, "ok": 0.0} for d in datasets}
        for row in case_results:
            ans = str(row.per_baseline[baseline_id]["answer"])
            corr = bool(row.per_baseline[baseline_id]["correct"])
            ok += 1.0 if corr else 0.0
            f1_sum += token_f1(ans, row.expected)
            by_ds[row.dataset]["n"] += 1.0
            by_ds[row.dataset]["ok"] += 1.0 if corr else 0.0

        metrics = {"exact_match": ok / n, "token_f1": f1_sum / n}
        for d in datasets:
            dn = by_ds[d]["n"] or 1.0
            metrics[f"{d}_exact_match"] = by_ds[d]["ok"] / dn
        by_baseline[baseline_id] = metrics
    return by_baseline

