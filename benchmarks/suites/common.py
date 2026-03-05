from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from benchmarks.core.types import BatchItem, CaseResult, Prediction


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def exact_match(answer: str, expected: str) -> bool:
    return normalize_text(answer) == normalize_text(expected)


def contains_expected(answer: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(answer)


def keyword_match(answer: str, keywords: Iterable[str]) -> bool:
    low = normalize_text(answer)
    return all(normalize_text(k) in low for k in keywords if k)


def build_case_results(
    items: List[BatchItem],
    prediction_map: Dict[str, Dict[str, Prediction]],
    judge_fn,
) -> Tuple[List[CaseResult], Dict[str, Dict[str, float]]]:
    case_results: List[CaseResult] = []
    metrics: Dict[str, Dict[str, float]] = {}
    baseline_ids = sorted(prediction_map.keys())
    for baseline_id in baseline_ids:
        metrics[baseline_id] = {"cases": float(len(items)), "correct": 0.0}

    for item in items:
        per_baseline: Dict[str, Dict[str, object]] = {}
        for baseline_id in baseline_ids:
            pred = prediction_map[baseline_id].get(item.case_id)
            answer = pred.answer if pred is not None else ""
            ok = bool(judge_fn(item, answer))
            per_baseline[baseline_id] = {
                "answer": answer,
                "correct": ok,
                "metadata": (pred.metadata if pred is not None else {}),
            }
            metrics[baseline_id]["correct"] += 1.0 if ok else 0.0
        case_results.append(
            CaseResult(
                case_id=item.case_id,
                prompt=item.prompt,
                expected=item.expected,
                category=item.category,
                dataset=item.dataset,
                per_baseline=per_baseline,
                metadata=item.metadata,
            )
        )

    for baseline_id in baseline_ids:
        total = max(1.0, metrics[baseline_id]["cases"])
        metrics[baseline_id]["accuracy"] = metrics[baseline_id]["correct"] / total
    return case_results, metrics
