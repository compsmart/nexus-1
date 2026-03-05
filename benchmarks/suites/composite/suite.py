"""Composite suite: mixed-mode session with recall, multihop, reasoning, learning.

Evolved from galileo_agent suite with standardised category scoring.
"""

from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchContext, BatchItem, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results, contains_expected, keyword_match

from .cases import composite_cases


def _judge_case(item: BatchItem, answer: str) -> bool:
    if item.keywords:
        return keyword_match(answer, item.keywords)
    return contains_expected(answer, item.expected)


class CompositeSuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        num_cases = int(self.config.get("num_cases", 24))
        category_weights = dict(self.config.get("category_weights", {
            "reasoning": 0.30, "memory": 0.25, "multi_hop": 0.25, "learning": 0.20,
        }))

        cases = composite_cases(num_cases=num_cases)
        items = [
            BatchItem(
                case_id=c["id"],
                prompt=c["q"],
                expected=c.get("expected", " ".join(c.get("keywords", []))),
                keywords=c.get("keywords", []),
                category=c.get("category", "general"),
                dataset="composite",
                context_docs=c.get("context_docs", []),
            )
            for c in cases
        ]

        prediction_map: Dict[str, Dict[str, Prediction]] = {}
        for baseline_id, baseline in run_ctx.baselines.items():
            run_ctx.progress_sink.emit(
                "suite_batch",
                f"{self.suite_id}: baseline={baseline_id} cases={len(items)}",
            )
            batch_ctx = BatchContext(
                suite_id=self.suite_id,
                items=items,
                run_spec=run_ctx.run_spec,
                metadata={"suite_config": self.config},
            )
            preds = baseline.answer(batch_ctx)
            prediction_map[baseline_id] = {p.case_id: p for p in preds}

        case_results, _ = build_case_results(items, prediction_map, _judge_case)

        # Compute per-baseline metrics
        baseline_metrics: Dict[str, Dict[str, float]] = {}
        baseline_ids = sorted(prediction_map.keys())
        categories = sorted({i.category for i in items})

        for bid in baseline_ids:
            cat_acc: Dict[str, float] = {}
            for cat in categories:
                cat_rows = [cr for cr in case_results if cr.category == cat]
                if cat_rows:
                    ok = sum(1 for cr in cat_rows if cr.per_baseline.get(bid, {}).get("correct", False))
                    cat_acc[cat] = ok / len(cat_rows)
                else:
                    cat_acc[cat] = 0.0

            # Weighted composite score
            total_w = sum(category_weights.get(c, 0.0) for c in categories) or 1.0
            composite_score = sum(
                cat_acc.get(c, 0.0) * category_weights.get(c, 0.0)
                for c in categories
            ) / total_w

            overall_correct = sum(
                1 for cr in case_results if cr.per_baseline.get(bid, {}).get("correct", False)
            )
            accuracy = overall_correct / len(case_results) if case_results else 0.0

            baseline_metrics[bid] = {
                "accuracy": accuracy,
                "composite_score": composite_score,
                "category_accuracy": cat_acc,
            }

        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(items), "profile": run_ctx.run_spec.profile},
        )
