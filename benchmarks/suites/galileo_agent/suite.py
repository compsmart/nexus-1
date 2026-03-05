from __future__ import annotations

from typing import Dict, List

from benchmarks.core.types import BatchContext, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results

from .adapters import to_batch_items
from .loader import load_cases
from .metrics import compute_metrics, judge_case


class GalileoAgentSuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        num_cases = int(self.config.get("num_cases", 24))
        category_weights = dict(self.config.get("category_weights", {}))
        cases = load_cases(num_cases=num_cases)
        items = to_batch_items(cases)

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

        case_results, _base_metrics = build_case_results(items, prediction_map, judge_case)
        baseline_metrics = compute_metrics(items, case_results, category_weights)
        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(items), "profile": run_ctx.run_spec.profile},
        )
