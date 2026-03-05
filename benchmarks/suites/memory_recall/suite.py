from __future__ import annotations

from typing import Dict

from benchmarks.core.types import BatchContext, BatchItem, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results

from .cases import recall_questions, seed_facts
from .metrics import compute_metrics, judge_case


class MemoryRecallSuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        include_indirect = bool(self.config.get("include_indirect_questions", True))
        facts = seed_facts()
        rows = recall_questions(include_indirect_questions=include_indirect)
        items = [
            BatchItem(
                case_id=row["id"],
                prompt=row["q"],
                expected=" ".join(row["keywords"]),
                keywords=list(row["keywords"]),
                category="memory",
                dataset="memory_recall",
                context_docs=facts,
            )
            for row in rows
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

        case_results, _ = build_case_results(items, prediction_map, judge_case)
        baseline_metrics = compute_metrics(items, case_results)
        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(items)},
        )
