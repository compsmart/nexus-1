from __future__ import annotations

from typing import Dict

from benchmarks.core.types import BatchContext, BatchItem, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results

from .apply_phase import application_tasks
from .metrics import compute_metrics, judge_case
from .teach_phase import teaching_docs


class LearningTransferSuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        reps = int(self.config.get("task_repetitions", 1))
        docs = teaching_docs()
        rows = application_tasks(repetitions=reps)
        items = [
            BatchItem(
                case_id=row["id"],
                prompt=row["q"],
                expected=" ".join(row["keywords"]),
                keywords=row["keywords"],
                category=row.get("category", "transfer"),
                dataset="learning_transfer",
                context_docs=docs,
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
