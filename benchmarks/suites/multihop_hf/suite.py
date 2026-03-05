from __future__ import annotations

from typing import Dict

from benchmarks.core.types import BatchContext, BatchItem, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results

from .loader import load_multihop_cases
from .metrics import compute_metrics, judge_case


class MultiHopHFSuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        split = self.config.get("split", "validation")
        dataset_samples = dict(self.config.get("dataset_samples", {}))
        docs_per_case = int(self.config.get("docs_per_case", 8))
        rows = load_multihop_cases(
            dataset_cfg=run_ctx.dataset_config,
            split=split,
            dataset_samples=dataset_samples,
            docs_per_case=docs_per_case,
        )
        items = []
        for idx, row in enumerate(rows, start=1):
            items.append(
                BatchItem(
                    case_id=f"MH{idx:05d}",
                    prompt=row["question"],
                    expected=row["answer"],
                    category="multi_hop",
                    dataset=row.get("dataset", "unknown"),
                    context_docs=list(row.get("context_docs", [])),
                    metadata={"split": split},
                )
            )

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

        case_results, _raw = build_case_results(items, prediction_map, judge_case)
        baseline_metrics = compute_metrics(items, case_results)
        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(items), "split": split},
        )
