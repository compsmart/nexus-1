from __future__ import annotations

from typing import Dict

from benchmarks.core.types import BatchContext, BatchItem, CaseResult, Prediction, SuiteResult

from benchmarks.suites.multihop_hf.loader import load_multihop_cases

from .metrics import compute_metrics


class MultiHopSupportHFSuite:
    """Retrieval-support version of `multihop_hf`.

    This suite does not require a baseline to generate the correct natural
    language answer. Instead, it measures whether the baseline retrieved
    an answer-bearing context document in hop-1 and/or hop-2.

    Baselines should populate Prediction.metadata with:
      - hop1_hit: bool
      - hop2_hit: bool
      - any_hit: bool
    """

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
                    case_id=f"MHS{idx:05d}",
                    prompt=row.get("question", ""),
                    expected=row.get("answer", ""),
                    category="multi_hop_support",
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

        baseline_ids = sorted(prediction_map.keys())
        case_results = []
        for item in items:
            per_baseline: Dict[str, Dict[str, object]] = {}
            for baseline_id in baseline_ids:
                pred = prediction_map[baseline_id].get(item.case_id)
                meta = pred.metadata if pred is not None else {}
                # Default correctness definition for this suite.
                ok = bool((meta or {}).get("any_hit", False))
                per_baseline[baseline_id] = {
                    "answer": (pred.answer if pred is not None else ""),
                    "correct": ok,
                    "metadata": meta,
                }
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

        baseline_metrics = compute_metrics(items, case_results)
        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(items), "split": split},
        )
