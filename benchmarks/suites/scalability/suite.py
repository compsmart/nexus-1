"""Scalability suite: accuracy degradation as K grows (10→500).

Ported from nexus-2's ScalabilitySuite, adapted to the
nexus-1 BatchContext/Prediction interface.
"""

from __future__ import annotations

import random
from typing import Dict, List

from benchmarks.core.types import BatchContext, BatchItem, Prediction, SuiteResult
from benchmarks.suites.common import build_case_results, contains_expected

_TEMPLATES = [
    "{entity} LIKES {attr}",
    "{entity} OWNS {attr}",
]

_ATTRS = [
    "red", "blue", "green", "gold", "silver",
    "iron", "oak", "pine", "elm", "maple",
    "hawk", "wolf", "bear", "fox", "deer",
    "swan", "crow", "dove", "lynx", "seal",
]


def _judge_case(item: BatchItem, answer: str) -> bool:
    return contains_expected(answer, item.expected)


class ScalabilitySuite:
    def __init__(self, suite_id: str, config: Dict):
        self.suite_id = suite_id
        self.config = config or {}

    def run(self, run_ctx) -> SuiteResult:
        k_values = list(self.config.get("k_values", [10, 25, 50, 100, 200, 500]))
        n_queries = int(self.config.get("n_queries", 20))
        rng = random.Random(42)

        # Build all items across K values
        all_items: List[BatchItem] = []
        for k in k_values:
            facts: Dict[str, str] = {}
            context_docs: List[str] = []
            for i in range(k):
                entity = f"Entity_{i:04d}"
                attr = rng.choice(_ATTRS)
                template = rng.choice(_TEMPLATES)
                text = template.format(entity=entity, attr=attr)
                facts[entity] = attr
                context_docs.append(text)

            entities = list(facts.keys())
            query_ents = rng.sample(entities, min(n_queries, len(entities)))
            for ent in query_ents:
                all_items.append(BatchItem(
                    case_id=f"SCALE_K{k}_{ent}",
                    prompt=f"What does {ent} like or own?",
                    expected=facts[ent],
                    category=f"k_{k}",
                    dataset="scalability",
                    context_docs=list(context_docs),
                ))

        # Run each baseline
        prediction_map: Dict[str, Dict[str, Prediction]] = {}
        for baseline_id, baseline in run_ctx.baselines.items():
            run_ctx.progress_sink.emit(
                "suite_batch",
                f"{self.suite_id}: baseline={baseline_id} cases={len(all_items)}",
            )
            batch_ctx = BatchContext(
                suite_id=self.suite_id,
                items=all_items,
                run_spec=run_ctx.run_spec,
                metadata={"suite_config": self.config},
            )
            preds = baseline.answer(batch_ctx)
            prediction_map[baseline_id] = {p.case_id: p for p in preds}

        case_results, _ = build_case_results(all_items, prediction_map, _judge_case)

        # Compute metrics per baseline
        baseline_metrics: Dict[str, Dict[str, float]] = {}
        baseline_ids = sorted(prediction_map.keys())
        for bid in baseline_ids:
            total_correct = sum(
                1 for cr in case_results if cr.per_baseline.get(bid, {}).get("correct", False)
            )
            accuracy = total_correct / len(case_results) if case_results else 0.0
            accuracy_by_k: Dict[str, float] = {}
            for k in k_values:
                cat = f"k_{k}"
                cat_rows = [cr for cr in case_results if cr.category == cat]
                if cat_rows:
                    cat_ok = sum(1 for cr in cat_rows if cr.per_baseline.get(bid, {}).get("correct", False))
                    accuracy_by_k[str(k)] = cat_ok / len(cat_rows)

            baseline_metrics[bid] = {
                "accuracy": accuracy,
                "accuracy_by_k": accuracy_by_k,
            }

        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics=baseline_metrics,
            case_results=case_results,
            metadata={"num_cases": len(all_items), "k_values": k_values},
        )
