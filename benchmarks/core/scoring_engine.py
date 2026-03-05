from __future__ import annotations

from typing import Dict, List

from .types import AggregateScore, SuiteResult


class WeightedScoreAggregator:
    def __init__(self, scoring_cfg: dict):
        defaults = scoring_cfg.get("defaults", {})
        self.suite_weights: Dict[str, float] = defaults.get("suite_weights", {})
        self.metric_key_by_suite: Dict[str, str] = defaults.get("metric_key_by_suite", {})
        self.fallback_metric_keys: List[str] = defaults.get(
            "fallback_metric_keys",
            ["exact_match", "accuracy", "keyword_accuracy"],
        )
        self.normalize_weights: bool = bool(defaults.get("normalize_suite_weights", True))

    def _suite_metric_value(self, suite_result: SuiteResult, baseline_id: str) -> float:
        metrics = suite_result.baseline_metrics.get(baseline_id, {})
        metric_key = self.metric_key_by_suite.get(suite_result.suite_id)
        if metric_key and metric_key in metrics:
            return float(metrics[metric_key])
        for key in self.fallback_metric_keys:
            if key in metrics:
                return float(metrics[key])
        if metrics:
            return float(sum(metrics.values()) / len(metrics))
        return 0.0

    def aggregate(self, suite_results: List[SuiteResult]) -> List[AggregateScore]:
        baseline_ids = sorted(
            {
                baseline_id
                for suite_result in suite_results
                for baseline_id in suite_result.baseline_metrics.keys()
            }
        )
        raw_weights = {
            suite_result.suite_id: float(self.suite_weights.get(suite_result.suite_id, 1.0))
            for suite_result in suite_results
        }
        if self.normalize_weights:
            total = sum(raw_weights.values()) or 1.0
            weights = {key: value / total for key, value in raw_weights.items()}
        else:
            weights = raw_weights

        out: List[AggregateScore] = []
        for baseline_id in baseline_ids:
            suite_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            for suite_result in suite_results:
                value = self._suite_metric_value(suite_result, baseline_id)
                suite_scores[suite_result.suite_id] = value
                weighted_sum += value * weights.get(suite_result.suite_id, 0.0)
            out.append(
                AggregateScore(
                    baseline_id=baseline_id,
                    overall_score=weighted_sum,
                    suite_scores=suite_scores,
                    metric_breakdown={"weighted_sum": weighted_sum},
                )
            )
        return sorted(out, key=lambda x: x.overall_score, reverse=True)

