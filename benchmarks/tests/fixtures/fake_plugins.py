from __future__ import annotations

from benchmarks.core.types import BatchContext, Prediction, SuiteResult


class FakeBaseline:
    def __init__(self, baseline_id: str, run_spec, config: dict):
        self.baseline_id = baseline_id
        self.run_spec = run_spec
        self.config = config

    def answer(self, batch_ctx: BatchContext):
        return [
            Prediction(case_id=item.case_id, answer=item.expected)
            for item in batch_ctx.items
        ]

    def close(self) -> None:
        return


class FakeSuite:
    def __init__(self, suite_id: str, config: dict):
        self.suite_id = suite_id
        self.config = config

    def run(self, run_ctx):
        return SuiteResult(
            suite_id=self.suite_id,
            baseline_metrics={bid: {"exact_match": 1.0} for bid in run_ctx.baselines.keys()},
            case_results=[],
            metadata={"fake": True},
        )

