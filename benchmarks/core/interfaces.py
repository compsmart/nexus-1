from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

from .types import AggregateScore, BatchContext, RunSpec, SuiteResult


@runtime_checkable
class BaselineAdapter(Protocol):
    baseline_id: str

    def answer(self, batch_ctx: BatchContext):
        """Generate predictions for all items in a batch context."""

    def close(self) -> None:
        """Release resources."""


@runtime_checkable
class BenchmarkSuite(Protocol):
    suite_id: str

    def run(self, run_ctx: "RunContext") -> SuiteResult:
        """Execute suite and return normalized result."""


@runtime_checkable
class ScoreAggregator(Protocol):
    def aggregate(self, suite_results: list[SuiteResult]) -> list[AggregateScore]:
        """Aggregate suite-level scores into run-level baseline scores."""


@dataclass
class RunContext:
    run_spec: RunSpec
    run_dir: Path
    baselines: Dict[str, BaselineAdapter]
    suite_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    progress_sink: Any
    runtime_config: Dict[str, Any] = field(default_factory=dict)

