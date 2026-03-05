from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunSpec:
    run_id: str
    name: str
    created_at: str
    agent: str = "nexus-1"
    schema_version: str = "2.0"
    profile: str = "quality_first"
    suites: List[str] = field(default_factory=list)
    baselines: List[str] = field(default_factory=list)
    primary_suite: str = "memory_recall"
    seed: int = 42
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    use_4bit: bool = True
    max_new_tokens: int = 128
    suite_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSpec":
        return cls(**data)


@dataclass
class RunRecord:
    run_id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    deleted_at: Optional[str] = None
    spec_path: Optional[str] = None
    results_path: Optional[str] = None
    metrics_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchItem:
    case_id: str
    prompt: str
    expected: str = ""
    keywords: List[str] = field(default_factory=list)
    category: str = "general"
    dataset: str = ""
    context_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchContext:
    suite_id: str
    items: List[BatchItem]
    run_spec: RunSpec
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    case_id: str
    answer: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    expected: str
    category: str
    dataset: str
    per_baseline: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteResult:
    suite_id: str
    baseline_metrics: Dict[str, Dict[str, float]]
    case_results: List[CaseResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "baseline_metrics": self.baseline_metrics,
            "case_results": [asdict(c) for c in self.case_results],
            "metadata": self.metadata,
        }


@dataclass
class AggregateScore:
    baseline_id: str
    overall_score: float
    suite_scores: Dict[str, float]
    metric_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankEstimate:
    baseline_id: str
    comparable: bool
    reference_source: str
    reference_date: str
    percentile: Optional[float] = None
    estimated_rank: Optional[int] = None
    total_models: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
