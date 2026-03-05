from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import read_json
from .types import AggregateScore, RankEstimate, utc_now_iso


class RankEstimator:
    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = snapshots_dir

    def _latest_snapshot(self) -> Optional[Path]:
        if not self.snapshots_dir.exists():
            return None
        candidates = sorted(self.snapshots_dir.glob("*.json"))
        return candidates[-1] if candidates else None

    def estimate(
        self,
        aggregate_score: AggregateScore,
        reference_source: str,
        snapshot_path: Optional[Path] = None,
    ) -> RankEstimate:
        path = snapshot_path or self._latest_snapshot()
        if path is None or not path.exists():
            return RankEstimate(
                baseline_id=aggregate_score.baseline_id,
                comparable=False,
                reference_source=reference_source,
                reference_date=utc_now_iso(),
                reason="No reference snapshot available.",
            )
        data = read_json(path)
        models: List[Dict] = data.get("models", [])
        score_field = data.get("score_field", "score")
        scores = [float(m.get(score_field)) for m in models if m.get(score_field) is not None]
        if not scores:
            return RankEstimate(
                baseline_id=aggregate_score.baseline_id,
                comparable=False,
                reference_source=reference_source,
                reference_date=data.get("created_at", utc_now_iso()),
                reason="Snapshot has no comparable score values.",
            )
        test_score = float(aggregate_score.overall_score)
        better_or_equal = sum(1 for score in scores if test_score >= score)
        percentile = (better_or_equal / len(scores)) * 100.0
        rank = 1 + sum(1 for score in scores if score > test_score)
        return RankEstimate(
            baseline_id=aggregate_score.baseline_id,
            comparable=True,
            reference_source=reference_source,
            reference_date=data.get("created_at", utc_now_iso()),
            percentile=percentile,
            estimated_rank=rank,
            total_models=len(scores) + 1,
            reason=f"Estimated against snapshot: {path.name}",
        )

