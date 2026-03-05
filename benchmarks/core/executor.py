from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import ensure_dir, write_json, write_text
from .interfaces import RunContext
from .progress import JsonlProgressSink
from .rank_estimator import RankEstimator
from .registry import BenchmarkRegistry
from .run_store import RunStore
from .scoring_engine import WeightedScoreAggregator
from .types import RunSpec, SuiteResult, utc_now_iso


class BenchmarkExecutor:
    def __init__(self, benchmark_root: Path):
        self.root = benchmark_root
        ensure_dir(self.root)
        self.registry = BenchmarkRegistry(self.root)
        self.run_store = RunStore(self.root / "runs" / "index.sqlite")
        self.scoring = WeightedScoreAggregator(self.registry.scoring_cfg)
        self.galileo_rank = RankEstimator(
            self.root / "reference_data" / "leaderboards" / "galileo" / "snapshots"
        )

    def build_default_run_spec(
        self,
        name: str = "benchmark-run",
        suites: Optional[List[str]] = None,
        baselines: Optional[List[str]] = None,
        profile: Optional[str] = None,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        use_4bit: bool = True,
    ) -> RunSpec:
        rid = uuid.uuid4().hex[:12]
        return RunSpec(
            run_id=rid,
            name=name,
            created_at=utc_now_iso(),
            agent="nexus-1",
            schema_version="2.0",
            profile=profile or self.registry.default_profile(),
            suites=suites or self.registry.default_suites(),
            baselines=baselines or self.registry.default_baselines(),
            primary_suite=self.registry.default_primary_suite(),
            model_name=model_name,
            use_4bit=use_4bit,
        )

    def run(self, run_spec: RunSpec) -> Dict:
        if not run_spec.run_id:
            run_spec.run_id = uuid.uuid4().hex[:12]
        run_dir = self.root / "runs" / run_spec.run_id
        ensure_dir(run_dir / "artifacts")
        events_path = run_dir / "events.jsonl"
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        spec_path = run_dir / "run_spec.json"
        results_path = run_dir / "results.json"
        metrics_path = run_dir / "metrics.json"
        status_path = run_dir / "status.json"

        self.run_store.create(run_spec)
        self.run_store.update_status(run_spec.run_id, "running")
        progress = JsonlProgressSink(events_path)
        progress.emit("run_start", f"Run started: {run_spec.run_id}", {"run_spec": run_spec.to_dict()})
        write_json(spec_path, run_spec.to_dict(), immutable=True)
        write_text(stdout_path, "", immutable=False)
        write_text(stderr_path, "", immutable=False)

        baselines: Dict[str, object] = {}
        suite_results: List[SuiteResult] = []
        try:
            for baseline_id in run_spec.baselines:
                progress.emit("baseline_init", f"Initializing baseline: {baseline_id}")
                baselines[baseline_id] = self.registry.instantiate_baseline(baseline_id, run_spec)

            for suite_id in run_spec.suites:
                progress.emit("suite_start", f"Running suite: {suite_id}")
                suite = self.registry.instantiate_suite(suite_id, run_spec)
                suite_cfg = self.registry.resolve_suite_config(suite_id, run_spec)
                run_ctx = RunContext(
                    run_spec=run_spec,
                    run_dir=run_dir,
                    baselines=baselines,
                    suite_config=suite_cfg,
                    dataset_config=self.registry.datasets_cfg,
                    progress_sink=progress,
                    runtime_config={"benchmark_root": str(self.root)},
                )
                result = suite.run(run_ctx)
                suite_results.append(result)
                progress.emit(
                    "suite_end",
                    f"Completed suite: {suite_id}",
                    {"baseline_metrics": result.baseline_metrics},
                )

            aggregate = self.scoring.aggregate(suite_results)
            rank_estimates = [
                self.galileo_rank.estimate(score, reference_source="galileo")
                for score in aggregate
            ]
            payload_results = {
                "run_id": run_spec.run_id,
                "created_at": run_spec.created_at,
                "suite_results": [sr.to_dict() for sr in suite_results],
            }
            payload_metrics = {
                "run_id": run_spec.run_id,
                "schema_version": getattr(run_spec, "schema_version", "2.0"),
                "agent": getattr(run_spec, "agent", "nexus-1"),
                "suite_weights": dict(self.scoring.suite_weights),
                "aggregate_scores": [a.to_dict() for a in aggregate],
                "rank_estimates": [r.to_dict() for r in rank_estimates],
            }
            write_json(results_path, payload_results, immutable=True)
            write_json(metrics_path, payload_metrics, immutable=True)
            write_json(
                status_path,
                {"run_id": run_spec.run_id, "status": "completed", "updated_at": utc_now_iso()},
                immutable=False,
            )
            self.run_store.update_artifacts(
                run_spec.run_id,
                str(spec_path),
                str(results_path),
                str(metrics_path),
            )
            self.run_store.update_status(run_spec.run_id, "completed")
            progress.emit("run_end", f"Run completed: {run_spec.run_id}")
            return payload_metrics
        except Exception as exc:
            tb = traceback.format_exc()
            write_text(stderr_path, tb, immutable=False)
            write_json(
                status_path,
                {
                    "run_id": run_spec.run_id,
                    "status": "failed",
                    "updated_at": utc_now_iso(),
                    "error": str(exc),
                },
                immutable=False,
            )
            self.run_store.update_status(run_spec.run_id, "failed", error=str(exc))
            progress.emit("run_error", f"Run failed: {run_spec.run_id}", {"error": str(exc)})
            raise
        finally:
            for baseline in baselines.values():
                try:
                    baseline.close()
                except Exception:
                    pass
