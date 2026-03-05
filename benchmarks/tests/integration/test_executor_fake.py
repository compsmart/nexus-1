from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.core.executor import BenchmarkExecutor


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)


def test_executor_with_fake_plugins(tmp_path: Path):
    root = tmp_path / "benchmarks"
    cfg = root / "config"
    _write(
        cfg / "suites.yaml",
        {
            "defaults": {"profile": "quality_first", "primary_suite": "fake", "suite_order": ["fake"]},
            "profiles": {"quality_first": {"suite_overrides": {}}},
            "suites": {
                "fake": {
                    "module": "benchmarks.tests.fixtures.fake_plugins",
                    "class": "FakeSuite",
                    "enabled": True,
                    "config": {},
                }
            },
        },
    )
    _write(
        cfg / "baselines.yaml",
        {
            "defaults": {"baseline_order": ["fake_a", "fake_b"]},
            "baselines": {
                "fake_a": {
                    "module": "benchmarks.tests.fixtures.fake_plugins",
                    "class": "FakeBaseline",
                    "enabled": True,
                    "config": {},
                },
                "fake_b": {
                    "module": "benchmarks.tests.fixtures.fake_plugins",
                    "class": "FakeBaseline",
                    "enabled": True,
                    "config": {},
                },
            },
        },
    )
    _write(
        cfg / "scoring.yaml",
        {
            "defaults": {
                "suite_weights": {"fake": 1.0},
                "metric_key_by_suite": {"fake": "exact_match"},
                "normalize_suite_weights": True,
            }
        },
    )
    _write(cfg / "datasets.yaml", {"datasets": {}})

    ex = BenchmarkExecutor(root)
    spec = ex.build_default_run_spec(name="fake-run")
    result = ex.run(spec)
    scores = result.get("aggregate_scores", [])
    assert len(scores) == 2
    assert all(abs(s["overall_score"] - 1.0) < 1e-9 for s in scores)

