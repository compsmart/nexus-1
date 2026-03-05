from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.core.registry import BenchmarkRegistry
from benchmarks.core.types import RunSpec, utc_now_iso


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)


def test_registry_loads_plugins(tmp_path: Path):
    root = tmp_path / "benchmarks"
    config = root / "config"
    _write(
        config / "suites.yaml",
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
        config / "baselines.yaml",
        {
            "defaults": {"baseline_order": ["fake"]},
            "baselines": {
                "fake": {
                    "module": "benchmarks.tests.fixtures.fake_plugins",
                    "class": "FakeBaseline",
                    "enabled": True,
                    "config": {},
                }
            },
        },
    )
    _write(config / "scoring.yaml", {"defaults": {"suite_weights": {"fake": 1.0}}})
    _write(config / "datasets.yaml", {"datasets": {}})

    reg = BenchmarkRegistry(root)
    spec = RunSpec(run_id="x", name="x", created_at=utc_now_iso())
    suite = reg.instantiate_suite("fake", spec)
    baseline = reg.instantiate_baseline("fake", spec)
    assert suite is not None
    assert baseline is not None

