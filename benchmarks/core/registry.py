from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .types import RunSpec


class BenchmarkRegistry:
    def __init__(self, benchmark_root: Path):
        self.root = benchmark_root
        self.config_dir = self.root / "config"
        self.suites_cfg = self._load_yaml("suites.yaml")
        self.baselines_cfg = self._load_yaml("baselines.yaml")
        self.scoring_cfg = self._load_yaml("scoring.yaml")
        self.datasets_cfg = self._load_yaml("datasets.yaml")

    def _load_yaml(self, name: str) -> Dict[str, Any]:
        path = self.config_dir / name
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_class(module_name: str, class_name: str):
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)

    def default_profile(self) -> str:
        return self.suites_cfg.get("defaults", {}).get("profile", "quality_first")

    def default_primary_suite(self) -> str:
        return self.suites_cfg.get("defaults", {}).get("primary_suite", "memory_recall")

    def default_suites(self) -> List[str]:
        order = self.suites_cfg.get("defaults", {}).get("suite_order", [])
        suites = self.suites_cfg.get("suites", {})
        return [sid for sid in order if suites.get(sid, {}).get("enabled", True)]

    def default_baselines(self) -> List[str]:
        order = self.baselines_cfg.get("defaults", {}).get("baseline_order", [])
        baselines = self.baselines_cfg.get("baselines", {})
        return [bid for bid in order if baselines.get(bid, {}).get("enabled", True)]

    def resolve_suite_config(self, suite_id: str, run_spec: RunSpec) -> Dict[str, Any]:
        suites = self.suites_cfg.get("suites", {})
        if suite_id not in suites:
            raise KeyError(f"Unknown suite id: {suite_id}")
        cfg = deepcopy(suites[suite_id].get("config", {}))
        profile = run_spec.profile or self.default_profile()
        profile_cfg = self.suites_cfg.get("profiles", {}).get(profile, {})
        profile_overrides = profile_cfg.get("suite_overrides", {}).get(suite_id, {})
        cfg.update(profile_overrides)
        cfg.update((run_spec.suite_overrides or {}).get(suite_id, {}))
        return cfg

    def instantiate_suite(self, suite_id: str, run_spec: RunSpec):
        suite_row = self.suites_cfg.get("suites", {}).get(suite_id)
        if not suite_row:
            raise KeyError(f"Unknown suite id: {suite_id}")
        cls = self._load_class(suite_row["module"], suite_row["class"])
        cfg = self.resolve_suite_config(suite_id, run_spec)
        return cls(suite_id=suite_id, config=cfg)

    def instantiate_baseline(self, baseline_id: str, run_spec: RunSpec):
        baseline_row = self.baselines_cfg.get("baselines", {}).get(baseline_id)
        if not baseline_row:
            raise KeyError(f"Unknown baseline id: {baseline_id}")
        cls = self._load_class(baseline_row["module"], baseline_row["class"])
        cfg = deepcopy(baseline_row.get("config", {}))
        cfg["benchmark_root"] = str(self.root)
        return cls(baseline_id=baseline_id, run_spec=run_spec, config=cfg)

