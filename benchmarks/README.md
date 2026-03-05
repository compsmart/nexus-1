# Benchmark Framework

This package provides a plugin-style benchmark system for the AGI agent.

## Key Principles

- Suites are isolated plugins under `benchmarks/suites/<suite_id>/suite.py`.
- Baselines are isolated adapters under `benchmarks/baselines/`.
- Scoring is config-driven (`benchmarks/config/scoring.yaml`).
- Run artifacts are immutable and stored per `run_id`.
- History is soft-deletable and recoverable via SQLite metadata.

## Quick Start

```bash
python -m benchmarks.cli.bench run
python -m benchmarks.cli.bench list
python -m benchmarks.cli.bench status <run_id>
streamlit run benchmarks/ui/app.py
```

## Main Entry Points

- CLI: `benchmarks/cli/bench.py`
- UI: `benchmarks/ui/app.py`
- Core orchestrator: `benchmarks/core/executor.py`
- Run DB: `benchmarks/core/run_store.py`

