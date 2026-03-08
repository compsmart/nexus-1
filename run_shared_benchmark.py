#!/usr/bin/env python3
"""Run nexus-1 benchmarks using the shared benchmark runner.

This uses the standardized shared_benchmarks suites for cross-agent comparison.
For nexus-1's own HuggingFace-based benchmarks, use: python -m benchmarks.cli.bench run

Usage:
    python run_shared_benchmark.py                     # Standard run
    python run_shared_benchmark.py --profile smoke     # Fast dev check
"""

import sys
from pathlib import Path

_NEXUS1_DIR = Path(__file__).resolve().parent
if str(_NEXUS1_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS1_DIR))

from benchmarks.adapter import Nexus1Adapter

try:
    from shared_benchmarks.runner import run_benchmark
except ImportError:
    _SHARED = Path(__file__).resolve().parent.parent / "shared_benchmarks"
    if _SHARED.exists() and str(_SHARED.parent) not in sys.path:
        sys.path.insert(0, str(_SHARED.parent))
    from shared_benchmarks.runner import run_benchmark


if __name__ == "__main__":
    run_benchmark(adapter_class=Nexus1Adapter)
