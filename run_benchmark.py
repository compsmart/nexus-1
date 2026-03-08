#!/usr/bin/env python3
"""Run NEXUS-1 benchmarks using the shared benchmark runner.

Usage:
    python run_benchmark.py                           # Standard run
    python run_benchmark.py --profile smoke           # Fast dev check
    python run_benchmark.py --profile quality_first   # Full validation
    python run_benchmark.py --suites memory_recall    # Single suite
"""

import sys
from pathlib import Path

# Ensure nexus-1 root is importable
_NEXUS1_DIR = Path(__file__).resolve().parent
if str(_NEXUS1_DIR) not in sys.path:
    sys.path.insert(0, str(_NEXUS1_DIR))

from benchmarks.adapter import Nexus1Adapter

try:
    from shared_benchmarks.runner import run_benchmark
except ImportError:
    # Fallback: shared_benchmarks may be at a sibling path
    _SHARED = Path(__file__).resolve().parent.parent / "shared_benchmarks"
    if _SHARED.exists() and str(_SHARED.parent) not in sys.path:
        sys.path.insert(0, str(_SHARED.parent))
    from shared_benchmarks.runner import run_benchmark


if __name__ == "__main__":
    run_benchmark(adapter_class=Nexus1Adapter)
