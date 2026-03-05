from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmarks.core.executor import BenchmarkExecutor


@st.cache_resource
def get_executor() -> BenchmarkExecutor:
    root = Path(__file__).resolve().parents[2]
    return BenchmarkExecutor(root)


def soft_delete_run(run_id: str) -> None:
    ex = get_executor()
    ex.run_store.soft_delete(run_id)


def restore_run(run_id: str) -> None:
    ex = get_executor()
    ex.run_store.restore(run_id)

