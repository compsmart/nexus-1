from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmarks.core.artifacts import read_json
from benchmarks.ui.components.charts import score_table
from benchmarks.ui.components.run_actions import get_executor


st.title("Unofficial Rank View")
st.caption("Rank is estimated from local score vs latest reference snapshot.")

ex = get_executor()
runs = [r for r in ex.run_store.list(include_deleted=False, limit=300) if r.metrics_path]
run_id = st.selectbox("Run", options=[r.run_id for r in runs] or [""])

if run_id:
    row = ex.run_store.get(run_id)
    if row and row.metrics_path:
        metrics = read_json(Path(row.metrics_path))
        rank_rows = metrics.get("rank_estimates", [])
        score_table(rank_rows)
    else:
        st.info("No metrics available for selected run.")

