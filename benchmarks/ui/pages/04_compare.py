from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmarks.core.artifacts import read_json
from benchmarks.ui.components.charts import score_table
from benchmarks.ui.components.run_actions import get_executor


st.title("Compare Runs")
ex = get_executor()
runs = ex.run_store.list(include_deleted=False, limit=300)
run_ids = [r.run_id for r in runs]

run_a = st.selectbox("Run A", options=run_ids, index=0 if run_ids else None)
run_b = st.selectbox("Run B", options=run_ids, index=1 if len(run_ids) > 1 else 0 if run_ids else None)

if run_a and run_b:
    row_a = ex.run_store.get(run_a)
    row_b = ex.run_store.get(run_b)
    if not row_a or not row_b:
        st.warning("Run records missing.")
    else:
        metrics_a = read_json(Path(row_a.metrics_path)) if row_a.metrics_path else {}
        metrics_b = read_json(Path(row_b.metrics_path)) if row_b.metrics_path else {}
        sa = {x["baseline_id"]: x["overall_score"] for x in metrics_a.get("aggregate_scores", [])}
        sb = {x["baseline_id"]: x["overall_score"] for x in metrics_b.get("aggregate_scores", [])}
        rows = []
        for baseline_id in sorted(set(sa.keys()) | set(sb.keys())):
            a = sa.get(baseline_id)
            b = sb.get(baseline_id)
            rows.append(
                {
                    "baseline_id": baseline_id,
                    "run_a": a,
                    "run_b": b,
                    "delta": (b - a) if (a is not None and b is not None) else None,
                }
            )
        score_table(rows)

