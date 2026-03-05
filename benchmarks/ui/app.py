from __future__ import annotations

import streamlit as st

from benchmarks.ui.components.run_actions import get_executor


st.set_page_config(page_title="Agent Benchmarks", page_icon=":bar_chart:", layout="wide")

st.title("Agent Benchmark Dashboard")
st.caption("Run, track, compare, and manage benchmark evaluations for AMM/RAG/LLM baselines.")

ex = get_executor()
runs = ex.run_store.list(include_deleted=True, limit=500)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Runs", len(runs))
col2.metric("Completed", sum(1 for r in runs if r.status == "completed"))
col3.metric("Failed", sum(1 for r in runs if r.status == "failed"))
col4.metric("Deleted", sum(1 for r in runs if r.deleted_at))

st.markdown(
    """
Use the left sidebar pages to:
- Start benchmark runs
- Monitor live progress and logs
- Compare historical runs
- Estimate unofficial rank versus reference snapshots
- Soft-delete and restore run history
"""
)

