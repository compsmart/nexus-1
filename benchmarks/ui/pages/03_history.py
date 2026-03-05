from __future__ import annotations

import streamlit as st

from benchmarks.ui.components.run_actions import get_executor, restore_run, soft_delete_run
from benchmarks.ui.components.tables import run_table


st.title("Run History")
ex = get_executor()
show_deleted = st.checkbox("Show deleted", value=False)
rows = ex.run_store.list(include_deleted=show_deleted, limit=500)

table_rows = [
    {
        "run_id": r.run_id,
        "name": r.name,
        "status": r.status,
        "created_at": r.created_at,
        "deleted_at": r.deleted_at,
    }
    for r in rows
]
run_table(table_rows)

run_id = st.selectbox("Select run to manage", options=[r.run_id for r in rows] or [""])
col1, col2 = st.columns(2)
if col1.button("Soft Delete", disabled=not run_id):
    soft_delete_run(run_id)
    st.success(f"Soft-deleted {run_id}")
if col2.button("Restore", disabled=not run_id):
    restore_run(run_id)
    st.success(f"Restored {run_id}")

