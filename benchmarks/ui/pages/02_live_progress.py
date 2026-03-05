from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmarks.ui.components.run_actions import get_executor


st.title("Live Progress")
ex = get_executor()
runs = ex.run_store.list(include_deleted=True, limit=200)

run_id = st.selectbox("Run", options=[r.run_id for r in runs] or [""])
if run_id:
    row = ex.run_store.get(run_id)
    st.json(row.__dict__ if row else {})
    run_dir = ex.root / "runs" / run_id
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    stdout_path = run_dir / "stdout.log"

    if events_path.exists():
        st.subheader("Events")
        st.code(events_path.read_text(encoding="utf-8"), language="json")
    else:
        st.info("No events logged yet.")

    cols = st.columns(2)
    with cols[0]:
        st.subheader("stdout.log")
        st.code(stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "")
    with cols[1]:
        st.subheader("stderr.log")
        st.code(stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "")

