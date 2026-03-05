from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmarks.core.artifacts import read_json
from benchmarks.ui.components.run_actions import get_executor


st.title("Reference Snapshots")
ex = get_executor()
leaderboard = st.selectbox("Leaderboard", options=["galileo", "open_llm"], index=0)
snap_dir = ex.root / "reference_data" / "leaderboards" / leaderboard / "snapshots"
snapshots = sorted(snap_dir.glob("*.json"))

if not snapshots:
    st.info("No snapshots available.")
else:
    selected = st.selectbox("Snapshot", options=[p.name for p in snapshots], index=len(snapshots) - 1)
    path = snap_dir / selected
    st.json(read_json(path))
