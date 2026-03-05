from __future__ import annotations

from typing import Dict, List

import streamlit as st


def score_table(rows: List[Dict]) -> None:
    if not rows:
        st.info("No scores available.")
        return
    st.dataframe(rows, use_container_width=True)


def trend_chart(points: List[Dict]) -> None:
    if not points:
        st.info("No trend data available.")
        return
    st.line_chart(points, x="created_at", y="overall_score", color="baseline_id")

