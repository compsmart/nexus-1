from __future__ import annotations

from typing import List

import streamlit as st


def run_table(rows: List[dict]) -> None:
    if not rows:
        st.info("No runs found.")
        return
    st.dataframe(rows, use_container_width=True)

