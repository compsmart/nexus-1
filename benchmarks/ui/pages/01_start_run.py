from __future__ import annotations

import streamlit as st

from benchmarks.ui.components.run_actions import get_executor


st.title("Start Benchmark Run")

ex = get_executor()
registry = ex.registry

with st.form("start_run_form"):
    name = st.text_input("Run Name", value="benchmark-run")
    profile = st.selectbox("Profile", options=list(registry.suites_cfg.get("profiles", {}).keys()), index=0)
    model = st.text_input("Model", value="microsoft/Phi-3.5-mini-instruct")
    use_4bit = st.checkbox("Use 4-bit quantization", value=True)
    max_new_tokens = st.number_input("Max new tokens", min_value=16, max_value=512, value=128, step=8)
    suites = st.multiselect("Suites", options=registry.default_suites(), default=registry.default_suites())
    baselines = st.multiselect("Baselines", options=registry.default_baselines(), default=registry.default_baselines())
    submitted = st.form_submit_button("Start Run")

if submitted:
    spec = ex.build_default_run_spec(
        name=name,
        suites=suites,
        baselines=baselines,
        profile=profile,
        model_name=model,
        use_4bit=use_4bit,
    )
    spec.max_new_tokens = int(max_new_tokens)
    try:
        ex.run(spec)
        st.success(f"Run completed: {spec.run_id}")
    except Exception as exc:
        st.error(f"Run failed: {exc}")

