"""
AgentSC-Bench Streamlit dashboard.

Launch:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AgentSC-Bench", layout="wide", page_icon="📦")

RESULTS = ROOT / "results"


def find_summaries():
    if not RESULTS.exists():
        return []
    return sorted(RESULTS.rglob("*_summary.csv"))


def find_timeseries():
    if not RESULTS.exists():
        return []
    return sorted(RESULTS.rglob("*_timeseries.csv"))


st.title("AgentSC-Bench")
st.caption("Open Benchmark for Reliable Agentic AI in Multi-Echelon Supply Chains")

pages = [
    "Overview",
    "Experiment Explorer",
    "Inventory & Orders",
    "Cost Comparison",
    "Bullwhip Analysis",
    "Reliability",
    "Research Protocol",
]
page = st.sidebar.radio("Pages", pages)

summaries = find_summaries()
timeseries = find_timeseries()

if page == "Overview":
    st.header("Overview")
    st.markdown("""
    **AgentSC-Bench** systematically compares classical inventory policies and
    LLM-based agent architectures on a multi-echelon Beer-Game-style simulator.

    Key measured dimensions:
    - Efficiency (total cost, service level)
    - Classical bullwhip & Agent Bullwhip Index
    - Decision reliability under repeated identical states
    - Disruption recovery (TTR, resilience score)
    - Communication volume & autonomy (later phases)
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Summary files found", len(summaries))
    col2.metric("Timeseries files found", len(timeseries))
    col3.metric("Scenarios available", 9)
    st.info("Run experiments first: `python -m experiments.runner --config configs/baseline.yaml`")

elif page == "Experiment Explorer":
    st.header("Experiment Explorer")
    if not summaries:
        st.warning("No results yet. Run an experiment.")
    else:
        chosen = st.selectbox("Select summary CSV", [str(p.relative_to(ROOT)) for p in summaries])
        df = pd.read_csv(ROOT / chosen)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=Path(chosen).name)

elif page == "Inventory & Orders":
    st.header("Inventory & Orders over Time")
    if not timeseries:
        st.warning("No timeseries results.")
    else:
        chosen = st.selectbox("Timeseries file", [str(p.relative_to(ROOT)) for p in timeseries])
        ts = pd.read_csv(ROOT / chosen)
        exp_ids = ts["experiment_id"].unique().tolist()
        eid = st.selectbox("Experiment ID", exp_ids)
        sub = ts[ts["experiment_id"] == eid]
        metric = st.radio("Metric", ["inventory", "order", "backlog"], horizontal=True)
        pivot = sub.groupby(["period", "echelon"])[metric].mean().unstack()
        st.line_chart(pivot)

elif page == "Cost Comparison":
    st.header("Cost Comparison")
    if not summaries:
        st.warning("No results.")
    else:
        frames = [pd.read_csv(p) for p in summaries]
        df = pd.concat(frames, ignore_index=True)
        if "total_cost" in df.columns:
            chart = df.groupby("architecture")["total_cost"].mean().sort_values()
            st.bar_chart(chart)
            st.dataframe(df.groupby(["architecture", "scenario"])[["total_cost", "mean_service_level"]].mean().round(2))
        else:
            st.write(df.head())

elif page == "Bullwhip Analysis":
    st.header("Bullwhip Analysis")
    st.markdown("""
    **Classical bullwhip ratio** = Var(Order_tier) / Var(CustomerDemand)

    **Agent Bullwhip Index** (project-defined) measures amplification of
    decision variance across repeated runs (inspired by Long et al. 2026).
    """)
    if summaries:
        frames = [pd.read_csv(p) for p in summaries]
        df = pd.concat(frames, ignore_index=True)
        cols = [c for c in df.columns if "bullwhip" in c.lower() or "abi" in c.lower()]
        if cols:
            st.dataframe(df[["architecture", "scenario"] + cols].head(20))
        else:
            st.info("Bullwhip columns not present in current summaries.")

elif page == "Reliability":
    st.header("Reliability Analysis")
    st.markdown("""
    Reliability is measured by presenting the **same** supply-chain state
    repeatedly and quantifying the variance of the agent's order decisions.

    Score (project definition) = 1 − normalised decision coefficient of variation.
    """)
    rel_files = list(RESULTS.rglob("*_reliability.json")) if RESULTS.exists() else []
    if rel_files:
        import json
        chosen = st.selectbox("Reliability result", [str(p.relative_to(ROOT)) for p in rel_files])
        with open(ROOT / chosen) as f:
            data = json.load(f)
        st.json(data)
        if "decisions" in data:
            st.bar_chart(pd.Series(data["decisions"], name="order_quantity"))
    else:
        st.info("Run: `python -m experiments.runner --config configs/reliability.yaml --reliability`")

elif page == "Research Protocol":
    st.header("Research Protocol")
    protocol = ROOT / "docs" / "research_protocol.md"
    if protocol.exists():
        st.markdown(protocol.read_text())
    else:
        st.write("Protocol not found.")

st.sidebar.markdown("---")
st.sidebar.caption("AgentSC-Bench · research benchmark · Apache-2.0")
