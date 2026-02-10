"""
Streamlit dashboard for the
University Bulk Order & Predictive Procurement Analytics System.

Run with:
    streamlit run dashboard_app.py
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl_pipeline import load_feature_table
from feature_engine import compute_feature_importance_example


st.set_page_config(
    page_title="University Bulk Order & Predictive Procurement Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling to match the blue analytics dashboard reference
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #274672 0%, #0c1b36 40%, #020617 100%);
    }
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #0c1b36, #274672);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #020617 100%);
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_feature_table()


def kpi_card(label: str, value: str, help_text: str | None = None, key: str | None = None):
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding: 0.75rem 1rem;
                border-radius: 0.6rem;
                background: linear-gradient(135deg, #1e3a8a, #0f172a);
                border: 1px solid rgba(191, 219, 254, 0.35);
            ">
                <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em;">
                    {label}
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #e5e7eb;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")

    term = st.sidebar.selectbox("Term", options=["All"] + sorted(df["Term"].unique().tolist()))
    dept = st.sidebar.selectbox("Department", options=["All"] + sorted(df["Dept_Code"].unique().tolist()))
    publisher = st.sidebar.selectbox(
        "Publisher", options=["All"] + sorted(df["Publisher"].unique().tolist())
    )
    student_type = st.sidebar.selectbox(
        "Student Type", options=["All"] + sorted(df["Student_Type"].unique().tolist())
    )
    fmt = st.sidebar.selectbox(
        "Format", options=["All"] + sorted(df["Format"].unique().tolist())
    )

    mask = pd.Series(True, index=df.index)
    if term != "All":
        mask &= df["Term"] == term
    if dept != "All":
        mask &= df["Dept_Code"] == dept
    if publisher != "All":
        mask &= df["Publisher"] == publisher
    if student_type != "All":
        mask &= df["Student_Type"] == student_type
    if fmt != "All":
        mask &= df["Format"] == fmt

    return df[mask].copy()


def render_header():
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            "### University Bulk Order & Predictive Procurement Analytics",
        )
        st.caption(
            "Machine-learning driven demand forecasting and risk segmentation for university textbook procurement."
        )
    with right:
        st.markdown(
            """
            <div style="text-align: right; font-size: 0.8rem; color: #9ca3af;">
                <div>Today</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_kpis(df: pd.DataFrame):
    total_pred_units = df["Predicted_Demand_Units"].sum()
    total_spend = df["Projected_Spend"].sum()

    digital_share = (
        (df["Format"] == "Digital").sum() / len(df) if len(df) else 0
    )
    physical_share = 1 - digital_share

    high_risk_rate = (df["Opt_Out_Probability"] > 0.6).mean() if len(df) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Predicted Demand", f"{int(total_pred_units):,} Units")
    with c2:
        kpi_card("Total Projected Spend", f"${total_spend:,.1f} M", "Millions equivalent")
    with c3:
        kpi_card(
            "Digital vs Physical",
            f"{digital_share*100:.0f}% / {physical_share*100:.0f}%",
            "Share of predicted units",
        )
    with c4:
        kpi_card(
            "High-Risk Opt-Out Rate",
            f"{high_risk_rate*100:.0f}%",
            "Share of records with Opt-Out Probability > 60%",
        )


def render_price_sensitivity(df: pd.DataFrame):
    st.markdown("#### Price Sensitivity & Opt-Out Threshold")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    threshold = st.slider(
        "Rental-to-Retail Ratio Opt-Out Threshold",
        min_value=0.4,
        max_value=1.0,
        value=0.7,
        step=0.02,
    )

    fig = px.scatter(
        df,
        x="Rental_to_Retail_Ratio",
        y="Opt_Out_Probability",
        color="Dept_Code",
        hover_data=["Publisher", "Student_Type", "Format"],
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Rental-to-Retail Ratio",
        yaxis_title="Opt-Out Probability",
        plot_bgcolor="#0b1630",
        paper_bgcolor="#0b1630",
        font=dict(color="#e5e7eb"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance():
    st.markdown("#### Feature Importance (Model Explainability)")
    fi = compute_feature_importance_example()

    fig = px.bar(
        fi.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Relative Importance",
        yaxis_title="Feature",
        plot_bgcolor="#0b1630",
        paper_bgcolor="#0b1630",
        font=dict(color="#e5e7eb"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_format_preference(df: pd.DataFrame):
    st.markdown("#### Format Preference by Segment")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    by_student_type = (
        df.groupby(["Student_Type", "Format"])["Predicted_Demand_Units"].sum().reset_index()
    )

    fig = px.bar(
        by_student_type,
        x="Student_Type",
        y="Predicted_Demand_Units",
        color="Format",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Student Type",
        yaxis_title="Predicted Units",
        plot_bgcolor="#0b1630",
        paper_bgcolor="#0b1630",
        font=dict(color="#e5e7eb"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_funding_flow(df: pd.DataFrame):
    st.markdown("#### Funding Source Planning & Strategy")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    # Simple synthetic funding sources
    sources = ["Financial Aid", "Self-Pay", "Scholarship"]
    outcomes = ["Opt-In", "Opt-Out"]

    rng = np.random.default_rng(0)
    values = rng.integers(50, 200, size=len(sources) * len(outcomes))
    values = values.astype(float)

    labels = sources + outcomes
    source_indices = []
    target_indices = []
    for i, _ in enumerate(sources):
        for j, _ in enumerate(outcomes):
            source_indices.append(i)
            target_indices.append(len(sources) + j)

    link = dict(source=source_indices, target=target_indices, value=values)

    node = dict(
        label=labels,
        pad=20,
        thickness=14,
        color=["#60a5fa", "#34d399", "#fbbf24", "#22c55e", "#f97316"],
    )

    fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#e5e7eb"),
        plot_bgcolor="#0b1630",
        paper_bgcolor="#0b1630",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_by_department(df: pd.DataFrame):
    st.markdown("#### Procurement Risk: Top 5 High-Risk Departments")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    agg = (
        df.groupby("Dept_Code")["Opt_Out_Probability"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    fig = px.bar(
        agg,
        x="Dept_Code",
        y="Opt_Out_Probability",
        color="Opt_Out_Probability",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Department",
        yaxis_title="Avg Opt-Out Probability",
        plot_bgcolor="#0b1630",
        paper_bgcolor="#0b1630",
        font=dict(color="#e5e7eb"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recommendations(df: pd.DataFrame):
    st.markdown("#### Recommended Actions")

    if df.empty:
        st.info("No data available for current filter selection.")
        return

    high_risk = df[df["Opt_Out_Probability"] > 0.6]
    risky_depts = (
        high_risk.groupby("Dept_Code")["Opt_Out_Probability"].mean().sort_values(ascending=False)
    )
    top_depts = ", ".join(risky_depts.head(3).index.tolist()) or "N/A"

    bullet_points = [
        f"Prioritize **bundle price negotiation** with key publishers in high-risk departments: {top_depts}.",
        "Increase **digital access options** for high commuter-friction and part-time student segments.",
        "Review **required vs recommended** status for titles with extreme rental-to-retail arbitrage.",
        "Share **early adoption reports** with department chairs to align on realistic order quantities.",
    ]

    st.markdown(
        "\n\n".join([f"- {bp}" for bp in bullet_points]),
    )


def main():
    df = get_data()

    render_header()
    st.markdown("---")

    render_top_kpis(df)

    st.markdown("### Feature Engineering Pipeline & Key Indicators")
    upper_left, upper_mid, upper_right = st.columns([1.2, 1.2, 1.0])

    with upper_left:
        render_price_sensitivity(df)
    with upper_mid:
        render_feature_importance()
    with upper_right:
        render_format_preference(df)

    st.markdown("### Procurement Planning & Strategy")
    lower_left, lower_mid, lower_right = st.columns([1.4, 1.0, 1.0])
    with lower_left:
        render_funding_flow(df)
    with lower_mid:
        render_risk_by_department(df)
    with lower_right:
        render_recommendations(df)


if __name__ == "__main__":
    main()

