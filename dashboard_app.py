"""
Streamlit dashboard for the
University Bulk Order & Predictive Procurement Analytics System.

Run with:
    streamlit run dashboard_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl_pipeline import load_feature_table
from feature_engine import train_model, apply_predictions

st.set_page_config(
    page_title="University Bulk Order & Predictive Procurement Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Overall background: darker navy gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1b2b4a 0%, #050b18 55%, #020309 100%);
    }
    /* Header bar: slightly darker blue */
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #15294b, #274a7b);
        color: #e5e7eb;
    }
    /* Target the native Streamlit border-containers in the sidebar */
    [data-testid="stSidebar"] [data-testid="stElementContainer"] [data-testid="stVerticalBlockBorder"] {
        background: linear-gradient(180deg, #1b3358, #091426);
        border-radius: 12px;
        padding: 16px 12px 20px 12px;
        border: 1px solid #304c7a !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        color: #e5e7eb;
        margin-bottom: 15px;
    }
    .filter-panel h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #e5e7eb;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-align: center;
    }
    .filter-panel label {
        color: #e5e7eb !important;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .filter-panel [data-baseweb="select"] > div {
        background-color: #15294b;
        color: #e5e7eb;
        border-radius: 6px;
        border: 1px solid #4b6b9c;
    }
    .filter-panel [data-baseweb="select"] svg {
        fill: #e5e7eb;
    }
    .filter-panel input {
        background-color: #15294b;
        color: #e5e7eb;
        border-radius: 6px;
        border: 1px solid #4b6b9c;
    }
    .glass-panel {
        background: rgba(15, 23, 42, 0.92);
        border-radius: 18px;
        padding: 0.6rem 0.9rem 0.9rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    .chart-title-bar {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 20px;
        padding: 10px 15px;
        margin-bottom: 25px;
        text-align: center;
        color: #f1f5f9;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def get_raw_data() -> pd.DataFrame:
    return load_feature_table()

@st.cache_resource(show_spinner=False)
def get_trained_model(df: pd.DataFrame):
    return train_model(df)

def kpi_card(label: str, value: str, help_text: str | None = None):
    with st.container():
        st.markdown(
            f"""
            <div style="
                padding: 0.9rem 1.1rem;
                margin-bottom: 1rem;
                border-radius: 0.7rem;
                background: linear-gradient(135deg, #d6ebfb, #c0def4);
                border: 1px solid rgba(148, 163, 184, 0.4);
            ">
                <div style="
                    background: rgba(15, 23, 42, 0.1);
                    border: 1px solid rgba(15, 23, 42, 0.15);
                    border-radius: 20px;
                    padding: 4px 12px;
                    margin-bottom: 10px;
                    display: inline-block;
                    font-size: 0.75rem;
                    color: #4b5563;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    font-weight: 600;
                ">
                    {label}
                </div>
                <div style="font-size: 1.6rem; font-weight: 600; color: #0f172a;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if help_text:
            st.caption(help_text)

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.container(border=True):
        st.markdown('<div class="chart-title-bar" style="font-size: 0.95rem; padding: 6px 10px; margin-bottom: 15px;">Student Segments</div>', unsafe_allow_html=True)
        college = st.selectbox("College", options=["All"] + sorted(df["College"].unique().tolist()), key="filter_college")
        year    = st.selectbox("Year",    options=["All"] + sorted(df["Year"].unique().tolist()),    key="filter_year")
        dept    = st.selectbox("Department", options=["All"] + sorted(df["Dept_Code"].unique().tolist()), key="filter_dept")
        sem     = st.selectbox("Semester", options=["All"] + sorted(df["Semester"].unique().tolist()), key="filter_sem")
        student_type = st.selectbox("Student Type", options=["All"] + sorted(df["Student_Type"].unique().tolist()), key="filter_student_type")

        mask = pd.Series(True, index=df.index)
        if college != "All":
            mask &= df["College"] == college
        if year != "All":
            mask &= df["Year"] == year
        if dept != "All":
            mask &= df["Dept_Code"] == dept
        if sem != "All":
            mask &= df["Semester"] == sem
        if student_type != "All":
            mask &= df["Student_Type"] == student_type

        return df[mask].copy()

def render_accuracy_gauge(acc: float):
    with st.container(border=True):
        st.markdown('<div class="chart-title-bar" style="font-size: 0.95rem; padding: 6px 10px; margin-bottom: 15px;">Model Accuracy</div>', unsafe_allow_html=True)
        
        # High-Impact Speedometer Calculation
        value = acc * 100
        angle = 180 - (value * 1.8) 
        rad = np.radians(angle)
        
        # Center alignment
        x_center, y_center = 0.5, 0.42
        radius = 0.38
        x = x_center + radius * np.cos(rad)
        y = y_center + radius * np.sin(rad)

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'font': {'size': 40, 'color': "#ffffff", 'family': 'Inter'}, 'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#ffffff", 'nticks': 10},
                'bar': {'color': "rgba(0,0,0,0)"}, 
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 60], 'color': '#dc2626'},   # Vibrant Red
                    {'range': [60, 85], 'color': '#f59e0b'},  # Vibrant Orange/Amber
                    {'range': [85, 100], 'color': '#22c55e'}  # Vibrant Green
                ],
            }
        ))

        # Vibrant Glowing Needle
        fig.add_shape(type='line', x0=x_center, y0=y_center, x1=x, y1=y, line=dict(color='#00d4ff', width=6))
        # Center Hub (Premium look)
        fig.add_shape(type='circle', x0=x_center-0.04, y0=y_center-0.04, x1=x_center+0.04, y1=y_center+0.04, fillcolor='#1e293b', line_color='#00d4ff', line_width=2)
        fig.add_shape(type='circle', x0=x_center-0.015, y0=y_center-0.015, x1=x_center+0.015, y1=y_center+0.015, fillcolor='#ffffff', line_color='#ffffff')

        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#ffffff", 'family': "Inter"}
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<p style='margin-top: -20px; font-size: 0.8rem; color: #e5e7eb; text-align: center;'>The Random Forest model is predicting student behavior with <b>{acc*100:.1f}%</b> accuracy via high-friction signal points.</p>",
            unsafe_allow_html=True
        )

def render_header():
    st.markdown(
        """
        <div class="chart-title-bar" style="height: auto; padding: 20px; text-align: center; margin-bottom: 25px;">
            <div style="font-size: 1.6rem; font-weight: 700;">University Bulk Order & Predictive Procurement Analytics</div>
            <div style="font-size: 0.9rem; font-weight: 400; color: #94a3b8; margin-top: 5px;">
                Machine-learning driven demand forecasting and risk segmentation with Investor ROI Modeling.
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def render_top_kpis(df: pd.DataFrame):
    if df.empty:
        st.warning("No data for current filter selection.")
        return
    students_buying = int(df["Predicted_Purchase_Prob"].sum())
    total_spend = df["Projected_Spend"].sum()
    total = len(df)
    pt_count = (df["Student_Type"] == "Part-Time").sum()
    ft_count = (df["Student_Type"] == "Full-Time").sum()
    pt_pct = pt_count / total * 100 if total else 0
    ft_pct = ft_count / total * 100 if total else 0
    buying     = int(df["Predicted_Purchase_Prob"].sum())
    not_buying = total - buying
    ratio_str  = f"{buying:,} : {not_buying:,}"

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Students Predicted to Buy", f"{students_buying:,}", f"Projected Spend: ${total_spend:,.0f}")
    with c2: kpi_card("Part-Time vs Full-Time", f"{pt_pct:.0f}% / {ft_pct:.0f}%", f"{pt_count:,} part-time · {ft_count:,} full-time")
    with c3: kpi_card("Buy vs Not-Buy Ratio", ratio_str, "Predicted opt-in : predicted opt-out")

def render_book_quantities(df: pd.DataFrame):
    st.markdown('<div class="chart-title-bar">Quantity of Each Book (Volume Forecast)</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("No data available.")
        return
    agg = df.groupby("Title")["Predicted_Demand_Units"].sum().reset_index()
    agg = agg.sort_values(by="Predicted_Demand_Units", ascending=False).head(20)
    fig = px.bar(agg, x="Title", y="Predicted_Demand_Units", color="Predicted_Demand_Units", color_continuous_scale="Viridis")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="", yaxis_title="Predicted Units", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"), coloraxis_showscale=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def render_feature_importance(fi: pd.DataFrame):
    st.markdown('<div class="chart-title-bar">Feature Importance (Model Explainability)</div>', unsafe_allow_html=True)
    fig = px.bar(fi.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Relative Importance", yaxis_title="Feature", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

def render_high_friction_titles(df: pd.DataFrame):
    st.markdown('<div class="chart-title-bar">Top 10 High-Friction Titles (Negotiation Targets)</div>', unsafe_allow_html=True)
    if df.empty: return
    agg = df.groupby("Title")["Opt_Out_Probability"].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(agg, x="Opt_Out_Probability", y="Title", orientation="h", color="Opt_Out_Probability", color_continuous_scale="Reds", text_auto='.1%')
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Avg Opt-Out Risk (%)", yaxis_title="", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f9fafb"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

def render_word_cloud(df: pd.DataFrame):
    st.markdown('<div class="chart-title-bar">Book Names Word Cloud</div>', unsafe_allow_html=True)
    if df.empty: return
    agg = df.groupby("Title")["Predicted_Demand_Units"].sum()
    freq_dict = agg.to_dict()
    if not freq_dict: return
    wc = WordCloud(width=1200, height=400, background_color=None, mode="RGBA", colormap="Blues").generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

def main():
    with st.spinner("Processing 7.5 Million Records (100% Signal)..."):
        raw_df = get_raw_data()
        clf, fi_df, features, acc = get_trained_model(raw_df)
    render_header()
    filters_col, main_col = st.columns([0.9, 4.1])
    with filters_col:
        filtered_df_base = render_filters(raw_df)
        render_accuracy_gauge(acc)
    filtered_df = apply_predictions(filtered_df_base, clf, features, discount_pct=0)
    with main_col:
        render_top_kpis(filtered_df)
        row2_left, row2_right = st.columns(2)
        with row2_left:
            render_feature_importance(fi_df)
        with row2_right:
            render_high_friction_titles(filtered_df)
        
        render_book_quantities(filtered_df)
        render_word_cloud(filtered_df)

if __name__ == "__main__":
    main()
