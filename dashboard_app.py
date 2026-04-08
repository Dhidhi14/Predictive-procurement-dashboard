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

from etl_pipeline import load_feature_table, load_summary_kpis
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Overall background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1b2b4a 0%, #050b18 55%, #020309 100%);
    }
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, #15294b, #274a7b);
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e36 0%, #050b18 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #1a2e4a;
        color: #e2e8f0;
        border: 1px solid #2d4a6e;
    }

    /* Chart section title bar */
    .chart-title-bar {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(100, 150, 220, 0.25);
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 8px 14px;
        margin-bottom: 14px;
        color: #e2e8f0;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* KPI card base */
    .kpi-card {
        border-radius: 10px;
        padding: 14px 16px 12px 16px;
        min-height: 100px;
        position: relative;
        overflow: hidden;
        margin-bottom: 4px;
    }
    .kpi-card .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
        margin-bottom: 2px;
    }
    .kpi-card .kpi-sub {
        font-size: 0.72rem;
        font-weight: 400;
        opacity: 0.7;
        margin-top: 4px;
        line-height: 1.35;
    }
    .kpi-card .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    /* Teal */
    .kpi-teal { background: linear-gradient(135deg, #0d9488, #0f766e); color: #fff; }
    /* Blue */
    .kpi-blue { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; }
    /* Pink */
    .kpi-pink { background: linear-gradient(135deg, #db2777, #be185d); color: #fff; }
    /* Orange */
    .kpi-orange { background: linear-gradient(135deg, #ea580c, #c2410c); color: #fff; }

    /* Plotly chart containers */
    .plot-container { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _price_bucket(price_series: pd.Series) -> pd.Series:
    """Assign a price-category label to each row's unit price."""
    bins   = [0, 50, 80, 120, np.inf]
    labels = ["<$50", "$50-$80", "$80-$120", ">$120"]
    return pd.cut(price_series, bins=bins, labels=labels, right=True)

_CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        bgcolor="rgba(15,23,42,0.7)",
        bordercolor="rgba(100,150,220,0.3)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

_STACKED_COLORS = ["#14b8a6", "#3b82f6", "#ec4899", "#f97316", "#8b5cf6", "#22c55e"]

# ── data / model cache ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_raw_data() -> pd.DataFrame:
    return load_feature_table()

@st.cache_data(show_spinner=False)
def get_summary_data() -> pd.DataFrame:
    return load_summary_kpis()

@st.cache_resource(show_spinner=False)
def get_trained_model(df: pd.DataFrame):
    return train_model(df)

# ── KPI cards ─────────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, sub: str, css_class: str):
    st.markdown(
        f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_top_kpis(summary_df: pd.DataFrame, sampled_df: pd.DataFrame):
    """
    Renders the top KPIs focusing on ROI and Demand volume.
    """
    if summary_df.empty:
        st.warning("No summary data for current filter selection.")
        return

    # Total Spend and Book Count from pre-computed summary for 100% accuracy
    total_spend = summary_df["Total_Spend"].sum()
    total_books = int(summary_df["Book_Count"].sum())
    
    # Potential Savings Opportunity calculation
    # Logic: We take the sampled Opt-Out risk and apply it to the total spend
    avg_opt_out = sampled_df["Opt_Out_Probability"].mean() if not sampled_df.empty else 0.0
    potential_savings = total_spend * avg_opt_out

    # High Risk Enrollment Count (Opt-Out Prob > 70%)
    high_risk_count = 0
    if not sampled_df.empty:
        # Scale sampled ratio to the total book volume
        risk_ratio = (sampled_df["Opt_Out_Probability"] > 0.70).mean()
        high_risk_count = int(total_books * risk_ratio)

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(
            "Total Book Demand",
            f"{total_books:,}",
            "Units across all departments",
            "kpi-teal",
        )
    with c2:
        kpi_card(
            "Total Projected Spend",
            f"${total_spend/1e6:.2f}M",
            "Based on full 7.5M records",
            "kpi-blue",
        )
    with c3:
        kpi_card(
            "Potential Savings (ROI)",
            f"${potential_savings/1e6:.2f}M",
            f"Expected {avg_opt_out*100:.1f}% Opt-Out",
            "kpi-orange",
        )

# ── sidebar filters ────────────────────────────────────────────────────────────

def render_filters(summary_df: pd.DataFrame) -> dict:
    """Renders filters in sidebar and returns the selected values."""
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title-bar" style="margin-bottom:12px;">📂 Book Filters</div>',
            unsafe_allow_html=True,
        )
        college = st.selectbox("College", ["All"] + sorted(summary_df["College"].unique().tolist()), key="f_college")
        year    = st.selectbox("Year",    ["All"] + sorted(summary_df["Year"].astype(str).unique().tolist()),    key="f_year")
        dept    = st.selectbox("Department", ["All"] + sorted(summary_df["Department"].unique().tolist()), key="f_dept")
        sem     = st.selectbox("Semester", ["All"] + sorted(summary_df["Semester"].unique().tolist()), key="f_sem")
        fmt     = st.selectbox("Format", ["All", "Digital", "Physical"], key="f_format")

    return {
        "College": college,
        "Year": year,
        "Department": dept,
        "Semester": sem,
        "Format": fmt
    }

def apply_filters(df: pd.DataFrame, filters: dict, is_summary: bool = True) -> pd.DataFrame:
    """Applies filters to either summary or sampled dataframe."""
    mask = pd.Series(True, index=df.index)
    
    # Handle column name differences
    dept_col = "Department" if "Department" in df.columns else "Dept_Code"
    year_col = "Year"
    
    if filters["College"] != "All": mask &= df["College"] == filters["College"]
    if filters["Year"] != "All": mask &= df[year_col].astype(str) == filters["Year"]
    if filters["Department"] != "All": mask &= df[dept_col] == filters["Department"]
    if filters["Semester"] != "All": mask &= df["Semester"] == filters["Semester"]
    if filters["Format"] != "All": mask &= df["Format"] == filters["Format"]
    
    return df[mask]
    if fmt     != "All": mask &= df["Format"] == fmt
    if pcat    != "All": mask &= df_tmp["Price_Category"].astype(str) == pcat

    return df[mask].copy()

# ── model accuracy gauge ───────────────────────────────────────────────────────

def render_accuracy_gauge(acc: float):
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title-bar" style="margin-bottom:10px;">🎯 Model Accuracy</div>',
            unsafe_allow_html=True,
        )
        value = acc * 100
        angle = 180 - (value * 1.8)
        rad   = np.radians(angle)
        xc, yc, r = 0.5, 0.42, 0.38
        x = xc + r * np.cos(rad)
        y = yc + r * np.sin(rad)

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            number={"font": {"size": 38, "color": "#ffffff", "family": "Inter"}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#ffffff", "nticks": 10},
                "bar": {"color": "rgba(0,0,0,0)"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60],  "color": "#dc2626"},
                    {"range": [60, 85], "color": "#f59e0b"},
                    {"range": [85, 100],"color": "#22c55e"},
                ],
            },
        ))
        fig.add_shape(type="line", x0=xc, y0=yc, x1=x, y1=y, line=dict(color="#00d4ff", width=6))
        fig.add_shape(type="circle", x0=xc-.04, y0=yc-.04, x1=xc+.04, y1=yc+.04, fillcolor="#1e293b", line_color="#00d4ff", line_width=2)
        fig.add_shape(type="circle", x0=xc-.015, y0=yc-.015, x1=xc+.015, y1=yc+.015, fillcolor="#ffffff", line_color="#ffffff")
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff", "family": "Inter"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<p style='margin-top:-18px;font-size:0.75rem;color:#94a3b8;text-align:center;'>RF Model · <b>{acc*100:.1f}%</b> accuracy</p>",
            unsafe_allow_html=True,
        )

# ── chart helpers ──────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        """
        <div style="background:rgba(30,41,59,0.55);border:1px solid rgba(100,150,220,0.25);border-radius:10px;
                    padding:18px 24px;margin-bottom:18px;">
            <div style="font-size:1.5rem;font-weight:700;color:#f1f5f9;">
                📚 University Bulk Order &amp; Predictive Procurement Analytics
            </div>
            <div style="font-size:0.85rem;color:#94a3b8;margin-top:4px;">
                ML-driven demand forecasting · price sensitivity · format adoption · risk segmentation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _section(title: str):
    st.markdown(f'<div class="chart-title-bar">{title}</div>', unsafe_allow_html=True)

# Row 2 ── Donut + Distribution histogram ──────────────────────────────────────

def render_adoption_type_optout_donut(df: pd.DataFrame):
    _section("Adoption-type-wise Opt-Out Probability")
    if df.empty:
        st.info("No data.")
        return
    agg = df.groupby("Student_Type")["Opt_Out_Probability"].mean().reset_index()
    agg.columns = ["Student_Type", "Avg_Opt_Out"]
    fig = go.Figure(go.Pie(
        labels=agg["Student_Type"],
        values=agg["Avg_Opt_Out"],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=["#14b8a6", "#3b82f6", "#f97316"]),
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    fig.update_layout(
        height=300,
        showlegend=True,
        legend=dict(orientation="v", bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(100,150,220,0.3)", borderwidth=1, font=dict(size=11, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color="#e2e8f0", family="Inter"),
    )
    fig.add_annotation(text=f"Avg<br>{agg['Avg_Opt_Out'].mean()*100:.1f}%", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#e2e8f0", family="Inter"))
    st.plotly_chart(fig, use_container_width=True)

def render_price_distribution(df: pd.DataFrame):
    _section("Distribution of Unit Price over Ranges")
    if df.empty:
        st.info("No data.")
        return
    tmp = df.copy()
    tmp["Price_Category"] = _price_bucket(tmp["Unit_Price"])
    agg = tmp.groupby("Price_Category", observed=True).size().reset_index(name="Count")
    cat_order = ["<$50", "$50-$80", "$80-$120", ">$120"]
    agg["Price_Category"] = pd.Categorical(agg["Price_Category"], categories=cat_order, ordered=True)
    agg = agg.sort_values("Price_Category")
    fig = px.bar(
        agg, x="Price_Category", y="Count",
        color="Price_Category",
        color_discrete_map={"<$50": "#14b8a6", "$50-$80": "#3b82f6", "$80-$120": "#ec4899", ">$120": "#f97316"},
        text="Count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_color="#e2e8f0")
    fig.update_layout(
        height=300, showlegend=False,
        xaxis_title="Price Range", yaxis_title="Record Count",
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 3 ── Price category bar charts ──────────────────────────────────────────

def render_price_cat_spend(df: pd.DataFrame):
    _section("Price-Category-wise Projected Spend")
    if df.empty:
        st.info("No data.")
        return
    tmp = df.copy()
    tmp["Price_Category"] = _price_bucket(tmp["Unit_Price"])
    agg = tmp.groupby("Price_Category", observed=True)["Projected_Spend"].sum().reset_index()
    cat_order = ["<$50", "$50-$80", "$80-$120", ">$120"]
    agg["Price_Category"] = pd.Categorical(agg["Price_Category"], categories=cat_order, ordered=True)
    agg = agg.sort_values("Price_Category")
    agg["Projected_Spend_M"] = agg["Projected_Spend"] / 1e6
    fig = px.bar(
        agg, x="Price_Category", y="Projected_Spend_M",
        color="Price_Category",
        color_discrete_map={"<$50": "#14b8a6", "$50-$80": "#3b82f6", "$80-$120": "#ec4899", ">$120": "#f97316"},
        text="Projected_Spend_M",
    )
    fig.update_traces(texttemplate="$%{text:.2f}M", textposition="outside", textfont_color="#e2e8f0")
    fig.update_layout(
        height=320, showlegend=False,
        xaxis_title="Price Category", yaxis_title="Total Projected Spend ($M)",
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# Row 4 ── Term-wise charts ────────────────────────────────────────────────────

def render_term_spend_ratio_by_price(df: pd.DataFrame):
    _section("Ratio of Term-wise Spend by Price Category")
    if df.empty:
        st.info("No data.")
        return
    tmp = df.copy()
    tmp["Price_Category"] = _price_bucket(tmp["Unit_Price"])
    agg = tmp.groupby(["Term", "Price_Category"], observed=True)["Projected_Spend"].sum().reset_index()
    agg["Projected_Spend_M"] = agg["Projected_Spend"] / 1e6

    # Limit to top-10 terms by total spend for readability
    top_terms = agg.groupby("Term")["Projected_Spend_M"].sum().nlargest(10).index
    agg = agg[agg["Term"].isin(top_terms)]

    fig = px.bar(
        agg, x="Term", y="Projected_Spend_M", color="Price_Category",
        barmode="relative",
        color_discrete_map={"<$50": "#14b8a6", "$50-$80": "#3b82f6", "$80-$120": "#ec4899", ">$120": "#f97316"},
    )
    fig.update_layout(
        height=340,
        xaxis_title="Term", yaxis_title="Projected Spend ($M)",
        xaxis_tickangle=45,
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_term_spend_by_adoption(df: pd.DataFrame):
    _section("Term-wise Spend by Adoption Type")
    if df.empty:
        st.info("No data.")
        return
    agg = df.groupby(["Term", "Student_Type"])["Projected_Spend"].sum().reset_index()
    agg["Projected_Spend_M"] = agg["Projected_Spend"] / 1e6

    top_terms = agg.groupby("Term")["Projected_Spend_M"].sum().nlargest(10).index
    agg = agg[agg["Term"].isin(top_terms)]

    fig = px.line(
        agg, x="Term", y="Projected_Spend_M", color="Student_Type",
        markers=True,
        color_discrete_sequence=_STACKED_COLORS,
        line_shape="spline",
    )
    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        height=340,
        xaxis_title="Term", yaxis_title="Projected Spend ($M)",
        xaxis_tickangle=45,
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 5 ── Feature importance + High-friction titles ──────────────────────────


def render_high_friction_titles(df: pd.DataFrame):
    _section("Top 10 High-Friction Titles (Negotiation Targets)")
    if df.empty:
        return
    agg = df.groupby("Title")["Opt_Out_Probability"].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(
        agg, x="Opt_Out_Probability", y="Title", orientation="h",
        color="Opt_Out_Probability", color_continuous_scale="Reds",
        text="Opt_Out_Probability",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", textfont_color="#e2e8f0")
    fig.update_layout(height=320, xaxis_title="Avg Opt-Out Risk", yaxis_title="", coloraxis_showscale=False, **_CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# Row 6 ── Top-10 Department bar + Word cloud ─────────────────────────────────

def render_dept_savings_opportunity(df: pd.DataFrame):
    _section("Top 10 Departments by Potential Savings (ROI)")
    if df.empty:
        return
    # Calculate savings per department in the sampled session
    agg = df.groupby("Dept_Code").apply(
        lambda x: (x["Unit_Price"] * x["Opt_Out_Probability"]).sum()
    ).nlargest(10).reset_index(name="Potential_Savings")
    
    fig = px.bar(
        agg.sort_values("Potential_Savings"), x="Potential_Savings", y="Dept_Code",
        orientation="h", color="Potential_Savings",
        color_continuous_scale="Viridis", text="Potential_Savings",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont_color="#e2e8f0")
    fig.update_layout(height=380, xaxis_title="Potential Savings ($)", yaxis_title="Dept Code", coloraxis_showscale=False, **_CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# Row 7 ── Scatter + Treemap ─────────────────────────────────────────────────

def render_price_vs_optout_scatter(df: pd.DataFrame):
    """Scatter: Unit Price vs Opt-Out Probability, coloured by Student Status."""
    _section("Price vs Opt-Out Probability (Scatter — Part-time vs Full-time)")
    if df.empty:
        st.info("No data.")
        return
    sample = df.sample(min(8_000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Unit_Price", y="Opt_Out_Probability",
        color="Student_Type",
        symbol="Format",
        color_discrete_map={"Full-Time": "#14b8a6", "Part-Time": "#ec4899"},
        opacity=0.55, size_max=6,
        hover_data=["Title", "Dept_Code", "Format"],
        labels={"Unit_Price": "Unit Price ($)", "Opt_Out_Probability": "Opt-Out Probability"},
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=360, xaxis_title="Unit Price ($)", yaxis_title="Opt-Out Probability", **_CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)




def render_book_quantities(df: pd.DataFrame):
    _section("Top 20 Books — Volume Forecast")
    if df.empty:
        st.info("No data.")
        return
    agg = df.groupby("Title")["Predicted_Demand_Units"].sum().reset_index()
    agg = agg.sort_values("Predicted_Demand_Units", ascending=False).head(20)
    fig = px.bar(agg, x="Title", y="Predicted_Demand_Units", color="Predicted_Demand_Units", color_continuous_scale="Viridis")
    fig.update_layout(height=420, xaxis_title="", yaxis_title="Predicted Units", coloraxis_showscale=False, **_CHART_LAYOUT)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

def render_word_cloud(df: pd.DataFrame):
    _section("Book Title Word Cloud")
    if df.empty:
        return
    agg = df.groupby("Title")["Predicted_Demand_Units"].sum()
    freq_dict = agg.to_dict()
    if not freq_dict:
        return
    wc = WordCloud(width=1200, height=350, background_color=None, mode="RGBA", colormap="Blues").generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Processing Hybrid Data Architecture (7.5M Summary + ML Sample)…"):
        summary_df_full = get_summary_data()
        raw_df_sampled = get_raw_data()
        clf, fi_df, features, acc = get_trained_model(raw_df_sampled)

    render_header()

    # Sidebar: filters + gauge
    filters_col, main_col = st.columns([0.85, 4.15])
    with filters_col:
        filter_values = render_filters(summary_df_full)
        
        # Filter both datasets
        summary_filtered = apply_filters(summary_df_full, filter_values, is_summary=True)
        sampled_filtered_base = apply_filters(raw_df_sampled, filter_values, is_summary=False)
        
        render_accuracy_gauge(acc)

    # Apply ML predictions to sampled filtered data for simulations
    sampled_filtered = apply_predictions(sampled_filtered_base, clf, features, discount_pct=0)

    with main_col:
        # ── Row 0: KPI Cards (Accuracy from Summary, Confidence from ML) ──────
        render_top_kpis(summary_filtered, sampled_filtered)

        st.markdown("<hr style='border:1px solid rgba(100,150,220,0.15);margin:10px 0;'>", unsafe_allow_html=True)

        # ── Row 1: Price Category Spend | Adoption Donut (Swapped) ─────────────
        r1_l, r1_r = st.columns(2)
        with r1_l:
            render_price_cat_spend(sampled_filtered)
        with r1_r:
            render_adoption_type_optout_donut(sampled_filtered)

        st.markdown("<hr style='border:1px solid rgba(100,150,220,0.15);margin:10px 0;'>", unsafe_allow_html=True)

        # ── Row 2: Price Distribution | High-Friction Titles ─────────────────
        r2_l, r2_r = st.columns(2)
        with r2_l:
            render_price_distribution(sampled_filtered)
        with r2_r:
            render_high_friction_titles(sampled_filtered)

        st.markdown("<hr style='border:1px solid rgba(100,150,220,0.15);margin:10px 0;'>", unsafe_allow_html=True)

        # ── Row 3: Term-wise Spend Ratio | Term-wise Adoption (line) ─────────
        r3_l, r3_r = st.columns(2)
        with r3_l:
            render_term_spend_ratio_by_price(sampled_filtered)
        with r3_r:
            render_term_spend_by_adoption(sampled_filtered)

        st.markdown("<hr style='border:1px solid rgba(100,150,220,0.15);margin:10px 0;'>", unsafe_allow_html=True)

        # ── Row 4: Scatter Plot (Student focus) | Dept Savings ROI ──────────
        r4_l, r4_r = st.columns(2)
        with r4_l:
            render_price_vs_optout_scatter(sampled_filtered)
        with r4_r:
            render_dept_savings_opportunity(sampled_filtered)

        st.markdown("<hr style='border:1px solid rgba(100,150,220,0.15);margin:10px 0;'>", unsafe_allow_html=True)

        # ── Row 5: Volume Forecast & Word Cloud ─────────────────────────────
        render_book_quantities(sampled_filtered)
        render_word_cloud(sampled_filtered)


if __name__ == "__main__":
    main()
