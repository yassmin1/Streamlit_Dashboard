# ============================================
# Student Insights Dashboard (Production)
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================
# CONFIG
# ============================================

COLUMNS = [
    "Academic Period",
    "Gender",
    "Student Type",
    "Full_Part Time",
    "Ethnicity",
    "Major",
    "Age_Group",
]

PRIMARY_COLOR = "#34568B"

# ============================================
# UTILITIES
# ============================================

def safe_image(path: str, width: int = 180):
    if Path(path).exists():
        st.image(path, width=width)

# ============================================
# DATA
# ============================================

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)[COLUMNS]
    df = df.astype(str).apply(lambda c: c.str.strip())
    return df

@st.cache_data
def value_counts_alpha(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts()
    vc = vc.sort_index()          # alphabetical order (INTENTIONAL)
    out = vc.reset_index()
    out.columns = [col, "Count"]
    return out

@st.cache_data
def crosstab(df: pd.DataFrame, c1: str, c2: str) -> pd.DataFrame:
    return pd.crosstab(df[c1], df[c2])

# ============================================
# VISUALS
# ============================================

def bar_chart(df: pd.DataFrame, cat_col: str, value_col: str, orientation: str):
    if orientation == "horizontal":
        fig = px.bar(
            df,
            y=cat_col,
            x=value_col,
            orientation="h",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_xaxes(type="linear", title=value_col)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=df[cat_col].tolist(),
            title=cat_col,
        )
    else:
        fig = px.bar(
            df,
            x=cat_col,
            y=value_col,
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_yaxes(type="linear", title=value_col)
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=df[cat_col].tolist(),
            title=cat_col,
        )

    fig.update_layout(
        height=480,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        if orientation == "vertical"
        else "<b>%{y}</b><br>Count: %{x}<extra></extra>"
    )

    return fig

def heatmap(ct: pd.DataFrame, r: str, c: str):
    ct = ct.copy()
    ct.index = ct.index.astype(str)
    ct.columns = ct.columns.astype(str)

    fig = go.Figure(
        go.Heatmap(
            z=ct.values,
            x=ct.columns,
            y=ct.index,
            colorscale="Viridis",
            text=ct.values,
            texttemplate="%{text}",
            hovertemplate=f"{r}=%{{y}}<br>{c}=%{{x}}<br>Count=%{{z}}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        height=450,
        title=f"{r} vs {c}",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig

# ============================================
# APP
# ============================================

def main():

    st.set_page_config(page_title="Student Insights", layout="wide")

    # Header
    h1, h2 = st.columns([1, 4])
    with h1:
        safe_image("assets/analysis.png")
    with h2:
        st.markdown(
            "<h1 style='color:#34568B;font-size:56px'>Student Insights</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:20px'>Interactive view of enrollment and demographics</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Load
    df = load_data("data/curated/data_merged.parquet")

    # Sidebar
    st.sidebar.header("Controls")
    col = st.sidebar.selectbox("Category", df.columns)
    orientation = st.sidebar.radio("Orientation", ["vertical", "horizontal"])

    # Main layout
    left, right = st.columns([2, 1])

    with left:
        counts = value_counts_alpha(df, col)
        fig = bar_chart(counts, col, "Count", orientation)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Counts")
        st.dataframe(counts, use_container_width=True)

    # Cross analysis
    st.divider()
    st.subheader("Cross Category Analysis")

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Row", df.columns, key="row")
    with c2:
        b = st.selectbox("Column", df.columns, key="col")

    if a != b:
        ct = crosstab(df, a, b)
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(heatmap(ct, a, b), use_container_width=True)
        with h2:
            st.dataframe(ct, use_container_width=True)

# ============================================
# ENTRY
# ============================================

if __name__ == "__main__":
    main()
