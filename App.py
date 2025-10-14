import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Student Insights", layout="wide")

# =========================================
# CANONICAL COLUMN NAMES (after ETL)
# =========================================
# We accept common variants from legacy files and normalize them once.
CANONICAL_COLUMNS = {
    "Academic Period": {"Academic Period"},
    "Calendar Year": {"Calendar Year"},
    "Gender": {"Gender"},
    "Student Type": {"Student Type"},
    "Full/Part Time": {"Full/Part Time", "Full_Part Time", "Full_Part_Time"},
    "Ethnicity": {"Ethnicity", "ETHNICITY"},
    "Major": {"Major", "Major Desc"},
    "Age Group": {"Age Group", "Age_Group", "AgeGroup"},
}

REQUIRED_CANONICAL = list(CANONICAL_COLUMNS.keys())

# =========================================
# HELPERS
# =========================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming DataFrame columns to the canonical names used by the app.
    - If multiple variants exist, the first found is mapped to the canonical name.
    - Unknown columns are kept as-is.
    """
    src_cols = set(df.columns)
    rename_map = {}
    for canonical, variants in CANONICAL_COLUMNS.items():
        for v in variants:
            if v in src_cols:
                rename_map[v] = canonical
                break
    out = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED_CANONICAL if c not in out.columns]
    if missing:
        st.warning(
            "Some expected columns are missing after normalization: "
            + ", ".join(missing)
            + ". The app will continue, but some views may be limited."
        )
    return out


def _safe_image(path: str, width: int = 200) -> None:
    """
    Try to display an image; show a small placeholder message if the file is missing.
    """
    try:
        if Path(path).exists():
            st.image(path, width=width)
        else:
            st.info("Logo not found (assets/analysis.png). You can add one later.")
    except Exception:
        st.info("Logo not available. Continuing without it.")


def _select_all_multiselect(label: str, options: list[str], key: str, default_all: bool = True) -> list[str]:
    """
    Multiselect with 'Select all' behavior preserved in session state.
    """
    # Keep options stable & sorted
    options = sorted(dict.fromkeys([str(x).strip() for x in options if pd.notna(x) and str(x).strip() != ""]))
    all_key = f"{key}_all"
    multi_key = f"{key}_values"

    select_all = st.checkbox(f"Select all {label.lower()}", value=default_all, key=all_key)
    if select_all:
        st.multiselect(label, options=options, default=options, key=multi_key, disabled=True, help="All selected")
        return options
    else:
        default_vals = st.session_state.get(multi_key, options)
        chosen = st.multiselect(label, options=options, default=default_vals, key=multi_key)
        if not chosen:
            st.info(f"No {label.lower()} selected — results may be empty.")
        return chosen


def _value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False)
    out = vc.reset_index()
    out.columns = [col, "Count"]
    return out


# =========================================
# CACHING LAYER
# =========================================
@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def normalize_cached(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_columns(df)

@st.cache_data(show_spinner=False)
def cross_tab(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    ct = pd.crosstab(df[a], df[b])
    # order columns by total frequency for readability
    return ct[ct.sum(axis=0).sort_values(ascending=False).index]

# =========================================
# CHART BUILDERS
# =========================================
def build_bar(count_df: pd.DataFrame, cat_col: str, show_pct: bool, orient: str, top_n: int) -> px.bar:
    data = count_df.copy()
    if len(data) > top_n:
        data = data.head(top_n)
        st.info(f"Showing top {top_n} categories")

    if show_pct:
        data["Percentage"] = (data["Count"] / data["Count"].sum()) * 100.0
        y = "Percentage"
        title = f"Distribution of {cat_col} (%)"
        color_col = y
        hover = '<b>%{x}</b><br>Count: %{customdata}<br>Percentage: %{y:.1f}%<extra></extra>'
        custom = data["Count"]
    else:
        y = "Count"
        title = f"Count of Each {cat_col}"
        color_col = y
        hover = '<b>%{x}</b><br>Count: %{y}<extra></extra>'
        custom = None

    if orient == "horizontal":
        fig = px.bar(data, x=y, y=cat_col, orientation="h", title=title, color=color_col, color_continuous_scale="viridis")
    else:
        fig = px.bar(data, x=cat_col, y=y, title=title, color=color_col, color_continuous_scale="viridis")

    fig.update_layout(
        title_font_size=16,
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    if custom is not None:
        fig.update_traces(customdata=custom, hovertemplate=hover)
    else:
        fig.update_traces(hovertemplate=hover)

    # add text labels when small
    if len(data) <= 15:
        if show_pct:
            text_vals = [f"{c}<br>({p:.1f}%)" for c, p in zip(data["Count"], data["Percentage"])]
        else:
            text_vals = data["Count"].astype(str)
        fig.update_traces(text=text_vals, textposition="outside" if orient == "vertical" else "inside")

    return fig


def build_heatmap(ct: pd.DataFrame, a: str, b: str) -> px.imshow:
    fig = px.imshow(
        ct.values,
        x=ct.columns.astype(str),
        y=ct.index.astype(str),
        aspect="auto",
        title=f"{a} × {b} (Heatmap)",
        labels=dict(x=b, y=a, color="Count")
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =========================================
# SIDEBAR / DATA LOADING
# =========================================
def sidebar_data_loader(default_path: str = "data/curated/data_merged.parquet") -> pd.DataFrame | None:
    st.sidebar.header("Data")
    st.sidebar.caption("Load the curated Parquet created by your ETL.")

    # Allow file upload OR use default path
    uploaded = st.sidebar.file_uploader("Upload Parquet", type=["parquet"])
    path_str = st.sidebar.text_input("Or Parquet path", value=default_path)

    df = None
    try:
        if uploaded is not None:
            df = pd.read_parquet(uploaded)
        else:
            if not Path(path_str).exists():
                st.sidebar.warning(f"Path not found: {path_str}")
            else:
                df = load_parquet(path_str)
    except Exception as e:
        st.sidebar.error(f"Failed to load data: {e}")
        return None

    if df is not None and not df.empty:
        df = normalize_cached(df)
        # keep only columns we actually need if present
        keep = [c for c in REQUIRED_CANONICAL if c in df.columns]
        if not keep:
            st.sidebar.error("No expected columns found after normalization.")
            return None
        df = df[keep].copy()
        return df

    st.sidebar.error("No data loaded.")
    return None

# =========================================
# MAIN APP
# =========================================
def main() -> None:
    # Header
    header_left, header_right = st.columns([1, 4])
    with header_left:
        _safe_image("assets/analysis.PNG", width=180)
    with header_right:
        st.title("Student Insights")
        st.markdown(
            """
            <div style='color:#555;font-size:18px;line-height:1.4;margin-top:-6px'>
              <strong>Explore student enrollment, demographics, and program trends.</strong><br>
              Use the sidebar filters to slice by term, majors, and characteristics.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # Data
    with st.spinner("Loading data..."):
        df = sidebar_data_loader()

    if df is None or df.empty:
        st.error("No data available. Please upload or set a valid Parquet path.")
        return

    # Derive categorical list (stable order)
    categorical_cols = [c for c in df.columns if df[c].dtype.name in ("object", "category")]

    # -----------------------------
    # SIDEBAR CONTROLS
    # -----------------------------
    st.sidebar.header("Filters")

    # Terms
    terms = sorted(df["Academic Period"].dropna().astype(str).unique().tolist()) if "Academic Period" in df.columns else []
    chosen_terms = st.sidebar.multiselect("Academic Periods", terms, default=terms)

    # Majors (select-all UX)
    chosen_majors = _select_all_multiselect("Majors", df["Major"].unique().tolist() if "Major" in df.columns else [], key="majors")

    # Optional extra filters (toggleable for simplicity)
    with st.sidebar.expander("More filters", expanded=False):
        chosen_gender = st.multiselect("Gender", sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else [])
        chosen_eth = st.multiselect("Ethnicity", sorted(df["Ethnicity"].dropna().unique().tolist()) if "Ethnicity" in df.columns else [])
        chosen_ftpt = st.multiselect("Full/Part Time", sorted(df["Full/Part Time"].dropna().unique().tolist()) if "Full/Part Time" in df.columns else [])
        chosen_ageg = st.multiselect("Age Group", sorted(df["Age Group"].dropna().unique().tolist()) if "Age Group" in df.columns else [])

    # Apply filters
    filtered = df.copy()
    if chosen_terms:
        filtered = filtered[filtered["Academic Period"].astype(str).isin(chosen_terms)]
    if chosen_majors:
        filtered = filtered[filtered["Major"].astype(str).isin(chosen_majors)]
    if chosen_gender:
        filtered = filtered[filtered["Gender"].isin(chosen_gender)]
    if chosen_eth:
        filtered = filtered[filtered["Ethnicity"].isin(chosen_eth)]
    if chosen_ftpt:
        filtered = filtered[filtered["Full/Part Time"].isin(chosen_ftpt)]
    if chosen_ageg:
        filtered = filtered[filtered["Age Group"].isin(chosen_ageg)]

    if filtered.empty:
        st.warning("No rows match your current filters.")
        return

    # Performance: optional sampling
    if len(filtered) > 100_000:
        st.sidebar.info(f"Large dataset detected ({len(filtered):,}). Sampling 20,000 rows for speed.")
        filtered = filtered.sample(n=min(20_000, len(filtered)), random_state=42)

    # -----------------------------
    # PRIMARY ANALYSIS CONTROLS
    # -----------------------------
    st.sidebar.header("Analysis")
    selected_col = st.sidebar.selectbox("Category to Analyze", categorical_cols, index=0 if categorical_cols else None)
    orientation = st.sidebar.radio("Chart Orientation", ["vertical", "horizontal"], index=0, horizontal=True)
    show_pct = st.sidebar.checkbox("Show Percentages", value=False)
    top_n = st.sidebar.slider("Top N categories", min_value=5, max_value=50, value=20, step=1)

    # -----------------------------
    # MAIN LAYOUT
    # -----------------------------
    left, right = st.columns([2, 1])

    with left:
        counts = _value_counts(filtered, selected_col)
        fig = build_bar(counts, selected_col, show_pct=show_pct, orient=orientation, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Summary")
        total_rows = len(filtered)
        uniq = filtered[selected_col].nunique(dropna=False)
        most = counts.iloc[0][selected_col] if not counts.empty else "N/A"
        most_n = int(counts.iloc[0]["Count"]) if not counts.empty else 0

        st.metric("Rows (filtered)", f"{total_rows:,}")
        st.metric("Unique categories", f"{uniq:,}")
        st.metric("Most common", str(most))
        st.metric("Most common count", f"{most_n:,}")

        st.subheader("Top Categories")
        preview = counts.head(min(10, len(counts))).copy()
        if show_pct and not preview.empty:
            preview["Percentage"] = (preview["Count"] / counts["Count"].sum() * 100).round(1)
        st.dataframe(preview, use_container_width=True, height=min(420, 40 + 35 * len(preview)))

        # Download filtered data
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="student_insights_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -----------------------------
    # CROSS-CATEGORY ANALYSIS
    # -----------------------------
    st.markdown("---")
    st.subheader("Cross-Category Analysis")

    if len(filtered) > 50_000:
        st.warning("Cross-category analysis disabled for large filtered datasets (>50,000 rows). Refine filters to enable.")
        return

    c1, c2 = st.columns(2)
    with c1:
        cat_a = st.selectbox("First Category", categorical_cols, index=0)
    with c2:
        # pick different default if possible
        idx2 = 1 if len(categorical_cols) > 1 else 0
        cat_b = st.selectbox("Second Category", categorical_cols, index=idx2)

    if cat_a == cat_b:
        st.info("Choose two different categories for cross-analysis.")
        return

    # Limit size of the crosstab for performance/readability
    u_a, u_b = filtered[cat_a].nunique(), filtered[cat_b].nunique()
    if u_a * u_b > 500:
        st.warning(f"Cross-tab too large ({u_a} x {u_b}). Please choose categories with fewer unique values or reduce filters.")
        return

    ct = cross_tab(filtered, cat_a, cat_b)

    vis_choice = st.radio("Visualization", ["Stacked Bars", "Heatmap"], horizontal=True, index=0)
    if vis_choice == "Stacked Bars":
        # bar of ct.T is an intuitive “per B” breakdown
        fig2 = px.bar(ct.T, title=f"{cat_a} vs {cat_b} (Stacked)", height=450)
        fig2.update_layout(
            xaxis_title=cat_b,
            yaxis_title="Count",
            legend_title=cat_a,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.plotly_chart(build_heatmap(ct, cat_a, cat_b), use_container_width=True)

    with st.expander("Cross-tab Table"):
        # Show a manageable slice if big
        if ct.size > 400:
            st.info("Previewing a 10 × 10 slice for readability.")
            st.dataframe(ct.iloc[:10, :10], use_container_width=True)
        else:
            st.dataframe(ct, use_container_width=True)

# =========================================
# ENTRYPOINT
# =========================================
if __name__ == "__main__":
    main()
