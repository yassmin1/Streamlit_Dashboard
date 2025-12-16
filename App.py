"""
Student Insights Dashboard
A Streamlit application for analyzing student enrollment, demographics, and program trends.
Optimized for performance and production deployment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define analysis columns - these are the categorical variables we'll analyze
ANALYSIS_COLUMNS = [
    "Academic Period",
    "Gender",
    "Student Type",
    "Full_Part Time",
    "Ethnicity",
    "Major",
    "Age_Group",
]

# Color scheme for consistent branding
COLORS = {
    "primary": "#34568B",      # Royal blue
    "secondary": "#FFB347",    # Light orange
    "chart": "Set3"            # Plotly color scheme
}

# Performance thresholds
LARGE_DATASET_THRESHOLD = 110000
MAX_CATEGORIES_DISPLAY = 20
MAX_CROSSTAB_CELLS = 1000
SAMPLE_SIZE = 10000

# =============================================================================
# DATA LOADING AND CACHING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(file_path: str, columns: List[str] = ANALYSIS_COLUMNS) -> Optional[pd.DataFrame]:
    """
    Load and preprocess student data with caching for optimal performance.
    
    Args:
        file_path: Path to the parquet file
        columns: List of columns to load
        
    Returns:
        Preprocessed DataFrame or None if loading fails
    """
    try:
        # Load only required columns for memory efficiency
        df = pd.read_parquet(file_path, columns=columns)
        
        # Clean data: convert to string and strip whitespace
        df = df.astype(str).apply(lambda x: x.str.strip())
        
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


@st.cache_data
def get_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Calculate value counts for a column with caching.
    
    Args:
        df: Source DataFrame
        column: Column name to analyze
        
    Returns:
        Series with value counts sorted by index
    """
    return df[column].value_counts().sort_index()


@st.cache_data
def get_crosstab(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Generate cross-tabulation between two columns with caching.
    
    Args:
        df: Source DataFrame
        col1: First categorical column
        col2: Second categorical column
        
    Returns:
        Cross-tabulation DataFrame
    """
    return pd.crosstab(df[col1], df[col2])


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    orientation: str = "vertical",
    title: str = "",
    hover_template: str = None,
    customdata: pd.Series = None
) -> go.Figure:
    """
    Create a styled bar chart with consistent branding.
    
    Args:
        data: DataFrame with chart data
        x_col: Column for x-axis
        y_col: Column for y-axis
        orientation: 'vertical' or 'horizontal'
        title: Chart title
        hover_template: Custom hover text template
        customdata: Additional data for hover
        
    Returns:
        Plotly figure object
    """
    # Create base chart
    if orientation == "horizontal":
        fig = px.bar(
            data,
            x=y_col,
            y=x_col,
            orientation="h",
            title=title,
            color=y_col,
            color_continuous_scale="viridis"
        )
    else:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=y_col,
            color_continuous_scale="viridis"
        )
    
    # Apply consistent styling
    fig.update_layout(
        title_font_size=16,
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Add custom hover if provided
    if hover_template:
        if customdata is not None:
            fig.update_traces(customdata=customdata, hovertemplate=hover_template)
        else:
            fig.update_traces(hovertemplate=hover_template)
    
    return fig


def create_heatmap(
    crosstab: pd.DataFrame,
    row_label: str,
    col_label: str,
    text_size: int = 12
) -> go.Figure:
    """
    Create an annotated heatmap from cross-tabulation data.
    
    Args:
        crosstab: Cross-tabulation DataFrame
        row_label: Label for rows (y-axis)
        col_label: Label for columns (x-axis)
        text_size: Font size for cell annotations
        
    Returns:
        Plotly figure object
    """
    # Ensure proper data types for labels
    crosstab.columns = crosstab.columns.astype(str)
    crosstab.index = crosstab.index.astype(str)
    
    # Create heatmap
    fig = px.imshow(
        crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        aspect="auto",
        title=f"{row_label} vs {col_label} (Heatmap)",
        labels=dict(x=col_label, y=row_label, color="Count"),
        color_continuous_scale="Viridis"
    )
    
    # Add text annotations
    text_values = crosstab.round(0).astype(int).astype(str)
    fig.update_traces(
        text=text_values.values,
        texttemplate="%{text}",
        textfont={"size": text_size},
        hovertemplate=f"{row_label}=%{{y}}<br>{col_label}=%{{x}}<br>Count=%{{z:.0f}}<extra></extra>"
    )
    
    # Style layout
    fig.update_layout(
        height=450,
        xaxis_title=col_label,
        yaxis_title=row_label,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def apply_custom_css():
    """Apply custom CSS styling for the application."""
    st.markdown(
        f"""
        <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['primary']};
        }}
        
        [data-testid="stSidebar"] * {{
            color: {COLORS['secondary']} !important;
        }}
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            color: {COLORS['secondary']} !important;
        }}
        
        /* Main content styling */
        .main {{
            background-color: #f8f9fa;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def display_header():
    """Display the application header with logo and title."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        logo_path = Path("assets/analysis.png")
        if logo_path.exists():
            st.image(str(logo_path), width=180)
        else:
            st.info("📊")  # Fallback emoji if logo missing
    
    with col2:
        st.markdown(
            f"""
            <h1 style='color:{COLORS['primary']}; font-size:70px; 
                       text-align:center; margin-top:-50px'>
                Student Insights
            </h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style='color:{COLORS['primary']}; font-size:28px; 
                        text-align:center; line-height:1.2; margin-top:-10px'>
                <strong>
                    An interactive dashboard designed to explore student enrollment, 
                    demographics, and program trends. Use the sidebar filters to select 
                    academic terms, majors, and student characteristics to uncover 
                    patterns that support data-informed decisions.
                </strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")


def create_multiselect_filter(
    df: pd.DataFrame,
    column: str,
    label: str,
    key_prefix: str
) -> List[str]:
    """
    Create a sidebar multiselect filter with 'Select All' functionality.
    
    Args:
        df: Source DataFrame
        column: Column to filter
        label: Display label
        key_prefix: Unique key prefix for session state
        
    Returns:
        List of selected values
    """
    # Get unique, sorted options
    options = sorted(
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    
    st.subheader(label)
    
    # Select all checkbox
    select_all_key = f"{key_prefix}_select_all"
    multi_key = f"{key_prefix}_multiselect"
    
    select_all = st.checkbox(f"Select all {label.lower()}", value=True, key=select_all_key)
    
    if select_all:
        # Show disabled multiselect when all selected
        st.multiselect(
            label,
            options=options,
            default=options,
            key=multi_key,
            disabled=True,
            help=f"All {label.lower()} are included."
        )
        return options
    else:
        # Interactive multiselect
        default_vals = st.session_state.get(multi_key, options)
        selected = st.multiselect(
            label,
            options=options,
            default=default_vals,
            key=multi_key,
            help=f"Uncheck 'Select all {label.lower()}' to filter."
        )
        
        if len(selected) == 0:
            st.warning(f"⚠️ No {label.lower()} selected — results may be empty.")
        
        return selected


def display_summary_stats(df: pd.DataFrame, column: str, value_counts: pd.Series):
    """
    Display summary statistics in the sidebar.
    
    Args:
        df: Source DataFrame
        column: Analyzed column
        value_counts: Value counts for the column
    """
    st.subheader("Summary Statistics")
    
    total_count = len(df)
    unique_categories = df[column].nunique()
    
    st.metric("Total Records", f"{total_count:,}")
    st.metric("Unique Categories", f"{unique_categories}")
    
    # Frequency table
    st.subheader("Count Summary")
    
    # Prepare display data
    freq_data = pd.DataFrame({
        column: value_counts.index,
        'Count': value_counts.values
    })
    
    # Limit display size
    if len(freq_data) > 10:
        freq_data = freq_data.head(10)
        st.info(f"Showing top 10 out of {len(value_counts)} categories")
    
    # Transpose for better display
    freq_table = freq_data.set_index(column).T
    st.dataframe(freq_table, use_container_width=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Configure page
    st.set_page_config(
        page_title="Student Insights Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    apply_custom_css()
    
    # Display header
    display_header()
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        data = load_data("data/curated/data_merged.parquet")
    
    if data is None or data.empty:
        st.error("❌ Failed to load data. Please check the file path and try again.")
        st.stop()
    
    # Store data info in session state
    if 'data_info' not in st.session_state:
        st.session_state.data_info = {
            'total_records': len(data),
            'columns': data.columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    categorical_columns = st.session_state.data_info['categorical_columns']
    
    # =============================================================================
    # SIDEBAR CONTROLS
    # =============================================================================
    
    with st.sidebar:
        st.markdown(
            f"<h1 style='color:{COLORS['secondary']}; font-size:35px; "
            f"text-align:center; margin-top:0px'>Analysis Controls</h1>",
            unsafe_allow_html=True
        )
        
        # Category selection
        selected_column = st.selectbox(
            "Select Category to Analyze",
            categorical_columns,
            index=0
        )
        
        st.markdown("---")
        
        # Chart orientation
        orientation = st.radio("Chart Orientation", ["vertical", "horizontal"])
        
        st.markdown("---")
        
        # Term filter
        selected_terms = create_multiselect_filter(
            data, "Academic Period", "Academic Periods", "terms"
        )
        
        st.markdown("---")
        
        # Major filter
        selected_majors = create_multiselect_filter(
            data, "Major", "Majors", "majors"
        )
        
        st.markdown("---")
        
        # Percentage toggle
        show_percentages = st.checkbox("Show Percentages", value=False)
    
    # =============================================================================
    # FILTER DATA
    # =============================================================================
    
    if not selected_terms or not selected_majors:
        st.warning("⚠️ Please select at least one term and one major to display data.")
        st.stop()
    
    # Apply filters
    filtered_data = data[
        data["Academic Period"].isin(selected_terms) & 
        data["Major"].isin(selected_majors)
    ]
    
    if filtered_data.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selections.")
        st.stop()
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION
    # =============================================================================
    
    display_data = filtered_data
    
    if len(filtered_data) > LARGE_DATASET_THRESHOLD:
        st.info(f"ℹ️ Dataset has {len(filtered_data):,} records. Using optimized processing.")
        
        use_sampling = st.checkbox("Use sampling for faster processing", value=False)
        
        if use_sampling:
            sample_size = min(SAMPLE_SIZE, len(filtered_data))
            display_data = filtered_data.sample(n=sample_size, random_state=42)
            st.info(f"📊 Using sample of {sample_size:,} records")
    
    # =============================================================================
    # MAIN VISUALIZATION
    # =============================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get value counts
        value_counts = get_value_counts(display_data, selected_column)
        
        # Limit categories
        if len(value_counts) > MAX_CATEGORIES_DISPLAY:
            st.info(f"ℹ️ Showing top {MAX_CATEGORIES_DISPLAY} categories out of {len(value_counts)} total")
            value_counts = value_counts.head(MAX_CATEGORIES_DISPLAY)
        
        # Prepare chart data
        chart_data = pd.DataFrame({
            selected_column: value_counts.index,
            'Count': value_counts.values
        })
        
        # Add percentages if requested
        if show_percentages:
            chart_data['Percentage'] = (chart_data['Count'] / chart_data['Count'].sum()) * 100
            y_col = 'Percentage'
            title = f"Distribution of {selected_column} (%)"
            hover_template = '<b>%{x}</b><br>Count: %{customdata}<br>Percentage: %{y:.1f}%<extra></extra>'
            customdata = chart_data['Count']
        else:
            y_col = 'Count'
            title = ""
            hover_template = '<b>%{x}</b><br>Count: %{y}<extra></extra>'
            customdata = None
        
        # Create and display chart
        fig = create_bar_chart(
            chart_data,
            selected_column,
            y_col,
            orientation,
            title,
            hover_template,
            customdata
        )
        
        # Add text annotations for small datasets
        if len(chart_data) <= 15:
            if show_percentages:
                text_values = [
                    f"{count}<br>({pct:.1f}%)" 
                    for count, pct in zip(chart_data['Count'], chart_data['Percentage'])
                ]
            else:
                text_values = chart_data['Count'].astype(str)
            
            fig.update_traces(
                text=text_values,
                textposition='outside' if orientation == 'vertical' else 'inside'
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        display_summary_stats(display_data, selected_column, value_counts)
    
    # =============================================================================
    # CROSS-CATEGORY ANALYSIS
    # =============================================================================
    
    st.markdown("---")
    st.subheader("Cross-Category Analysis")
    
    # Check dataset size
    if len(display_data) > LARGE_DATASET_THRESHOLD:
        st.warning(
            f"⚠️ Cross-category analysis disabled for large datasets (>{LARGE_DATASET_THRESHOLD:,} records) "
            "to maintain performance. Please use filters to narrow down the data, or enable sampling."
        )
        st.stop()
    
    # Category selection
    col_a, col_b = st.columns(2)
    
    with col_a:
        category_1 = st.selectbox("First Category", categorical_columns, index=0)
    
    with col_b:
        if len(categorical_columns) > 1:
            default_idx = 1
        else:
            st.warning("⚠️ Need at least 2 categorical columns for cross-analysis")
            st.stop()
        category_2 = st.selectbox("Second Category", categorical_columns, index=default_idx)
    
    if category_1 == category_2:
        st.info("ℹ️ Please select two different categories for comparison")
        st.stop()
    
    # Check cross-tabulation size
    unique_cat1 = display_data[category_1].nunique()
    unique_cat2 = display_data[category_2].nunique()
    
    if unique_cat1 * unique_cat2 > MAX_CROSSTAB_CELLS:
        st.warning(
            f"⚠️ Cross-tabulation too large ({unique_cat1} × {unique_cat2} cells). "
            "Please select categories with fewer unique values."
        )
        st.stop()
    
    # Create cross-tabulation
    col3, col4 = st.columns(2)
    
    with col3:
        crosstab = get_crosstab(display_data, category_1, category_2)
        
        # Limit display size
        if len(crosstab.index) > 10:
            crosstab = crosstab.head(10)
            st.info("ℹ️ Showing top 10 categories for performance")
        
        if len(crosstab.columns) > 10:
            crosstab = crosstab.iloc[:, :10]
            st.info("ℹ️ Showing top 10 subcategories for performance")
        
        # Stacked bar chart
        fig_bar = px.bar(
            crosstab.T,
            title=f"{category_1} vs {category_2} Distribution",
            height=400
        )
        
        fig_bar.update_layout(
            xaxis_title=category_2,
            yaxis_title="Count",
            showlegend=True,
            legend_title=category_1
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Heatmap
        fig_heatmap = create_heatmap(crosstab, category_1, category_2)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col4:
        st.subheader("Cross-Tabulation Data")
        
        # Show preview for large tables
        if crosstab.size > 100:
            st.info("ℹ️ Showing preview of cross-tabulation")
            st.dataframe(crosstab.iloc[:5, :5], use_container_width=True)
        else:
            st.dataframe(crosstab, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary")
        st.write(f"**Categories in {category_1}:** {len(crosstab.index)}")
        st.write(f"**Categories in {category_2}:** {len(crosstab.columns)}")
        st.write(f"**Total Combinations:** {crosstab.sum().sum():,}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
