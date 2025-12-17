"""
Student Insights Dashboard v7
Fixed version ensuring proper rendering on normal servers
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

# Set page config FIRST
st.set_page_config(
    page_title="Student Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ANALYSIS_COLUMNS = [
    "Academic Period",
    "Gender",
    "Student Type",
    "Full_Part Time",
    "Ethnicity",
    "Major",
    "Age_Group",
]

COLORS = {
    "primary": "#34568B",
    "secondary": "#FFB347",
}

LARGE_DATASET_THRESHOLD = 110000
MAX_CATEGORIES_DISPLAY = 20
MAX_CROSSTAB_CELLS = 1000
SAMPLE_SIZE = 10000

DATA_FILE_PATH = os.environ.get('DATA_FILE_PATH', 'data/curated/data_merged.parquet')
ASSETS_PATH = os.environ.get('ASSETS_PATH', 'assets')

# =============================================================================
# CACHING FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(file_path, columns=ANALYSIS_COLUMNS):
    """Load and preprocess data"""
    try:
        df = pd.read_parquet(file_path, columns=columns)
        df = df.astype(str).apply(lambda x: x.str.strip())
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


@st.cache_data
def get_value_counts(df, column, sort_by="Name"):
    """Calculate value counts with sorting"""
    value_counts = df[column].value_counts()
    if sort_by == "Name":
        return value_counts.sort_index()
    else:
        return value_counts.sort_values(ascending=False)


@st.cache_data
def get_crosstab(df, col1, col2):
    """Generate cross-tabulation"""
    return pd.crosstab(df[col1], df[col2])


# =============================================================================
# STYLING
# =============================================================================

def apply_custom_css():
    """Apply custom CSS"""
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {COLORS['primary']};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLORS['secondary']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_bar_chart(data, x_col, y_col, orientation="vertical", title=""):
    """
    Create bar chart with EXPLICIT y-axis configuration to ensure counts show
    """
    if orientation == "horizontal":
        fig = px.bar(
            data,
            x=y_col,
            y=x_col,
            orientation="h",
            title=title,
            color=y_col,
            color_continuous_scale="Viridis"
        )
        # EXPLICIT axis configuration
        fig.update_xaxes(
            title=y_col,
            showticklabels=True,
            showgrid=True
        )
        fig.update_yaxes(
            title=x_col,
            showticklabels=True,
            type='category'
        )
    else:
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=y_col,
            color_continuous_scale="Viridis"
        )
        # EXPLICIT axis configuration
        fig.update_xaxes(
            title=x_col,
            showticklabels=True,
            type='category'
        )
        fig.update_yaxes(
            title=y_col,
            showticklabels=True,
            showgrid=True,
            rangemode='tozero'  # Ensure y-axis starts at 0
        )
    
    # Force layout updates
    fig.update_layout(
        title_font_size=16,
        height=450,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showline=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2
        ),
        yaxis=dict(
            showline=True,
            showticklabels=True,
            showgrid=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            gridcolor='rgb(230, 230, 230)'
        ),
        margin=dict(l=60, r=20, t=50, b=60)  # Ensure space for labels
    )
    
    return fig


def create_heatmap(crosstab, row_label, col_label):
    """
    Create heatmap with EXPLICIT text annotations to ensure values show
    """
    # Convert to string for labels
    crosstab_copy = crosstab.copy()
    crosstab_copy.columns = crosstab_copy.columns.astype(str)
    crosstab_copy.index = crosstab_copy.index.astype(str)
    
    # Create annotations as text
    annotations = []
    for i, row in enumerate(crosstab_copy.index):
        for j, col in enumerate(crosstab_copy.columns):
            value = crosstab_copy.iloc[i, j]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(int(value)),
                    showarrow=False,
                    font=dict(size=12, color='white' if value > crosstab_copy.values.max()/2 else 'black')
                )
            )
    
    # Create heatmap using graph_objects for more control
    fig = go.Figure(data=go.Heatmap(
        z=crosstab_copy.values,
        x=crosstab_copy.columns.tolist(),
        y=crosstab_copy.index.tolist(),
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate=f'{row_label}: %{{y}}<br>{col_label}: %{{x}}<br>Count: %{{z}}<extra></extra>'
    ))
    
    # Add annotations
    fig.update_layout(
        annotations=annotations,
        title=f"{row_label} vs {col_label} (Heatmap)",
        xaxis_title=col_label,
        yaxis_title=row_label,
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=20, t=50, b=100)
    )
    
    # Ensure axes show
    fig.update_xaxes(
        showticklabels=True,
        type='category',
        side='bottom'
    )
    fig.update_yaxes(
        showticklabels=True,
        type='category',
        autorange='reversed'
    )
    
    return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_header():
    """Display header with logo and responsive sizing"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        logo_path = Path(ASSETS_PATH) / "analysis.png"
        if logo_path.exists():
            try:
                # Use container width for responsive sizing
                st.image(str(logo_path), use_column_width=True)
            except:
                st.info("📊")
        else:
            st.info("📊")
    
    with col2:
        st.markdown(
            f"""
            <h1 style='color:{COLORS['primary']}; font-size:clamp(40px, 5vw, 70px); 
                       text-align:center; margin-top:-20px; line-height:1.2;'>
                Student Insights
            </h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style='color:{COLORS['primary']}; font-size:clamp(16px, 2vw, 28px); 
                        text-align:center; line-height:1.3; margin-top:-5px; padding:0 10px;'>
                <strong>
                    An interactive dashboard to explore student enrollment, 
                    demographics, and program trends.
                </strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")


def create_multiselect_filter(df, column, label, key_prefix):
    """Create multiselect filter with select all option"""
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
    
    select_all_key = f"{key_prefix}_select_all"
    multi_key = f"{key_prefix}_multiselect"
    
    select_all = st.checkbox(f"Select all {label.lower()}", value=True, key=select_all_key)
    
    if select_all:
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
        default_vals = st.session_state.get(multi_key, options)
        selected = st.multiselect(
            label,
            options=options,
            default=default_vals,
            key=multi_key,
            help=f"Uncheck 'Select all {label.lower()}' to filter."
        )
        
        if len(selected) == 0:
            st.warning(f"⚠️ No {label.lower()} selected")
        
        return selected


def display_summary_stats(df, column, value_counts):
    """Display summary statistics"""
    st.subheader("Summary Statistics")
    
    total_count = len(df)
    unique_categories = df[column].nunique()
    
    st.metric("Total Records", f"{total_count:,}")
    st.metric("Unique Categories", f"{unique_categories}")
    
    st.subheader("Count Summary")
    
    freq_data = pd.DataFrame({
        column: value_counts.index,
        'Count': value_counts.values
    })
    
    if len(freq_data) > 10:
        freq_data = freq_data.head(10)
        st.info(f"Showing top 10 out of {len(value_counts)} categories")
    
    freq_table = freq_data.set_index(column).T
    st.dataframe(freq_table, use_container_width=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application"""
    
    apply_custom_css()
    display_header()
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        data = load_data(DATA_FILE_PATH)
    
    if data is None or data.empty:
        st.error(f"❌ Failed to load data from: {DATA_FILE_PATH}")
        st.info("Please check file path and permissions")
        st.stop()
    
    # Store data info
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
        
        selected_column = st.selectbox(
            "Select Category to Analyze",
            categorical_columns,
            index=0
        )
        
        st.markdown("---")
        
        # Sort and Orientation on same line
        col_sort, col_orient = st.columns(2)
        
        with col_sort:
            sort_by = st.radio(
                "Sort By",
                ["Name", "Count"],
                index=0,
                help="Sort by category name or frequency"
            )
        
        with col_orient:
            orientation = st.radio(
                "Orientation",
                ["vertical", "horizontal"],
                help="Chart layout direction"
            )
        
        st.markdown("---")
        
        selected_terms = create_multiselect_filter(
            data, "Academic Period", "Academic Periods", "terms"
        )
        
        st.markdown("---")
        
        selected_majors = create_multiselect_filter(
            data, "Major", "Majors", "majors"
        )
        
        st.markdown("---")
        
        show_percentages = st.checkbox("Show Percentages", value=False)
    
    # =============================================================================
    # FILTER DATA
    # =============================================================================
    
    if not selected_terms or not selected_majors:
        st.warning("⚠️ Please select at least one term and one major")
        st.stop()
    
    filtered_data = data[
        data["Academic Period"].isin(selected_terms) & 
        data["Major"].isin(selected_majors)
    ]
    
    if filtered_data.empty:
        st.warning("⚠️ No data matches the selected filters")
        st.stop()
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION
    # =============================================================================
    
    display_data = filtered_data
    
    if len(filtered_data) > LARGE_DATASET_THRESHOLD:
        st.info(f"ℹ️ Dataset has {len(filtered_data):,} records")
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
        value_counts = get_value_counts(display_data, selected_column, sort_by)
        
        # Limit categories
        if len(value_counts) > MAX_CATEGORIES_DISPLAY:
            st.info(f"ℹ️ Showing top {MAX_CATEGORIES_DISPLAY} categories")
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
            chart_title = f"Distribution of {selected_column} (%)"
        else:
            y_col = 'Count'
            chart_title = f"{selected_column} Distribution"
        
        # Create chart
        fig = create_bar_chart(
            chart_data,
            selected_column,
            y_col,
            orientation,
            chart_title
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
                textposition='outside' if orientation == 'vertical' else 'inside',
                textfont=dict(size=12)
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        display_summary_stats(display_data, selected_column, value_counts)
    
    # =============================================================================
    # CROSS-CATEGORY ANALYSIS
    # =============================================================================
    
    st.markdown("---")
    st.subheader("Cross-Category Analysis")
    
    if len(display_data) > LARGE_DATASET_THRESHOLD:
        st.warning(
            f"⚠️ Cross-category analysis disabled for large datasets. "
            "Please use filters or enable sampling."
        )
        st.stop()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        category_1 = st.selectbox("First Category", categorical_columns, index=0)
    
    with col_b:
        if len(categorical_columns) > 1:
            default_idx = 1
        else:
            st.warning("⚠️ Need at least 2 categorical columns")
            st.stop()
        category_2 = st.selectbox("Second Category", categorical_columns, index=default_idx)
    
    if category_1 == category_2:
        st.info("ℹ️ Please select two different categories")
        st.stop()
    
    # Check size
    unique_cat1 = display_data[category_1].nunique()
    unique_cat2 = display_data[category_2].nunique()
    
    if unique_cat1 * unique_cat2 > MAX_CROSSTAB_CELLS:
        st.warning(
            f"⚠️ Cross-tabulation too large ({unique_cat1} × {unique_cat2}). "
            "Please select categories with fewer unique values."
        )
        st.stop()
    
    # Create visualizations
    col3, col4 = st.columns(2)
    
    with col3:
        crosstab = get_crosstab(display_data, category_1, category_2)
        
        # Limit display
        if len(crosstab.index) > 10:
            crosstab = crosstab.head(10)
            st.info("ℹ️ Showing top 10 categories")
        
        if len(crosstab.columns) > 10:
            crosstab = crosstab.iloc[:, :10]
            st.info("ℹ️ Showing top 10 subcategories")
        
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
            legend_title=category_1,
            yaxis=dict(showticklabels=True, showgrid=True),
            xaxis=dict(showticklabels=True)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Heatmap
        fig_heatmap = create_heatmap(crosstab, category_1, category_2)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col4:
        st.subheader("Cross-Tabulation Data")
        
        if crosstab.size > 100:
            st.info("ℹ️ Showing preview")
            st.dataframe(crosstab.iloc[:5, :5], use_container_width=True)
        else:
            st.dataframe(crosstab, use_container_width=True)
        
        st.subheader("Summary")
        st.write(f"**Categories in {category_1}:** {len(crosstab.index)}")
        st.write(f"**Categories in {category_2}:** {len(crosstab.columns)}")
        st.write(f"**Total Combinations:** {crosstab.sum().sum():,}")


if __name__ == "__main__":
    main()
