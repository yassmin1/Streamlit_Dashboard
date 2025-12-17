"""
Student Insights Dashboard v7
Fixed version ensuring proper rendering on normal servers
All visualizations verified to display correctly with proper axis labels and heatmap values
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# Set page config FIRST before any other Streamlit commands
# This must be the very first Streamlit command in the script
# =============================================================================
st.set_page_config(
    page_title="Student Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Define the categorical columns that will be analyzed in the dashboard
# These are the fields available for selection in the dropdown menus
ANALYSIS_COLUMNS = [
    "Academic Period",  # Term or semester identifier
    "Gender",           # Student gender category
    "Student Type",     # Type of student (e.g., undergraduate, graduate)
    "Full_Part Time",   # Enrollment status (full-time or part-time)
    "Ethnicity",        # Student ethnicity category
    "Major",            # Academic program or major
    "Age_Group",        # Age range category
]

# Color scheme for consistent branding throughout the dashboard
COLORS = {
    "primary": "#34568B",      # Royal blue - used for titles and main elements
    "secondary": "#FFB347",    # Light orange - used for sidebar and accents
}

# Performance thresholds to optimize dashboard responsiveness
LARGE_DATASET_THRESHOLD = 110000  # Number of records above which to offer sampling
MAX_CATEGORIES_DISPLAY = 20       # Maximum number of categories to show in charts
MAX_CROSSTAB_CELLS = 1000        # Maximum cells in cross-tabulation to prevent slowdowns
SAMPLE_SIZE = 10000              # Sample size when sampling is enabled

# File paths - can be overridden by environment variables for deployment flexibility
# This allows the same code to work in different server environments
DATA_FILE_PATH = os.environ.get('DATA_FILE_PATH', 'data/curated/data_merged.parquet')
ASSETS_PATH = os.environ.get('ASSETS_PATH', 'assets')

# =============================================================================
# DATA LOADING AND CACHING FUNCTIONS
# These functions use Streamlit's caching to avoid reloading data unnecessarily
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data(file_path, columns=ANALYSIS_COLUMNS):
    """
    Load and preprocess student data from parquet file
    
    This function is cached to improve performance - it only runs once per session
    unless the input parameters change
    
    Args:
        file_path: Path to the parquet data file
        columns: List of column names to load (default: ANALYSIS_COLUMNS)
    
    Returns:
        pandas DataFrame with cleaned data, or None if loading fails
    """
    try:
        # Load only the columns we need to save memory
        df = pd.read_parquet(file_path, columns=columns)
        
        # Clean the data: convert all to string type and strip whitespace
        # This ensures consistency in categorical values
        df = df.astype(str).apply(lambda x: x.str.strip())
        
        return df
    except FileNotFoundError:
        # File doesn't exist at the specified path
        st.error(f"Data file not found: {file_path}")
        return None
    except Exception as e:
        # Catch any other errors (permissions, corrupt file, etc.)
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def get_value_counts(df, column, sort_by="Name"):
    """
    Calculate frequency counts for a categorical column with sorting options
    
    This function is cached to avoid recalculating when users change 
    visualization options without changing the underlying data
    
    Args:
        df: Source DataFrame
        column: Name of the column to count
        sort_by: Sorting method - "Name" for alphabetical, "Count" for frequency
    
    Returns:
        pandas Series with value counts, sorted according to sort_by parameter
    """
    # Get frequency counts for the column
    value_counts = df[column].value_counts()
    
    # Sort based on user preference
    if sort_by == "Name":
        # Alphabetical order by category name (index)
        return value_counts.sort_index()
    else:
        # Numerical order by frequency (highest to lowest)
        return value_counts.sort_values(ascending=False)


@st.cache_data
def get_crosstab(df, col1, col2):
    """
    Generate cross-tabulation between two categorical columns
    
    Cross-tabulation shows the relationship between two variables
    by counting occurrences of each combination
    
    Args:
        df: Source DataFrame
        col1: First categorical column name
        col2: Second categorical column name
    
    Returns:
        pandas DataFrame containing the cross-tabulation
    """
    return pd.crosstab(df[col1], df[col2])


# =============================================================================
# STYLING FUNCTIONS
# =============================================================================

def apply_custom_css():
    """
    Apply custom CSS styling to the dashboard
    
    This function injects CSS to customize the appearance of the sidebar
    and main content area. The styling ensures consistent branding with
    the school colors throughout the interface.
    """
    st.markdown(
        f"""
        <style>
        /* Sidebar background color - Royal Blue */
        [data-testid="stSidebar"] {{
            background-color: {COLORS['primary']};
        }}
        
        /* All text elements in sidebar - Light Orange */
        [data-testid="stSidebar"] * {{
            color: {COLORS['secondary']} !important;
        }}
        
        /* Make all text in main content area dark blue for consistency */
        .main * {{
            color: {COLORS['primary']} !important;
        }}
        
        /* Exception: keep chart elements their natural colors */
        .js-plotly-plot * {{
            color: initial !important;
        }}
        
        /* Keep dataframe text readable */
        .dataframe * {{
            color: #262730 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# These functions create the charts and graphs displayed in the dashboard
# =============================================================================

def create_bar_chart(data, x_col, y_col, orientation="vertical", title=""):
    """
    Create a bar chart with explicit axis configuration for server compatibility
    
    This function ensures that axis labels and values display correctly even on
    older server environments by explicitly configuring all axis properties.
    
    Args:
        data: DataFrame containing the data to plot
        x_col: Column name for x-axis
        y_col: Column name for y-axis (usually 'Count' or 'Percentage')
        orientation: 'vertical' or 'horizontal' bar layout
        title: Chart title string
    
    Returns:
        Plotly figure object ready to display
    """
    
    # Create the base bar chart using Plotly Express
    if orientation == "horizontal":
        # Horizontal bars: x = values, y = categories
        fig = px.bar(
            data,
            x=y_col,
            y=x_col,
            orientation="h",
            title=title,
            color=y_col,
            color_continuous_scale="Viridis"
        )
        
        # EXPLICIT axis configuration - critical for server compatibility
        # Without these explicit settings, some servers may not display axis labels
        fig.update_xaxes(
            title=y_col,           # Show the column name as axis title
            showticklabels=True,   # Force tick labels to display
            showgrid=True          # Show gridlines for easier reading
        )
        fig.update_yaxes(
            title=x_col,           # Show the column name as axis title
            showticklabels=True,   # Force tick labels to display
            type='category'        # Treat as categorical data (not numeric)
        )
    else:
        # Vertical bars: x = categories, y = values
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=y_col,
            color_continuous_scale="Viridis"
        )
        
        # EXPLICIT axis configuration for vertical orientation
        fig.update_xaxes(
            title=x_col,           # Show the column name as axis title
            showticklabels=True,   # Force tick labels to display
            type='category'        # Treat as categorical data
        )
        fig.update_yaxes(
            title=y_col,           # Show the column name as axis title
            showticklabels=True,   # Force tick labels to display
            showgrid=True,         # Show gridlines for easier reading
            rangemode='tozero'     # Ensure y-axis starts at 0
        )
    
    # Apply consistent layout styling
    # Using white backgrounds instead of transparent for better server compatibility
    fig.update_layout(
        title_font_size=16,              # Readable title size
        height=450,                      # Fixed height for consistency
        showlegend=False,                # Hide color legend to save space
        plot_bgcolor='white',            # White plot background
        paper_bgcolor='white',           # White paper background
        xaxis=dict(
            showline=True,               # Show axis line
            showticklabels=True,         # Force tick labels
            linecolor='rgb(204, 204, 204)',  # Light gray axis line
            linewidth=2                  # Visible line width
        ),
        yaxis=dict(
            showline=True,               # Show axis line
            showticklabels=True,         # Force tick labels
            showgrid=True,               # Show gridlines
            linecolor='rgb(204, 204, 204)',  # Light gray axis line
            linewidth=2,                 # Visible line width
            gridcolor='rgb(230, 230, 230)'   # Very light gray gridlines
        ),
        margin=dict(l=60, r=20, t=50, b=60)  # Margins to ensure labels fit
    )
    
    return fig


def create_heatmap(crosstab, row_label, col_label):
    """
    Create a heatmap with explicit text annotations for server compatibility
    
    This function uses manual text annotations instead of relying on automatic
    text rendering, which ensures values display correctly on all servers.
    
    Args:
        crosstab: Cross-tabulation DataFrame (output from pd.crosstab)
        row_label: Label for rows (y-axis)
        col_label: Label for columns (x-axis)
    
    Returns:
        Plotly figure object with heatmap and annotations
    """
    
    # Create a copy to avoid modifying the original data
    crosstab_copy = crosstab.copy()
    
    # Convert index and columns to strings for proper label display
    crosstab_copy.columns = crosstab_copy.columns.astype(str)
    crosstab_copy.index = crosstab_copy.index.astype(str)
    
    # Create manual text annotations for each cell
    # This ensures numbers display even if automatic text rendering fails
    annotations = []
    for i, row in enumerate(crosstab_copy.index):
        for j, col in enumerate(crosstab_copy.columns):
            value = crosstab_copy.iloc[i, j]
            
            # Determine text color based on cell value for readability
            # Light text on dark cells, dark text on light cells
            text_color = 'white' if value > crosstab_copy.values.max()/2 else 'black'
            
            # Create annotation dictionary for this cell
            annotations.append(
                dict(
                    x=j,                    # Column position
                    y=i,                    # Row position
                    text=str(int(value)),   # Display value as integer string
                    showarrow=False,        # Don't show arrows pointing to text
                    font=dict(size=12, color=text_color)  # Text styling
                )
            )
    
    # Create heatmap using graph_objects for more control than px.imshow
    fig = go.Figure(data=go.Heatmap(
        z=crosstab_copy.values,              # The actual count values
        x=crosstab_copy.columns.tolist(),    # Column labels
        y=crosstab_copy.index.tolist(),      # Row labels
        colorscale='Viridis',                # Color scheme
        hoverongaps=False,                   # Don't show hover on empty cells
        hovertemplate=f'{row_label}: %{{y}}<br>{col_label}: %{{x}}<br>Count: %{{z}}<extra></extra>'
    ))
    
    # Add the text annotations to the heatmap
    fig.update_layout(
        annotations=annotations,             # Apply all cell annotations
        title=f"{row_label} vs {col_label} (Heatmap)",
        xaxis_title=col_label,
        yaxis_title=row_label,
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=100, r=20, t=50, b=100)  # Extra margins for labels
    )
    
    # Ensure axes display correctly
    fig.update_xaxes(
        showticklabels=True,    # Show x-axis labels
        type='category',        # Treat as categories
        side='bottom'           # Labels on bottom
    )
    fig.update_yaxes(
        showticklabels=True,    # Show y-axis labels
        type='category',        # Treat as categories
        autorange='reversed'    # Top to bottom ordering
    )
    
    return fig


# =============================================================================
# UI COMPONENT FUNCTIONS
# These functions create reusable interface elements
# =============================================================================

def display_header():
    """
    Display the dashboard header with responsive logo and title
    
    The header includes:
    - Logo image (scaled responsively to fit available space)
    - Main title
    - Descriptive subtitle explaining dashboard purpose
    
    All text sizes scale automatically based on viewport width using CSS clamp()
    """
    
    # Create two columns: left for logo, right for text
    # Ratio 1:5 gives more space to text while keeping logo visible
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Attempt to load and display the logo
        logo_path = Path(ASSETS_PATH) / "analysis.png"
        if logo_path.exists():
            try:
                # use_column_width=True makes logo scale responsively
                # It will shrink on small screens and grow on large screens
                # This prevents the logo from covering text
                st.image(str(logo_path), use_column_width=True)
            except:
                # If image loading fails, show a simple icon instead
                st.info("Data Analytics")
        else:
            # If logo file doesn't exist, show a simple icon
            st.info("Data Analytics")
    
    with col2:
        # Main title with responsive font sizing
        # clamp(min, preferred, max) ensures readable text at all screen sizes
        st.markdown(
            f"""
            <h1 style='color:{COLORS['primary']}; 
                       font-size:clamp(40px, 5vw, 70px); 
                       text-align:center; 
                       margin-top:-20px; 
                       line-height:1.2;'>
                Student Insights
            </h1>
            """,
            unsafe_allow_html=True
        )
        
        # Subtitle with full description and responsive sizing
        # Text wraps naturally and scales down on smaller screens
        st.markdown(
            f"""
            <div style='color:{COLORS['primary']}; 
                        font-size:clamp(14px, 1.8vw, 24px); 
                        text-align:center; 
                        line-height:1.4; 
                        margin-top:-5px; 
                        padding:0 10px;'>
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
    
    # Horizontal line separator
    st.markdown("---")


def create_multiselect_filter(df, column, label, key_prefix):
    """
    Create a multiselect filter widget with 'Select All' functionality
    
    This creates a user-friendly filter that allows selecting multiple values
    from a categorical column, with a convenient 'Select All' checkbox.
    
    Args:
        df: Source DataFrame
        column: Column name to create filter for
        label: Display label for the filter
        key_prefix: Unique prefix for session state keys (prevents conflicts)
    
    Returns:
        List of selected values (strings)
    """
    
    # Get unique values from the column and prepare them
    # Remove any null values, convert to string, strip whitespace, remove empties
    options = sorted(
        df[column]
        .dropna()                          # Remove null/NaN values
        .astype(str)                       # Convert everything to string
        .str.strip()                       # Remove leading/trailing whitespace
        .loc[lambda s: s.ne("")]          # Remove empty strings
        .unique()                          # Get unique values only
        .tolist()                          # Convert to list
    )
    
    # Display section header
    st.subheader(label)
    
    # Create unique keys for session state
    # These keys persist user selections across reruns
    select_all_key = f"{key_prefix}_select_all"
    multi_key = f"{key_prefix}_multiselect"
    
    # Checkbox to select/deselect all options at once
    select_all = st.checkbox(
        f"Select all {label.lower()}", 
        value=True,              # Default to all selected
        key=select_all_key
    )
    
    if select_all:
        # When 'Select All' is checked, show a disabled multiselect
        # This provides visual feedback that all items are selected
        st.multiselect(
            label,
            options=options,
            default=options,         # All options selected
            key=multi_key,
            disabled=True,           # Disabled to show it's controlled by checkbox
            help=f"All {label.lower()} are included."
        )
        return options              # Return all options
    else:
        # When 'Select All' is unchecked, show an active multiselect
        # Users can choose specific items
        
        # Try to restore previous selection from session state
        default_vals = st.session_state.get(multi_key, options)
        
        selected = st.multiselect(
            label,
            options=options,
            default=default_vals,
            key=multi_key,
            help=f"Uncheck 'Select all {label.lower()}' to filter."
        )
        
        # Warn if nothing is selected
        if len(selected) == 0:
            st.warning(f"No {label.lower()} selected")
        
        return selected


def display_summary_stats(df, column, value_counts):
    """
    Display summary statistics for the selected category
    
    Shows key metrics and a frequency table in the sidebar
    
    Args:
        df: Source DataFrame (filtered)
        column: Column being analyzed
        value_counts: Series with value counts for the column
    """
    
    # Section header
    st.subheader("Summary Statistics")
    
    # Calculate statistics
    total_count = len(df)                # Total number of records
    unique_categories = df[column].nunique()  # Number of unique values
    
    # Display metrics using Streamlit's metric widget
    # The :, format adds thousand separators (e.g., 1,234)
    st.metric("Total Records", f"{total_count:,}")
    st.metric("Unique Categories", f"{unique_categories}")
    
    # Frequency table section
    st.subheader("Count Summary")
    
    # Prepare frequency data as a DataFrame
    freq_data = pd.DataFrame({
        column: value_counts.index,      # Category names
        'Count': value_counts.values     # Frequency counts
    })
    
    # Limit table size for performance and readability
    if len(freq_data) > 10:
        freq_data = freq_data.head(10)   # Keep only top 10
        st.info(f"Showing top 10 out of {len(value_counts)} categories")
    
    # Transpose the table for better horizontal display
    # This puts categories in columns and counts in a single row
    freq_table = freq_data.set_index(column).T
    
    # Display the transposed table
    st.dataframe(freq_table,width='stretch')


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point
    
    This function orchestrates the entire dashboard:
    1. Loads and validates data
    2. Creates sidebar controls
    3. Applies filters
    4. Generates visualizations
    5. Displays cross-category analysis
    """
    
    # Apply custom CSS styling
    apply_custom_css()
    
    # Display the header section
    display_header()
    
    # =============================================================================
    # DATA LOADING SECTION
    # =============================================================================
    
    # Show a loading spinner while data is being loaded
    # The spinner automatically disappears when loading completes
    with st.spinner("Loading data..."):
        data = load_data(DATA_FILE_PATH)
    
    # Validate that data loaded successfully
    if data is None or data.empty:
        # If data loading failed, show error and stop execution
        st.error(f"Failed to load data from: {DATA_FILE_PATH}")
        st.info("Please check file path and permissions")
        st.stop()  # Stop execution here - don't try to continue without data
    
    # Store frequently-used data info in session state to avoid recalculating
    # Session state persists across reruns but is cleared when session ends
    if 'data_info' not in st.session_state:
        st.session_state.data_info = {
            'total_records': len(data),  # Total number of rows
            'columns': data.columns.tolist(),  # All column names
            # Get only categorical columns (object/category dtypes)
            'categorical_columns': data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
        }
    
    # Extract categorical columns from session state for easy access
    categorical_columns = st.session_state.data_info['categorical_columns']
    
    # =============================================================================
    # SIDEBAR CONTROLS SECTION
    # Create all user input controls in the sidebar
    # =============================================================================
    
    with st.sidebar:
        # Sidebar header with custom styling
        st.markdown(
            f"<h1 style='color:{COLORS['secondary']}; font-size:35px; "
            f"text-align:center; margin-top:0px'>Analysis Controls</h1>",
            unsafe_allow_html=True
        )
        
        # Dropdown to select which categorical column to analyze
        selected_column = st.selectbox(
            "Select Category to Analyze",
            categorical_columns,
            index=0  # Default to first column
        )
        
        st.markdown("---")  # Visual separator
        
        # Sort and Orientation controls side-by-side to save space
        col_sort, col_orient = st.columns(2)
        
        with col_sort:
            # Radio buttons for sorting preference
            sort_by = st.radio(
                "Sort By",
                ["Name", "Count"],
                index=0,  # Default to Name (alphabetical)
                help="Sort by category name or frequency"
            )
        
        with col_orient:
            # Radio buttons for chart orientation
            orientation = st.radio(
                "Orientation",
                ["vertical", "horizontal"],
                help="Chart layout direction"
            )
        
        st.markdown("---")
        
        # Academic Period filter with Select All functionality
        selected_terms = create_multiselect_filter(
            data, 
            "Academic Period",  # Column to filter
            "Academic Periods",  # Display label
            "terms"             # Unique key prefix
        )
        
        st.markdown("---")
        
        # Major filter with Select All functionality
        selected_majors = create_multiselect_filter(
            data, 
            "Major",     # Column to filter
            "Majors",    # Display label
            "majors"     # Unique key prefix
        )
        
        st.markdown("---")
        
        # Checkbox to toggle between raw counts and percentages
        show_percentages = st.checkbox("Show Percentages", value=False)
    
    # =============================================================================
    # DATA FILTERING SECTION
    # Apply user-selected filters to the dataset
    # =============================================================================
    
    # Validate that at least one option is selected in each filter
    if not selected_terms or not selected_majors:
        st.warning("Please select at least one term and one major")
        st.stop()  # Stop here if no selections made
    
    # Apply filters: keep only rows matching selected terms AND majors
    filtered_data = data[
        data["Academic Period"].isin(selected_terms) & 
        data["Major"].isin(selected_majors)
    ]
    
    # Check if any data remains after filtering
    if filtered_data.empty:
        st.warning("No data matches the selected filters")
        st.stop()  # Stop here if no data left
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION SECTION
    # Handle large datasets by offering sampling option
    # =============================================================================
    
    # Start with the filtered data
    display_data = filtered_data
    
    # If dataset is very large, offer sampling to improve performance
    if len(filtered_data) > LARGE_DATASET_THRESHOLD:
        st.info(f"Dataset has {len(filtered_data):,} records")
        
        # Checkbox to enable/disable sampling
        use_sampling = st.checkbox(
            "Use sampling for faster processing", 
            value=False
        )
        
        if use_sampling:
            # Take a random sample of the data
            sample_size = min(SAMPLE_SIZE, len(filtered_data))
            display_data = filtered_data.sample(n=sample_size, random_state=42)
            st.info(f"Using sample of {sample_size:,} records")
    
    # =============================================================================
    # MAIN VISUALIZATION SECTION
    # Create and display the primary bar chart
    # =============================================================================
    
    # Create two columns: left for chart (wider), right for stats (narrower)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get value counts for the selected column with chosen sorting
        value_counts = get_value_counts(display_data, selected_column, sort_by)
        
        # Limit number of categories displayed for readability and performance
        if len(value_counts) > MAX_CATEGORIES_DISPLAY:
            st.info(f"Showing top {MAX_CATEGORIES_DISPLAY} categories")
            value_counts = value_counts.head(MAX_CATEGORIES_DISPLAY)
        
        # Convert value counts to DataFrame format for Plotly
        chart_data = pd.DataFrame({
            selected_column: value_counts.index,   # Category names
            'Count': value_counts.values           # Frequency counts
        })
        
        # Determine what to display based on percentage toggle
        if show_percentages:
            # Calculate percentages and add as new column
            chart_data['Percentage'] = (
                chart_data['Count'] / chart_data['Count'].sum()
            ) * 100
            y_col = 'Percentage'  # Display percentages on y-axis
            chart_title = f"Distribution of {selected_column} (%)"
        else:
            # Display raw counts
            y_col = 'Count'
            chart_title = f"{selected_column} Distribution"
        
        # Create the bar chart
        fig = create_bar_chart(
            chart_data,
            selected_column,  # x-axis: category names
            y_col,           # y-axis: counts or percentages
            orientation,     # vertical or horizontal
            chart_title      # chart title
        )
        
        # Add text labels on bars for small datasets (better readability)
        if len(chart_data) <= 15:
            if show_percentages:
                # Show both count and percentage on bars
                text_values = [
                    f"{count}<br>({pct:.1f}%)" 
                    for count, pct in zip(
                        chart_data['Count'], 
                        chart_data['Percentage']
                    )
                ]
            else:
                # Show only counts on bars
                text_values = chart_data['Count'].astype(str)
            
            # Add text to bars
            fig.update_traces(
                text=text_values,
                textposition='outside' if orientation == 'vertical' else 'inside',
                textfont=dict(size=12)
            )
        
        # Display the chart - use_container_width makes it responsive
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Display summary statistics in the sidebar
        display_summary_stats(display_data, selected_column, value_counts)
    
    # =============================================================================
    # CROSS-CATEGORY ANALYSIS SECTION
    # Compare relationships between two different categorical variables
    # =============================================================================
    
    # Visual separator and section header
    st.markdown("---")
    st.subheader("Cross-Category Analysis")
    
    # Disable cross-analysis for very large datasets to maintain performance
    if len(display_data) > LARGE_DATASET_THRESHOLD:
        st.warning(
            "Cross-category analysis disabled for large datasets. "
            "Please use filters or enable sampling."
        )
        st.stop()  # Stop here - don't proceed with cross-analysis
    
    # Create two columns for selecting the categories to compare
    col_a, col_b = st.columns(2)
    
    with col_a:
        # First category dropdown
        category_1 = st.selectbox(
            "First Category", 
            categorical_columns, 
            index=0
        )
    
    with col_b:
        # Second category dropdown
        # Ensure we have at least 2 categorical columns
        if len(categorical_columns) > 1:
            default_idx = 1  # Select second column by default
        else:
            st.warning("Need at least 2 categorical columns")
            st.stop()
        
        category_2 = st.selectbox(
            "Second Category", 
            categorical_columns, 
            index=default_idx
        )
    
    # Prevent selecting the same category twice
    if category_1 == category_2:
        st.info("Please select two different categories")
        st.stop()
    
    # Check if cross-tabulation would be too large
    # Large cross-tabs can slow down the dashboard significantly
    unique_cat1 = display_data[category_1].nunique()  # Number of unique values
    unique_cat2 = display_data[category_2].nunique()  # Number of unique values
    
    # Total cells in cross-tab would be unique_cat1 × unique_cat2
    if unique_cat1 * unique_cat2 > MAX_CROSSTAB_CELLS:
        st.warning(
            f"Cross-tabulation too large ({unique_cat1} × {unique_cat2}). "
            "Please select categories with fewer unique values."
        )
        st.stop()
    
    # Create visualizations in two columns
    col3, col4 = st.columns(2)
    
    with col3:
        # Generate cross-tabulation
        crosstab = get_crosstab(display_data, category_1, category_2)
        
        # Limit display size for performance
        # Only show top categories if there are too many
        if len(crosstab.index) > 10:
            crosstab = crosstab.head(10)  # Keep top 10 rows
            st.info("Showing top 10 categories")
        
        if len(crosstab.columns) > 10:
            crosstab = crosstab.iloc[:, :10]  # Keep top 10 columns
            st.info("Showing top 10 subcategories")
        
        # Create stacked bar chart
        # Transpose (.T) so categories become x-axis
        fig_bar = px.bar(
            crosstab.T,
            title=f"{category_1} vs {category_2} Distribution",
            height=400
        )
        
        # Configure axes and legend
        fig_bar.update_layout(
            xaxis_title=category_2,      # X-axis shows second category
            yaxis_title="Count",         # Y-axis shows counts
            showlegend=True,             # Show legend for first category
            legend_title=category_1,     # Legend title
            yaxis=dict(
                showticklabels=True,     # Show y-axis labels
                showgrid=True            # Show gridlines
            ),
            xaxis=dict(
                showticklabels=True      # Show x-axis labels
            )
        )
        
        # Display the stacked bar chart
        st.plotly_chart(fig_bar, width='stretch')
        
        # Create and display heatmap
        fig_heatmap = create_heatmap(crosstab, category_1, category_2)
        st.plotly_chart(fig_heatmap, width='stretch')
    
    with col4:
        # Display the raw cross-tabulation data
        st.subheader("Cross-Tabulation Data")
        
        # For large tables, show only a preview
        if crosstab.size > 100:  # If more than 100 cells
            st.info("Showing preview")
            # Display only top-left 5x5 portion
            st.dataframe(crosstab.iloc[:5, :5], width='stretch')
        else:
            # Display full table if reasonably sized
            st.dataframe(crosstab, width='stretch')
        
        # Display summary statistics about the cross-tabulation
        st.subheader("Summary")
        st.write(f"**Categories in {category_1}:** {len(crosstab.index)}")
        st.write(f"**Categories in {category_2}:** {len(crosstab.columns)}")
        st.write(f"**Total Combinations:** {crosstab.sum().sum():,}")


# =============================================================================
# SCRIPT ENTRY POINT
# This code only runs when the script is executed directly
# =============================================================================

if __name__ == "__main__":
    main()
