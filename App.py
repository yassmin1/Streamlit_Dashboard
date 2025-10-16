#https://student-insights.streamlit.app/
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from pathlib import Path
import json 
#from etl.etl_cbmc1 import  ETLPaths, run_etl

##############################################
COLUMNS=[
"Academic Period",
# "Calendar Year",
"Gender",
"Student Type",
"Full_Part Time",
"Ethnicity",
"Major",
"Age_Group",

]


def create_bar_chart(data, x_col, y_col, title="Bar Chart", color_col=None, 
                     orientation='vertical', color_scheme='viridis'):
    """
    Create a customizable bar chart for Streamlit dashboard

    Parameters:
    - data: DataFrame with the data
    - x_col: Column name for x-axis
    - y_col: Column name for y-axis (values)
    - title: Chart title
    - color_col: Column name for color grouping (optional)
    - orientation: 'vertical' or 'horizontal'
    - color_scheme: Color palette ('viridis', 'blues', 'reds', etc.)
    """

    if orientation == 'horizontal':
        fig = px.bar(data, x=y_col, y=x_col, 
                     color=color_col if color_col else None,
                     orientation='h',
                     title=title,
                     color_discrete_sequence=px.colors.qualitative.Set3)
    else:
        fig = px.bar(data, x=x_col, y=y_col, 
                     color=color_col if color_col else None,
                     title=title,
                     color_discrete_sequence=px.colors.qualitative.Set3)

    # Customize the layout
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,  # Center the title
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        font_size=12,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )

    # Add hover information
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
    )

    return fig
###
def create_stacked_bar_chart(data, x_col, y_cols, title="Stacked Bar Chart"):
    """Create a stacked bar chart with multiple y-columns"""
    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Bar(
            name=col,
            x=data[x_col],
            y=data[col]
        ))

    fig.update_layout(
        title=title,
        barmode='stack',
        height=500
    )

    return fig

def create_grouped_bar_chart(data, x_col, y_cols, title="Grouped Bar Chart"):
    """Create a grouped bar chart with multiple y-columns"""
    fig = go.Figure()

    for col in y_cols:
        fig.add_trace(go.Bar(
            name=col,
            x=data[x_col],
            y=data[col]
        ))

    fig.update_layout(
        title=title,
        barmode='group',
        height=500
    )

def build_heatmap(ct: pd.DataFrame, index_col: str, col_col: str,text_font_size:int=12,
                  ) -> px.imshow:
    ct.columns = ct.columns.astype(str)
    ct.index = ct.index.astype(str)
    fig = px.imshow(
        ct.values,
        x=ct.columns,
        y=ct.index,
        aspect="auto",
        title=f"{index_col} vs {col_col} (Heatmap)",
        labels=dict(x=col_col, y=index_col, color="Count")
    )
    
        
    text = ct.round(0).astype(int).astype(str)
    hover_fmt = "%{z:.0f}"
    fig.update_xaxes(categoryorder="array", categoryarray=ct.columns)
    fig.update_yaxes(categoryorder="array", categoryarray=ct.index)
    fig.update_traces(
        text=text.values,
        texttemplate="%{text}",
        textfont={"size": text_font_size},
        hovertemplate=f"{index_col}=%{{y}}<br>{col_col}=%{{x}}<br>Value={hover_fmt}<extra></extra>",
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_layout(
        height=450,
        xaxis_title=col_col,  # X-axis shows second category
        yaxis_title=index_col,     # Y-axis shows counts
        showlegend=True,         # Show legend for first category
        #legend_title=category_1  # Legend title is first category
            )
    return fig
  
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
##

# =============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE OPTIMIZATION
# =============================================================================

# Cache the data loading function to avoid reloading data every time the app runs
@st.cache_data
def load_and_process_data(file_par,columns=COLUMNS):
    """
    Load and process CBM data with caching for performance

    The @st.cache_data decorator ensures this function only runs once per session.
    Subsequent calls will return the cached result, dramatically improving performance.

    Returns:
        pd.DataFrame: Processed CBM data or None if loading fails
    """
    try:

        sample_data=(pd.read_parquet(file_par)[columns])
        sample_data=sample_data.astype(str).apply(lambda x: x.str.strip())
        #sample_data['Calendar Year'] = "Year"+sample_data['Calendar Year']



        return sample_data
    except Exception as e:
        # Display error message to user if data loading fails
        st.error(f"Error loading data: {str(e)}")
        return None

# Cache the value counts calculation to avoid recalculating for the same column
@st.cache_data
def get_value_counts(data, column):
    """
    Get value counts for a specific column with caching

    This function calculates how many times each unique value appears in a column.
    Caching prevents recalculation when the user switches between different view options
    (like percentages vs counts) for the same column.

    Args:
        data (pd.DataFrame): The dataset to analyze
        column (str): Name of the column to count values for

    Returns:
        pd.Series: Value counts for the specified column
    """
    return data[column].value_counts()

# Cache cross-tabulation calculations to improve performance for category comparisons
@st.cache_data
def get_cross_tabulation(data, col1, col2):
    """
    Get cross-tabulation between two columns with caching

    Cross-tabulation shows the relationship between two categorical variables
    by counting occurrences of each combination. Caching this expensive operation
    prevents recalculation when users switch between visualization options.

    Args:
        data (pd.DataFrame): The dataset to analyze
        col1 (str): First categorical column
        col2 (str): Second categorical column

    Returns:
        pd.DataFrame: Cross-tabulation table showing relationships between categories
    """
    ct = pd.crosstab(data[col1], data[col2])
    #ct = ct[ct.sum(axis=0).sort_values(ascending=False).index] # sort columns based on sum

    return ct


def sidebar_major_selector(df: pd.DataFrame,
                           major_col: str = "Major",
                           title: str = "Majors",
                           key_prefix: str = "majors") -> list[str]:
    """Sidebar widget: 'Select All' + multiselect for majors.

    - De-dupes & sorts options
    - Preserves selection in session state
    - Disables the multiselect when 'Select All' is checked
    - Guards against empty selections
    """
    # Build stable, clean options
    majors = (
        df[major_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    majors = sorted(set(majors))  # de-dup + sort once

    with st.sidebar:
        st.subheader(title)
        # Select-all checkbox
        all_key = f"{key_prefix}_select_all"
        multi_key = f"{key_prefix}_multiselect"

        select_all = st.checkbox("Select all majors", value=True, key=all_key)

        if select_all:
            # Show disabled multiselect for clarity (everything selected)
            st.multiselect(
                "Majors",
                options=majors,
                default=majors,
                key=multi_key,
                disabled=True,
                help="All majors are included."
            )
            selected = majors
        else:
            # Use prior selection if exists; otherwise default to all
            default_vals = st.session_state.get(multi_key, majors)
            selected = st.multiselect(
                "Majors",
                options=majors,
                default=default_vals,
                key=multi_key,
                help="Uncheck 'Select all majors' to filter."
            )
            # Guard: if user clears everything, keep empty but warn
            if len(selected) == 0:
                st.info("No majors selected — results may be empty.")

    return selected

#----
def sidebar_Term_selector(df: pd.DataFrame,
                           term_col: str = "Academic Period",
                           title: str = "Academic Period",
                           key_prefix: str = "Academic Period") -> list[str]:
    """Sidebar widget: 'Select All' + multiselect for majors.

    - De-dupes & sorts options
    - Preserves selection in session state
    - Disables the multiselect when 'Select All' is checked
    - Guards against empty selections
    """
    # Build stable, clean options
    terms = (
        df[term_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    terms = sorted(set(terms))  # de-dup + sort once

    with st.sidebar:
        st.subheader(title)
        # Select-all checkbox
        all_key = f"{key_prefix}_select_all"
        multi_key = f"{key_prefix}_multiselect"

        select_all = st.checkbox("Select all Terms", value=True, key=all_key)

        if select_all:
            # Show disabled multiselect for clarity (everything selected)
            st.multiselect(
                "Terms",
                options=terms,
                default=terms,
                key=multi_key,
                disabled=True,
                help="All terms are included."
            )
            selected = terms
        else:
            # Use prior selection if exists; otherwise default to all
            default_vals = st.session_state.get(multi_key, terms)
            selected = st.multiselect(
                "Terms",
                options=terms,
                default=default_vals,
                key=multi_key,
                help="Uncheck 'Select all terms' to filter."
            )
            # Guard: if user clears everything, keep empty but warn
            if len(selected) == 0:
                st.info("No terms selected — results may be empty.")

    return selected


# =============================================================================
# MAIN DASHBOARD FUNCTION
# =============================================================================

def main():
    """
    Main function that creates the Streamlit dashboard for SPC data analysis.

    This function sets up the entire user interface, loads data, creates visualizations,
    and handles user interactions. It's optimized for performance with large datasets
    through caching, sampling, and smart data limiting.
    """

    # Configure the Streamlit page with title and wide layout for better space utilization
    st.set_page_config(page_title="Student Insights", layout="wide")
    col1, col2 = st.columns([1, 4])  
    with col1:
        # Display the SPC logo from local assets
        _safe_image("assets/analysis.png", width=180)

    with col2:     
        # Create the main title and separator line
        #st.title("Student Insights", anchor=None, help="Explore student enrollment, demographics, and program trends.",  )               )
        st.markdown("<h1 style='color:#34568B; font-size:70px; text-align:center;margin-top:-50px'>Student Insights</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='color:#34568B;font-size:28px; text-align:center;line-height:1.2 ;margin-top:-10px'>
              <strong>An interactive dashboard designed to explore student enrollment, demographics, and program trends. 
               Use the sidebar filters to select academic terms, majors, and student characteristics to uncover patterns that support data-informed decisions. </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")  # Creates a horizontal line for visual separation

    # =============================================================================
    # DATA LOADING SECTION
    # =============================================================================

    # Load data with caching - this spinner shows while data is loading
    # The spinner improves user experience by indicating that something is happening
    with st.spinner("Loading data..."):
        sample_data = load_and_process_data("data/curated/data_merged.parquet")

    # Check if data loading was successful
    if sample_data is None or sample_data.empty:
        # Display error message and stop execution if no data available
        st.error("Failed to load data. Please check the file path and try again.")
        return

    # =============================================================================
    # SESSION STATE OPTIMIZATION
    # =============================================================================

    # Store frequently used data information in session state to avoid repeated calculations
    # Session state persists across user interactions, improving performance
    if 'data_info' not in st.session_state:
        st.session_state.data_info = {
            'total_records': len(sample_data),  # Total number of rows in dataset
            'columns': sample_data.columns.tolist(),  # All column names
            # Only categorical columns (object/category types) are suitable for counting analysis
            'categorical_columns': sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
        }

    # Use cached categorical columns list for better performance
    categorical_columns = st.session_state.data_info['categorical_columns']

    # =============================================================================
    # SIDEBAR CONTROLS SECTION
    # =============================================================================
    # Add custom CSS to style sidebar
    st.markdown("""
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #34568B; /* royal blue */
        color:#FFB347 ; /* text color: light orange*/
    }

    /* Sidebar text (labels, markdown, etc.) */
    [data-testid="stSidebar"] * {
        color: #FFB347 !important;
    }

    /* Optional: sidebar title styling */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #FFB347 !important; /* light orange */
    }      

    </style>
    """, unsafe_allow_html=True)
    # Create sidebar for user controls - keeps main area clean for visualizations
    #st.sidebar.header("Analysis Controls")
    st.sidebar.markdown("<h1 style='color:#FFB347; font-size:35px; text-align:center;margin-top:0px'>Analysis Controls</h1>", unsafe_allow_html=True)


    # Dropdown to select which categorical column to analyze
    # Index=0 means the first column is selected by default
    selected_column = st.sidebar.selectbox(
        "Select Category to Analyze",
        categorical_columns,
        index=0
    )
    st.sidebar.markdown("==========================")
    st.sidebar.markdown(    "<hr style='margin:0px 0;'>",
    unsafe_allow_html=True)
    # Radio buttons for chart orientation - affects how bars are displayed
    orientation = st.sidebar.radio(
        "Chart Orientation",
        ["vertical", "horizontal"]  # vertical = bars go up, horizontal = bars go sideways
    )
    st.sidebar.markdown("==========================")

    #checkbox to taggle between Terms

    show_terms=sidebar_Term_selector(sample_data, term_col="Academic Period")
    st.sidebar.markdown("==========================")

    #checkbox to taggle between Majors
    show_majors = sidebar_major_selector(sample_data, major_col="Major")
    st.sidebar.markdown("==========================")


    # Checkbox to toggle between showing counts vs percentages
    show_percentages = st.sidebar.checkbox("Show Percentages", value=False)
    st.sidebar.markdown("==========================")
     #----------------------------
    #fiter based on select terms 
    #--------------------------------
    if show_terms:
        sample_data = sample_data[sample_data["Academic Period"].isin(show_terms) & sample_data['Major'].isin(show_majors)]
    else:
        st.warning("Please select at least one term to display data.")
        return
    #-----------------------

    # =============================================================================
    # PERFORMANCE OPTIMIZATION FOR LARGE DATASETS
    # =============================================================================

    # If dataset is very large (>110,000 records), offer sampling option for better performance
    if len(sample_data) > 110000:
        # Inform user about large dataset and optimization options
        st.sidebar.info(f"Dataset has {len(sample_data):,} records. Using optimized processing.")

        # Checkbox to enable/disable sampling
        use_sampling = st.sidebar.checkbox("Use sampling for faster processing", value=False)

        if use_sampling:
            # Use maximum of 10,000 records or the full dataset size, whichever is smaller
            sample_size = min(10000, len(sample_data))
            # random_state=42 ensures reproducible sampling
            display_data = sample_data.sample(n=sample_size, random_state=42)
            st.sidebar.info(f"Using sample of {sample_size:,} records")
        else:
            # Use full dataset if sampling is disabled
            display_data = sample_data
    else:
        # For smaller datasets, use all data
        display_data = sample_data



    # =============================================================================
    # MAIN DASHBOARD LAYOUT
    # =============================================================================

    # Create two columns: left for chart (2/3 width), right for statistics (1/3 width)
    col1, col2 = st.columns([2, 1])

    # =============================================================================
    # LEFT COLUMN: MAIN VISUALIZATION
    # =============================================================================

    with col1:
        # Get cached value counts for the selected column
        # This prevents recalculation when user changes visualization options
        value_counts = get_value_counts(display_data, selected_column)

        # Limit categories for performance - showing too many bars makes charts unreadable
        if len(value_counts) > 20:
            st.info(f"Showing top 20 categories out of {len(value_counts)} total")
            value_counts = value_counts.head(20)  # Keep only top 20 most frequent categories

        # Convert value counts to DataFrame format required by Plotly
        count_data = value_counts.reset_index()
        count_data.columns = [selected_column, 'Count']  # Rename columns for clarity

        # Calculate percentages if user requested them
        if show_percentages:
            # Calculate percentage of each category relative to total
            count_data['Percentage'] = (count_data['Count'] / count_data['Count'].sum()) * 100
            y_col = 'Percentage'  # Use percentage column for y-axis
            title = f"Distribution of {selected_column} (%)"
            # Custom hover template showing both count and percentage
            hover_template = '<b>%{x}</b><br>Count: %{customdata}<br>Percentage: %{y:.1f}%<extra></extra>'
            customdata = count_data['Count']  # Show actual counts in hover
        else:
            # Use raw counts
            y_col = 'Count'
            #title = f"{selected_column} Summary"
            title = " "
            # Simpler hover template for count-only display
            hover_template = '<b>%{x}</b><br>Count: %{y}<extra></extra>'
            customdata = None

        # Create bar chart using Plotly Express (faster than Graph Objects)
        if orientation == 'horizontal':
            # Horizontal bar chart: x-axis = values, y-axis = categories
            fig = px.bar(
                count_data, 
                x=y_col,  # Values (count or percentage)
                y=selected_column,  # Categories
                orientation='h',  # 'h' = horizontal bars
                title=title,
                color=y_col,  # Color bars by their height (creates gradient effect)
                color_continuous_scale='viridis'  # Professional color scheme
            )
        else:
            # Vertical bar chart: x-axis = categories, y-axis = values
            fig = px.bar(
                count_data, 
                x=selected_column,  # Categories
                y=y_col,  # Values (count or percentage)
                title=title,
                color=y_col,  # Color bars by their height
                color_continuous_scale='viridis'
            )

        # Customize chart appearance for better performance and aesthetics
        fig.update_layout(
            title_font_size=16,  # Readable title size
            height=450,  # Fixed height for consistent layout
            showlegend=False,  # Remove color legend to save space and improve performance
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
        )

        # Add custom hover information
        if customdata is not None:
            # When showing percentages, include both count and percentage in hover
            fig.update_traces(customdata=customdata, hovertemplate=hover_template)
        else:
            # When showing counts only, use simpler hover template
            fig.update_traces(hovertemplate=hover_template)

        # Add text annotations on bars only for small datasets to avoid clutter
        if len(count_data) <= 15:
            if show_percentages:
                # Show both count and percentage on bars
                text_values = [f"{count}<br>({pct:.1f}%)" for count, pct in 
                              zip(count_data['Count'], count_data['Percentage'])]
            else:
                # Show only counts on bars
                text_values = count_data['Count'].astype(str)

            # Position text outside bars for vertical, inside for horizontal orientation
            fig.update_traces(
                text=text_values,
                textposition='outside' if orientation == 'vertical' else 'inside'
            )

        # Display the chart using full container width
        st.plotly_chart(fig, use_container_width=True)

    # =============================================================================
    # RIGHT COLUMN: SUMMARY STATISTICS AND DATA TABLE
    # =============================================================================

    with col2:
        st.subheader("Summary Statistics")

        # Get summary statistics using cached data info for better performance
        #total_count = st.session_state.data_info['total_records']  # Total records in original dataset

        total_count = len(display_data) # Total records in original dataset
        unique_categories = display_data[selected_column].nunique()  # Number of unique values

        # Handle case where value_counts might be empty (error prevention)
        if not value_counts.empty:
            most_common = value_counts.index[0]  # Most frequent category
            most_common_count = value_counts.iloc[0]  # Count of most frequent category
        else:
            most_common = "N/A"
            most_common_count = 0

        # Display key metrics using Streamlit's metric widget for nice formatting
        st.metric("Total Records", f"{total_count:,}")  # :, adds thousand separators
        st.metric("Unique Categories", f"{unique_categories}")
        st.metric("Most Common", str(most_common))  # Convert to string for display
        st.metric("Most Common Count", f"{most_common_count:,}")

        # Show frequency table with the data
        st.subheader("Frequency Table")
        freq_table = count_data.copy()  # Copy to avoid modifying original data

        # Round percentages if they exist
        if 'Percentage' in freq_table.columns:
            freq_table['Percentage'] = freq_table['Percentage'].round(1)

        # Limit table size for performance - tables with many rows are slow to render
        if len(freq_table) > 10:
            freq_table_display = freq_table.head(10)  # Show only top 10 rows
            st.info(f"Showing top 10 out of {len(freq_table)} categories")
        else:
            freq_table_display = freq_table

        def auto_height(df, row_height=35, max_height=600):
            return min(len(df) * row_height + 40, max_height)
        # Display the frequency table with fixed height for consistent layout
        freq_table_flipped = freq_table_display.set_index(freq_table_display.columns[0]).T ##.style.set_properties(**{'text-align': 'center'})
        #st.dataframe(freq_table_flipped, use_container_width=True, height=auto_height(freq_table_flipped))
        st.dataframe(freq_table_flipped, use_container_width=True)
    # =============================================================================
    # CROSS-CATEGORY ANALYSIS SECTION
    # =============================================================================

    # Add visual separator and section header
    st.markdown("---")
    st.subheader("Cross-Category Analysis")

    # Disable cross-analysis for very large datasets to maintain performance
    if len(display_data) > 110000:
        st.warning("Cross-category analysis disabled for large datasets (>110,000 )to maintain performance. Please use filters to narrow down the data size, or enable the sampling option.")
        return  # Exit function early

    # Create two columns for category selection
    col_a, col_b = st.columns(2)

    with col_a:
        # Dropdown for first category in comparison
        category_1 = st.selectbox("First Category", categorical_columns, index=0)

    with col_b:
        # Dropdown for second category, defaulting to second column if available
        if len(categorical_columns) > 1:
            default_index = 1 if len(categorical_columns) > 1 else 0
            category_2 = st.selectbox("Second Category", categorical_columns, index=default_index)
        else:
            # Show warning if not enough categorical columns for cross-analysis
            st.warning("Need at least 2 categorical columns for cross-analysis")
            return  # Exit function early

    # Only proceed if user selected two different categories
    if category_1 != category_2:
        # Check if cross-tabulation would be too large (performance safeguard)
        unique_cat1 = display_data[category_1].nunique()  # Number of unique values in first category
        unique_cat2 = display_data[category_2].nunique()  # Number of unique values in second category

        # Limit cross-tabulation size to prevent performance issues
        if unique_cat1 * unique_cat2 > 1000:  # 1000 cells is reasonable limit
            st.warning(f"Cross-tabulation too large ({unique_cat1} x {unique_cat2}). "
                      "Please select categories with fewer unique values.")
            return  # Exit function early

        # Create two columns for cross-analysis visualization and data
        col3, col4 = st.columns(2)

        # =============================================================================
        # LEFT COLUMN: CROSS-TABULATION VISUALIZATION
        # =============================================================================

        with col3:
            # Get cached cross-tabulation to avoid recalculation
            cross_tab = get_cross_tabulation(display_data, category_1, category_2)

            # Limit categories shown in visualization for readability and performance
            if len(cross_tab.index) > 10:
                cross_tab = cross_tab.head(10)  # Keep only top 10 rows
                st.info("Showing top 10 categories for performance")

            if len(cross_tab.columns) > 10:
                cross_tab = cross_tab.iloc[:, :10]  # Keep only first 10 columns
                st.info("Showing top 10 subcategories for performance")

            # Create stacked bar chart using Plotly Express for better performance
            # Transpose (.T) the data for better visualization
            fig2 = px.bar(
                cross_tab.T,  # Transpose so categories become x-axis
                title=f"{category_1} vs {category_2} Distribution",
                height=400  # Fixed height for consistency
            )

            # Customize chart layout
            fig2.update_layout(
                xaxis_title=category_2,  # X-axis shows second category
                yaxis_title="Count",     # Y-axis shows counts
                showlegend=True,         # Show legend for first category
                legend_title=category_1  # Legend title is first category
            )


            # Display the cross-tabulation chart
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(build_heatmap(cross_tab, category_1, category_2), use_container_width=True)


        # =============================================================================
        # RIGHT COLUMN: CROSS-TABULATION DATA TABLE
        # =============================================================================

        with col4:
            st.subheader("Cross-Tabulation")

            # Show preview for large tables to maintain performance
            if cross_tab.size > 100:  # If table has more than 100 cells
                st.info("Showing preview of cross-tabulation")
                # Show only 5x5 preview of the full table
                st.dataframe(cross_tab.iloc[:5, :5], use_container_width=True)
            else:
                # Show full table for smaller cross-tabulations
                st.dataframe(cross_tab, use_container_width=True)

            # Show summary statistics instead of full proportions table (better performance)
            st.subheader("Summary")
            st.write(f"**Categories in {category_1}:** {len(cross_tab.index)}")
            st.write(f"**Categories in {category_2}:** {len(cross_tab.columns)}")
            st.write(f"**Total Combinations:** {cross_tab.sum().sum():,}")  # Total count across all cells

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the main function when script is executed directly
    main()


























