import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

##

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from datetime import datetime

fall24=r"\\acdfs.alamo.edu\spc$\Departments\PlanningResearch\Data Analyst\2 Data\1 Most Frequently Used Data Sources\CBM0C1\Current Files\F24 XST_CBM0C1_SPC.xlsx"
spring25=r"\\acdfs.alamo.edu\spc$\Departments\PlanningResearch\Data Analyst\2 Data\1 Most Frequently Used Data Sources\CBM0C1\Current Files\SP25_XST_CBM0C1_ACCD_SPC.csv"
summer25=r"\\acdfs.alamo.edu\spc$\Departments\PlanningResearch\Data Analyst\2 Data\1 Most Frequently Used Data Sources\CBM0C1\Current Files\summer25_CBM0C1.xlsx"
def merge_input_files(input_files=[fall24, spring25, summer25]):
    columns = [
        "C1_BANNER_ID",
        "C1_CBM_TERM_DESC",
        
        "C1_COLLEGE",
        "C1_GENDER_DESC",
        "C1_DATE_OF_BIRTH",
        "C1_FTIC_DC_DESC",
        "C1_TYPE_MAJOR_DESC",
        "C1_FTPT_COLLEGE_CENSUS",
        "C1_THECB_ETHNICITY"
    ]
    
    dfs = []
    for file in input_files:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            df = pd.read_excel(file)
            df = df[columns]
        elif file.endswith('.csv'):
            df = pd.read_csv(file, encoding='latin1')
            df = df[columns]
        else:
            continue
        dfs.append(df)
    
    df_merge = pd.concat(dfs, ignore_index=True)
    return df_merge
def create_input_from_cbmc1(dataframe_input):
    """
    Load and process CBM (Coordinating Board Management) data from Excel file
    
    Parameters:
    file_path (str): Path to the Excel file. If None, will prompt for file selection.
    
    Returns:
    pd.DataFrame: Processed CBM data with renamed columns and age groups
    """
    try:
        # Use the provided dataframe input directly  
        cbm = dataframe_input.copy()
        # Define columns of interest
        interest_cols = [
            "C1_CBM_TERM_DESC",
             
            "C1_COLLEGE",
            "C1_GENDER_DESC",
            "C1_DATE_OF_BIRTH",
            "C1_FTIC_DC_DESC",
            "C1_TYPE_MAJOR_DESC",
            "C1_FTPT_COLLEGE_CENSUS",
            "C1_THECB_ETHNICITY"
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in interest_cols if col not in cbm.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            # Use only available columns
            interest_cols = [col for col in interest_cols if col in cbm.columns]
        
        # Select only columns of interest
        cbm = cbm[interest_cols].copy()  # Use .copy() to avoid SettingWithCopyWarning
        
        # Create column mapping for renaming
        column_mapping = {
            "C1_CBM_TERM_DESC": "Term",
            
            "C1_COLLEGE": "SPC College",
            "C1_GENDER_DESC": "Gender",
            "C1_FTIC_DC_DESC": "Student Type",
            "C1_TYPE_MAJOR_DESC": "Type_Major",
            "C1_FTPT_COLLEGE_CENSUS": "Full_Part_Time",
            "C1_THECB_ETHNICITY": "ETHNICITY",
            "C1_DATE_OF_BIRTH": "DATE_OF_BIRTH"
        }
        
        # Rename columns (only rename columns that exist)
        existing_mapping = {k: v for k, v in column_mapping.items() if k in cbm.columns}
        cbm.rename(columns=existing_mapping, inplace=True)
        
        # Process date of birth if column exists
        if 'DATE_OF_BIRTH' in cbm.columns:
            # Convert to datetime with error handling
            cbm['DATE_OF_BIRTH'] = pd.to_datetime(cbm['DATE_OF_BIRTH'], errors='coerce')
            
            # Calculate age - fix the age calculation
            current_date = pd.Timestamp.now()
            cbm['Age'] = (current_date - cbm['DATE_OF_BIRTH']).dt.days // 365.25
            
            # Convert to integer and handle NaN values
            cbm['Age'] = cbm['Age'].astype('Int64')  # Use nullable integer type
            
            # Create age groups with proper handling of NaN values
            cbm['Age_Group'] = pd.cut(
                cbm['Age'], 
                bins=[0, 18, 25, 30, 35, 40, 50, 60, 100],  # Added 0 and 100 for edge cases
                labels=["Under 18", "18-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"],
                include_lowest=True
            )
            
            # Drop the original date column
            cbm.drop(columns=['DATE_OF_BIRTH','Age'], inplace=True)
        else:
            print("Warning: DATE_OF_BIRTH column not found. Age calculation skipped.")
        
        # Clean the data - remove rows with all NaN values
        cbm.dropna(how='all', inplace=True)
        
        # Print data info
        print(f"Data loaded successfully!")
        print(f"Shape: {cbm.shape}")
        print(f"Columns: {list(cbm.columns)}")
        
        return cbm
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
        
    except pd.errors.EmptyDataError:
        print("Error: The Excel file is empty.")
        return None
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

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


##

# =============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE OPTIMIZATION
# =============================================================================

# Cache the data loading function to avoid reloading data every time the app runs
@st.cache_data
def load_and_process_data():
    """
    Load and process CBM data with caching for performance
    
    The @st.cache_data decorator ensures this function only runs once per session.
    Subsequent calls will return the cached result, dramatically improving performance.
    
    Returns:
        pd.DataFrame: Processed CBM data or None if loading fails
    """
    try:
        # Call the custom function to load and process CBM data from Excel file
        # This file path points to the SPC CBM data on the network drive
        sample_data = create_input_from_cbmc1(merge_input_files())
        
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
    return pd.crosstab(data[col1], data[col2])

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
    st.set_page_config(page_title="SPC Data Dashboard", layout="wide")
    
    # Create the main title and separator line
    st.title("SPC Data Dashboard")
    st.markdown("---")  # Creates a horizontal line for visual separation
    
    # =============================================================================
    # DATA LOADING SECTION
    # =============================================================================
    
    # Load data with caching - this spinner shows while data is loading
    # The spinner improves user experience by indicating that something is happening
    with st.spinner("Loading data..."):
        sample_data = load_and_process_data()
    
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
    
    # Create sidebar for user controls - keeps main area clean for visualizations
    st.sidebar.header("Analysis Controls")
    
    # Dropdown to select which categorical column to analyze
    # Index=0 means the first column is selected by default
    selected_column = st.sidebar.selectbox(
        "Select Category to Analyze",
        categorical_columns,
        index=0
    )
    

    
    #checkbox to taggle between Terms
    show_terms = st.sidebar.multiselect(
        "Select Terms to Include",
        options=sample_data["Academic Period"].unique(),
        default=sample_data["Academic Period"].unique().tolist()  # Select all terms by default
    )
    
    # Checkbox to toggle between showing counts vs percentages
    show_percentages = st.sidebar.checkbox("Show Percentages", value=False)
    
    #----------------------------
    #fiter based on select terms 
    #--------------------------------
    if show_terms:
        sample_data = sample_data[sample_data["Academic Period"].isin(show_terms)]
    else:
        st.warning("Please select at least one term to display data.")
        return
    
        # Radio buttons for chart orientation - affects how bars are displayed
    orientation = st.sidebar.radio(
        "Chart Orientation",
        ["vertical", "horizontal"]  # vertical = bars go up, horizontal = bars go sideways
    )
    #-----------------------
    
    # =============================================================================
    # PERFORMANCE OPTIMIZATION FOR LARGE DATASETS
    # =============================================================================
    
    # If dataset is very large (>10,000 records), offer sampling option for better performance
    if len(sample_data) > 10000:
        # Inform user about large dataset and optimization options
        st.sidebar.info(f"Dataset has {len(sample_data):,} records. Using optimized processing.")
        
        # Checkbox to enable/disable sampling
        use_sampling = st.sidebar.checkbox("Use sampling for faster processing", value=True)
        
        if use_sampling:
            # Use maximum of 5,000 records or the full dataset size, whichever is smaller
            sample_size = min(5000, len(sample_data))
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
            title = f"Count of Each {selected_column}"
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
        st.dataframe(freq_table_display, use_container_width=True, height=auto_height(freq_table_display))
    
    # =============================================================================
    # CROSS-CATEGORY ANALYSIS SECTION
    # =============================================================================
    
    # Add visual separator and section header
    st.markdown("---")
    st.subheader("Cross-Category Analysis")
    
    # Disable cross-analysis for very large datasets to maintain performance
    if len(display_data) > 50000:
        st.warning("Cross-category analysis disabled for large datasets to maintain performance.")
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
        if unique_cat1 * unique_cat2 > 500:  # 300 cells is reasonable limit
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
