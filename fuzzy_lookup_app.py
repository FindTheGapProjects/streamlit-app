import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime
from fuzzywuzzy import process, fuzz
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Fuzzy Lookup Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_text(text, clean_options):
    """
    Clean and normalize text based on selected options
    
    Parameters:
    text (str): Text to clean
    clean_options (dict): Dictionary of cleaning options
    
    Returns:
    str: Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string if not already
    text = str(text).strip()
    
    # Apply selected cleaning options
    if clean_options.get('lowercase', False):
        text = text.lower()
    
    if clean_options.get('remove_punctuation', False):
        text = re.sub(r'[^\w\s]', ' ', text)
    
    if clean_options.get('remove_extra_spaces', False):
        text = re.sub(r'\s+', ' ', text).strip()
    
    if clean_options.get('remove_special_chars', False):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    if clean_options.get('expand_abbreviations', False):
        # Define common abbreviations and their expansions
        abbreviations = {
            'inc': 'incorporated',
            'corp': 'corporation',
            'co': 'company',
            'ltd': 'limited',
            'llc': 'limited liability company',
            'intl': 'international',
            'bro': 'brothers',
            'mfg': 'manufacturing',
            'tech': 'technology',
            'sys': 'systems',
            'svcs': 'services',
            'svc': 'service',
            'hldgs': 'holdings',
            'sol': 'solutions',
            '&': 'and',
            'assn': 'association',
            'assoc': 'associates',
            'dept': 'department',
            'grp': 'group',
            'inst': 'institute',
            'univ': 'university',
            'dev': 'development',
            'ctr': 'center',
            'natl': 'national',
            'ent': 'enterprises'
        }
        
        # Split text into words
        words = text.split()
        
        # Replace abbreviations
        for i, word in enumerate(words):
            word_clean = word.strip(',.:;()[]{}').lower()
            if word_clean in abbreviations:
                words[i] = abbreviations[word_clean]
        
        text = ' '.join(words)
    
    if clean_options.get('remove_common_words', False):
        # Define common words to remove
        common_words = {
            'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
            'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
            'but', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
            'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them',
            'these', 'so', 'some', 'her', 'would', 'make', 'like', 'him', 'into',
            'time', 'has', 'look', 'two', 'more', 'go', 'see', 'no', 'way', 'could',
            'my', 'than', 'been', 'call', 'who', 'its', 'now', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Split text into words
        words = text.split()
        
        # Remove common words
        words = [word for word in words if word.lower() not in common_words]
        
        text = ' '.join(words)
    
    return text

def clean_dataframe(df, column, clean_options):
    """
    Apply cleaning to a specific column in a dataframe
    
    Parameters:
    df (DataFrame): Dataframe to clean
    column (str): Column name to clean
    clean_options (dict): Dictionary of cleaning options
    
    Returns:
    DataFrame: DataFrame with cleaned column and original column
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Create a new column for the cleaned text
    clean_col_name = f"clean_{column}"
    df_copy[clean_col_name] = df_copy[column].apply(
        lambda x: clean_text(x, clean_options)
    )
    
    return df_copy, clean_col_name

def fuzzy_lookup(df1, df2, name_col1, name_col2, threshold=70, max_matches=1, 
                 clean_data=False, clean_options=None):
    """
    Perform fuzzy matching between two dataframes based on specified name columns.
    
    Parameters:
    df1 (DataFrame): First dataframe
    df2 (DataFrame): Second dataframe
    name_col1 (str): Name column in first dataframe
    name_col2 (str): Name column in second dataframe
    threshold (int): Minimum similarity score (0-100) to include in results
    max_matches (int): Maximum number of matches to return per name
    clean_data (bool): Whether to clean and normalize data
    clean_options (dict): Dictionary of cleaning options
    
    Returns:
    DataFrame: The matched results with similarity scores
    """
    # Create suffix for overlapping column names
    df1_suffix = '_file1'
    df2_suffix = '_file2'
    
    # Create copies to avoid modifying original dataframes
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Clean data if requested
    if clean_data and clean_options:
        df1_copy, clean_col1 = clean_dataframe(df1_copy, name_col1, clean_options)
        df2_copy, clean_col2 = clean_dataframe(df2_copy, name_col2, clean_options)
        
        # Use cleaned columns for matching
        match_col1 = clean_col1
        match_col2 = clean_col2
        
        # Keep original columns for display
        original_col1 = name_col1
        original_col2 = name_col2
    else:
        # Use original columns for matching
        match_col1 = name_col1
        match_col2 = name_col2
        original_col1 = name_col1
        original_col2 = name_col2
    
    # Rename columns to avoid conflicts
    # First, rename all columns except the matching columns
    df1_columns = {col: f"{col}{df1_suffix}" for col in df1_copy.columns if col != original_col1}
    df2_columns = {col: f"{col}{df2_suffix}" for col in df2_copy.columns if col != original_col2}
    
    # Rename matching columns separately
    df1_columns[original_col1] = f"{original_col1}{df1_suffix}"
    df2_columns[original_col2] = f"{original_col2}{df2_suffix}"
    
    # If we have clean columns, rename them too
    if clean_data and clean_options:
        df1_columns[match_col1] = f"{match_col1}{df1_suffix}"
        df2_columns[match_col2] = f"{match_col2}{df2_suffix}"
    
    df1_copy = df1_copy.rename(columns=df1_columns)
    df2_copy = df2_copy.rename(columns=df2_columns)
    
    # Create a dictionary for name lookups
    if clean_data and clean_options:
        match_dict = dict(zip(df2_copy[f"{match_col2}{df2_suffix}"], df2_copy[f"{match_col2}{df2_suffix}"]))
    else:
        match_dict = dict(zip(df2_copy[f"{match_col2}{df2_suffix}"], df2_copy[f"{match_col2}{df2_suffix}"]))
    
    # Perform fuzzy matching
    results = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df1_copy)
    
    for i, (_, row) in enumerate(df1_copy.iterrows()):
        # Get the value to match on
        if clean_data and clean_options:
            match_value = row[f"{match_col1}{df1_suffix}"]
        else:
            match_value = row[f"{match_col1}{df1_suffix}"]
        
        # Update progress
        progress_percent = min(1.0, (i + 1) / total_rows)
        progress_bar.progress(progress_percent)
        status_text.text(f"Processing record {i+1}/{total_rows} ({progress_percent*100:.1f}%)")
        
        if pd.isna(match_value) or match_value == '':
            continue
        
        if max_matches == 1:
            # Find the best match and score
            try:
                match, score = process.extractOne(str(match_value), match_dict.keys(), scorer=fuzz.token_sort_ratio)
                
                # Only include matches above the threshold
                if score >= threshold:
                    # Get the matching row from df2
                    if clean_data and clean_options:
                        match_df2 = df2_copy[df2_copy[f"{match_col2}{df2_suffix}"] == match]
                    else:
                        match_df2 = df2_copy[df2_copy[f"{match_col2}{df2_suffix}"] == match]
                    
                    if not match_df2.empty:
                        match_row = match_df2.iloc[0].to_dict()
                        
                        # Create a new row with data from both dataframes
                        result_row = row.to_dict()
                        
                        # Add the match information
                        result_row.update(match_row)
                        
                        # Add the similarity score
                        result_row['similarity_score'] = float(score)
                        
                        # Add the matching values used
                        if clean_data and clean_options:
                            result_row['matching_value_file1'] = match_value
                            result_row['matching_value_file2'] = match
                        
                        results.append(result_row)
            except Exception as e:
                st.warning(f"Error processing row {i+1}: {str(e)}")
                continue
        else:
            # Find top N matches
            try:
                matches = process.extract(str(match_value), match_dict.keys(), 
                                         scorer=fuzz.token_sort_ratio, 
                                         limit=max_matches)
                
                for match, score in matches:
                    # Only include matches above the threshold
                    if score >= threshold:
                        # Get the matching row from df2
                        if clean_data and clean_options:
                            match_df2 = df2_copy[df2_copy[f"{match_col2}{df2_suffix}"] == match]
                        else:
                            match_df2 = df2_copy[df2_copy[f"{match_col2}{df2_suffix}"] == match]
                        
                        if not match_df2.empty:
                            match_row = match_df2.iloc[0].to_dict()
                            
                            # Create a new row with data from both dataframes
                            result_row = row.to_dict()
                            
                            # Add the match information
                            result_row.update(match_row)
                            
                            # Add the similarity score
                            result_row['similarity_score'] = float(score)
                            
                            # Add the matching values used
                            if clean_data and clean_options:
                                result_row['matching_value_file1'] = match_value
                                result_row['matching_value_file2'] = match
                            
                            results.append(result_row)
            except Exception as e:
                st.warning(f"Error processing row {i+1}: {str(e)}")
                continue
    
    # Clear progress displays
    progress_bar.progress(1.0)
    status_text.text("Matching complete!")
    
    # Convert results to DataFrame
    if results:
        result_df = pd.DataFrame(results)
        
        # Drop the clean columns if they exist
        if clean_data and clean_options:
            cols_to_drop = [col for col in result_df.columns if col.startswith('clean_') and col not in ['matching_value_file1', 'matching_value_file2']]
            result_df = result_df.drop(columns=cols_to_drop, errors='ignore')
        
        return result_df
    else:
        return pd.DataFrame()

def save_uploaded_file(file):
    """Temporarily save an uploaded file and return its path"""
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp_uploads'):
        os.makedirs('temp_uploads')
    
    file_path = os.path.join('temp_uploads', file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    
    return file_path

def main():
    # Application header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Fuzzy Lookup Tool")
        st.subheader("Match records between files with fuzzy string matching")
    
    # Add info in sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool performs fuzzy string matching between two files, 
        similar to Microsoft Excel's Fuzzy Lookup add-in.
        
        It allows you to find matching records even when the text 
        isn't identical due to typos, abbreviations, or different formats.
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload two files (CSV or Excel)
        2. Select the column to match on in each file
        3. Configure matching parameters
        4. Run the matching process
        5. Review and download the results
        """)
        
        st.header("Settings")
        # Add global settings here if needed
        
        # Add reset button to sidebar
        if st.button("Reset and Start Over", type="primary", icon="🔄"):
            # Clear all session state variables to reset the app
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Force browser refresh using JavaScript
            st.markdown(
                """
                <script>
                    window.top.location.reload();
                </script>
                """,
                unsafe_allow_html=True
            )
    
    # Main content
    st.markdown("""
    Upload two files and select the columns you want to match on. 
    The tool will find all pairs of records with similar text in the chosen columns.
    """)
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First File")
        file1 = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'], key="file1_uploader")
    
    with col2:
        st.subheader("Second File")
        file2 = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'], key="file2_uploader")
    
    if file1 is not None and file2 is not None:
        # Load the first file
        try:
            if file1.name.endswith('.csv'):
                df1 = pd.read_csv(file1)
            else:
                df1 = pd.read_excel(file1)
            
            # Load the second file
            if file2.name.endswith('.csv'):
                df2 = pd.read_csv(file2)
            else:
                df2 = pd.read_excel(file2)
            
            # Display file information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**First file:** {file1.name}")
                st.write(f"Records: {len(df1)}, Columns: {len(df1.columns)}")
                with st.expander("Preview first file"):
                    st.dataframe(df1.head(5))
                
                # Select name column from first file
                name_col1_options = [""] + list(df1.columns)
                default_index1 = 0
                # Try to find a "name" column automatically
                name_candidates = ["name", "company", "supplier", "customer", "entity", "organization"]
                for candidate in name_candidates:
                    for col in df1.columns:
                        if candidate in col.lower():
                            default_index1 = name_col1_options.index(col)
                            break
                    if default_index1 > 0:
                        break
                
                name_col1 = st.selectbox("Select column to match on (from first file)", 
                                        name_col1_options,
                                        index=default_index1)
            
            with col2:
                st.write(f"**Second file:** {file2.name}")
                st.write(f"Records: {len(df2)}, Columns: {len(df2.columns)}")
                with st.expander("Preview second file"):
                    st.dataframe(df2.head(5))
                
                # Select name column from second file
                name_col2_options = [""] + list(df2.columns)
                default_index2 = 0
                # Try to find a "name" column automatically
                for candidate in name_candidates:
                    for col in df2.columns:
                        if candidate in col.lower():
                            default_index2 = name_col2_options.index(col)
                            break
                    if default_index2 > 0:
                        break
                
                name_col2 = st.selectbox("Select column to match on (from second file)", 
                                        name_col2_options,
                                        index=default_index2)
            
            # Data cleaning options
            st.subheader("Data Cleaning & Normalization")
            
            clean_data = st.checkbox("Clean and normalize data before matching", value=True,
                                    help="Apply cleaning operations to improve match quality")
            
            if clean_data:
                st.write("Select data cleaning options:")
                
                clean_col1, clean_col2, clean_col3 = st.columns(3)
                
                with clean_col1:
                    lowercase = st.checkbox("Convert to lowercase", value=True)
                    remove_punctuation = st.checkbox("Remove punctuation", value=True)
                    remove_extra_spaces = st.checkbox("Remove extra spaces", value=True)
                
                with clean_col2:
                    remove_special_chars = st.checkbox("Remove special characters", value=False)
                    expand_abbreviations = st.checkbox("Expand common abbreviations", value=True,
                                                     help="E.g., 'Inc' → 'Incorporated', 'Ltd' → 'Limited'")
                
                with clean_col3:
                    remove_common_words = st.checkbox("Remove common words", value=False,
                                                    help="Remove common words like 'the', 'and', etc.")
                
                # Create a dictionary of cleaning options
                clean_options = {
                    'lowercase': lowercase,
                    'remove_punctuation': remove_punctuation,
                    'remove_extra_spaces': remove_extra_spaces,
                    'remove_special_chars': remove_special_chars,
                    'expand_abbreviations': expand_abbreviations,
                    'remove_common_words': remove_common_words
                }
                
                # Show preview of cleaned data
                if st.checkbox("Show cleaning preview"):
                    if name_col1 and name_col1 in df1.columns:
                        st.write("**Cleaning preview for first file:**")
                        
                        # Get a few sample values
                        sample_values = df1[name_col1].dropna().head(5).tolist()
                        
                        # Create a dataframe to display original and cleaned values
                        preview_df = pd.DataFrame({
                            'Original': sample_values,
                            'Cleaned': [clean_text(val, clean_options) for val in sample_values]
                        })
                        
                        st.dataframe(preview_df)
                    
                    if name_col2 and name_col2 in df2.columns:
                        st.write("**Cleaning preview for second file:**")
                        
                        # Get a few sample values
                        sample_values = df2[name_col2].dropna().head(5).tolist()
                        
                        # Create a dataframe to display original and cleaned values
                        preview_df = pd.DataFrame({
                            'Original': sample_values,
                            'Cleaned': [clean_text(val, clean_options) for val in sample_values]
                        })
                        
                        st.dataframe(preview_df)
            else:
                clean_options = None
            
            # Matching parameters
            st.subheader("Matching Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                threshold = st.slider("Minimum similarity score (0-100)", 0, 100, 40,
                                    help="Only matches with a similarity score above this threshold will be included")
            
            with col2:
                max_matches = st.number_input("Maximum matches per record", 1, 10, 1,
                                            help="Number of potential matches to return for each record in the first file")
            
            with col3:
                output_format = st.selectbox("Output file format", ["Excel", "CSV"], index=0)
            
            # Create a progress placeholder
            progress_placeholder = st.empty()
            
            # Run matching
            if st.button("Run Fuzzy Lookup", type="primary", icon="🔍"):
                if not name_col1 or not name_col2:
                    st.error("Please select columns to match on for both files")
                else:
                    with st.container() as progress_container:
                        st.subheader("Matching in progress...")
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        time_indicator = st.empty()
                        
                        try:
                            # Start the timer
                            start_time = time.time()
                            
                            
                            # Update time display
                            def update_time():
                                elapsed = time.time() - start_time
                                time_indicator.text(f"Elapsed time: {elapsed:.1f} seconds")
                                
                            update_time()
                            progress_text.text("Preparing data for matching...")
                            
                            
                            # Perform the matching
                            result_df = fuzzy_lookup(df1, df2, name_col1, name_col2, 
                                                   threshold, max_matches,
                                                   clean_data, clean_options)
                            
                            
                            # Final updates (no need to clear container)
                            update_time()
                            progress_text.text("Matching complete!")
                            progress_bar.progress(1.0)

                            
                            if len(result_df) > 0:
                                # Display the results
                                st.subheader("Matching Results")
                                st.write(f"Found {len(result_df)} matches with similarity score >= {threshold}")
                                
                                # Reorganize columns for better display:
                                # 1. Get column names from original dataframes
                                first_file_name_col = f"{name_col1}_file1"
                                second_file_name_col = f"{name_col2}_file2"
                                
                                # 2. Find all columns from first file and second file
                                first_file_cols = [col for col in result_df.columns if col.endswith('_file1')]
                                second_file_cols = [col for col in result_df.columns if col.endswith('_file2')]
                                
                                # 3. Remove name columns from these lists as we'll place them first
                                if first_file_name_col in first_file_cols:
                                    first_file_cols.remove(first_file_name_col)
                                if second_file_name_col in second_file_cols:
                                    second_file_cols.remove(second_file_name_col)
                                
                                # 4. Create a new column 'is_match' that will be empty (no checkbox)
                                result_df['is_match'] = 0  # Default to 0 (not matched)
                                
                                # 5. Create the final column order
                                column_order = [
                                    first_file_name_col, 
                                    second_file_name_col,
                                    'is_match',
                                    'similarity_score'
                                ] + first_file_cols + second_file_cols
                                
                                # Drop matching value columns if they exist
                                for col in ['matching_value_file1', 'matching_value_file2']:
                                    if col in result_df.columns and col in column_order:
                                        column_order.remove(col)
                                
                                # Reorder columns
                                result_df = result_df[column_order]
                                
                                # Rename columns for better display
                                rename_dict = {
                                    first_file_name_col: "Source Supplier Name",
                                    second_file_name_col: "Target Supplier Name"
                                }
                                result_df = result_df.rename(columns=rename_dict)
                                
                                # Create a display copy with formatted scores for display only
                                display_df = result_df.copy()
                                if 'similarity_score' in display_df.columns:
                                    display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.1f}%")
                                
                                # Display the table (non-editable for now)
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Create download button based on selected format
                                if output_format == "CSV":
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        "Download Results as CSV",
                                        csv,
                                        f"fuzzy_matches_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        "text/csv",
                                        key="download_csv"
                                    )
                                else:
                                    # For Excel, we need to save to a BytesIO object
                                    buffer = BytesIO()
                                    
                                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                        # Convert similarity_score to decimals before writing to Excel
                                        export_df = result_df.copy()
                                        if 'similarity_score' in export_df.columns:
                                            export_df['similarity_score'] = export_df['similarity_score'] / 100  # Convert to decimal for proper percentage formatting
                                        
                                        export_df.to_excel(writer, index=False, sheet_name='Matches')
                                        
                                        # Get the xlsxwriter workbook and worksheet objects
                                        workbook = writer.book
                                        worksheet = writer.sheets['Matches']
                                        
                                        # Add formats
                                        score_format = workbook.add_format({'num_format': '0.0%'})
                                        
                                        # Find the similarity score column
                                        if 'similarity_score' in export_df.columns:
                                            score_col_idx = export_df.columns.get_loc('similarity_score')
                                            # Apply custom formatting to the similarity score column
                                            worksheet.set_column(score_col_idx, score_col_idx, None, score_format)
                                    
                                    buffer.seek(0)
                                    st.download_button(
                                        "Download Results as Excel",
                                        buffer,
                                        f"fuzzy_matches_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        "application/vnd.ms-excel",
                                        key="download_excel"
                                    )
                                
                                # Show match quality statistics
                                st.subheader("Match Quality Distribution")
                                
                                if 'similarity_score' in result_df.columns:
                                    # Use the numeric scores directly for analysis
                                    scores = result_df['similarity_score']
                                    
                                    # Create bins for score distribution
                                    bins = [float(threshold), 75.0, 85.0, 95.0, 100.1]  # Add 0.1 to include 100%
                                    labels = [f"{threshold}-74%", "75-84%", "85-94%", "95-100%"]
                                    
                                    # Count matches in each bin
                                    score_bins = pd.cut(scores, bins=bins, labels=labels, right=False)
                                    score_counts = score_bins.value_counts().sort_index()
                                    
                                    # Display stats
                                    st.bar_chart(score_counts)
                                    
                                    # Show some summary statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Average Match Score", f"{scores.mean():.1f}%")
                                    with col2:
                                        st.metric("Median Match Score", f"{scores.median():.1f}%")
                                    with col3:
                                        st.metric("Min Match Score", f"{scores.min():.1f}%")
                                    with col4:
                                        st.metric("Max Match Score", f"{scores.max():.1f}%")
                                
                                # If cleaning was applied, show a note about it
                                if clean_data and clean_options:
                                    st.info("""
                                    **Note about data cleaning:** The matches were performed using normalized data. 
                                    The original values are shown in the results, but the matching was done on cleaned values.
                                    """)
                                    
                                    if 'matching_value_file1' in result_df.columns and 'matching_value_file2' in result_df.columns:
                                        with st.expander("View cleaned values used for matching"):
                                            # Create a display dataframe with original and cleaned values
                                            clean_preview = pd.DataFrame({
                                                f"Original {name_col1}": result_df[f"{name_col1}_file1"],
                                                f"Cleaned {name_col1}": result_df["matching_value_file1"],
                                                f"Original {name_col2}": result_df[f"{name_col2}_file2"],
                                                f"Cleaned {name_col2}": result_df["matching_value_file2"],
                                                "Similarity Score": result_df["similarity_score"].apply(lambda x: f"{x:.1f}%")
                                            })
                                            
                                            st.dataframe(clean_preview)
                            else:
                                st.info(f"No matches found with similarity score >= {threshold}. Try lowering the threshold or adjusting the data cleaning options.")
                        
                        except Exception as e:
                            st.error(f"Error during matching: {str(e)}")
                            import traceback
                            st.exception(traceback.format_exc())
        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            import traceback
            st.exception(traceback.format_exc())
    else:
        st.info("Please upload both files to begin matching")
        
    # Footer
    st.markdown("---")
    st.markdown("### Tips for Better Results")
    st.markdown("""
    - **Data Quality:** Clean your data to standardize formats and remove inconsistencies
    - **Threshold Tuning:** Start with a higher threshold (90+) and lower it gradually if needed
    - **Validation:** Examine matches with scores 70-85% carefully as they often need manual review
    - **Company Names:** Consider enabling the abbreviation expansion option for company names
    - **Special Cases:** For highly specialized data, you may need to customize the cleaning options
    """)


if __name__ == "__main__":
    main()