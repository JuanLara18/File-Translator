import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from excel_translation import translate_text, create_translation_cache
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Excel Translator",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 5px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    /* Mejoras para las tablas */
    .dataframe {
        border-collapse: collapse;
        margin: 1rem 0;
        width: 100%;
    }
    .dataframe th {
        background-color: #f8f9fa;
        padding: 0.5rem;
    }
    .dataframe td {
        padding: 0.5rem;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Get OpenAI API key from Streamlit secrets
if 'OPENAI_API_KEY' not in st.secrets:
    st.error('OPENAI_API_KEY not found in secrets. Please add it to your .streamlit/secrets.toml file.')
    st.stop()

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_file_size(file):
    """Get file size in appropriate units."""
    size_bytes = len(file.getvalue())
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} GB"

def process_file(uploaded_file, columns_to_translate, batch_size):
    """Process the uploaded Excel file and translate selected columns."""
    df = pd.read_excel(uploaded_file)
    
    # Initialize session state for stats if not exists
    if 'translation_stats' not in st.session_state:
        st.session_state.translation_stats = {
            'total_cells': 0,
            'unique_texts': 0,
            'processed_columns': 0
        }
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total translations needed
    total_unique_texts = sum(len(df[col].dropna().unique()) for col in columns_to_translate)
    current_progress = 0
    
    # Update stats
    st.session_state.translation_stats['total_cells'] = sum(len(df[col].dropna()) for col in columns_to_translate)
    st.session_state.translation_stats['unique_texts'] = total_unique_texts
    st.session_state.translation_stats['processed_columns'] = len(columns_to_translate)
    
    # Process each selected column
    for column in columns_to_translate:
        status_text.text(f"Processing column: {column}")
        
        # Create translation cache for the column
        translation_cache = {}
        unique_texts = df[column].dropna().unique()
        
        # Process in batches
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i + batch_size]
            for text in batch:
                if pd.notna(text):  # Skip NaN values
                    translated = translate_text(str(text), translation_cache)
                    translation_cache[text] = translated
                    
                    # Update progress
                    current_progress += 1
                    progress_percentage = min(current_progress / total_unique_texts, 1.0)
                    progress_bar.progress(progress_percentage)
        
        # Create new column name
        new_column = f"{column}_English"
        
        # Apply translations to DataFrame
        df[new_column] = df[column].map(lambda x: translation_cache.get(x, x) if pd.notna(x) else x)
    
    progress_bar.progress(1.0)
    status_text.text("Translation completed!")
    
    return df

def main():
    # Title and description
    st.title("🌐 Excel Translator")
    st.markdown("""
    Transform your Excel files by translating columns from German to English using OpenAI's GPT API.
    Upload your file, select columns, and get instant translations!
    """)
    
    # Sidebar with settings and stats
    with st.sidebar:
        st.header("⚙️ Settings & Stats")
        
        # Batch size slider
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of texts to translate in each batch"
        )
        
        if 'translation_stats' in st.session_state:
            st.markdown("### 📊 Translation Statistics")
            st.metric("Processed Columns", st.session_state.translation_stats['processed_columns'])
            st.metric("Total Cells", st.session_state.translation_stats['total_cells'])
            st.metric("Unique Texts", st.session_state.translation_stats['unique_texts'])
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload your Excel file containing German text"
        )
        
        if uploaded_file is not None:
            file_size = get_file_size(uploaded_file)
            st.info(f"File uploaded: {uploaded_file.name} ({file_size})")
            
            # Read and display original file
            df = pd.read_excel(uploaded_file)
            st.markdown("### 👀 File Preview")
            st.dataframe(
                df.head(),
                use_container_width=True,
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        width="auto",
                        help=f"Column: {col}"
                    ) for col in df.columns
                }
            )
            
            # Column selection
            st.markdown("### 🎯 Select Columns")
            available_columns = df.columns.tolist()
            columns_to_translate = st.multiselect(
                "Select columns to translate",
                available_columns,
                help="You can select multiple columns for translation"
            )
            
            if columns_to_translate:
                if st.button("🚀 Start Translation", use_container_width=True):
                    with st.spinner("Translation in progress..."):
                        translated_df = process_file(uploaded_file, columns_to_translate, batch_size)
                        
                        st.markdown("### ✨ Preview of Translated File")
                        st.dataframe(
                            translated_df.head(),
                            use_container_width=True,
                            column_config={
                                col: st.column_config.TextColumn(
                                    col,
                                    width="auto",
                                    help=f"Column: {col}"
                                ) for col in translated_df.columns
                            }
                        )
                        
                        # Save to temporary file for download
                        output_buffer = BytesIO()
                        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                            translated_df.to_excel(writer, index=False)
                        
                        # Create download button
                        st.download_button(
                            label="📥 Download Translated File",
                            data=output_buffer.getvalue(),
                            file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
    
    with col2:
        if uploaded_file is None:
            st.markdown("### 🔍 How to Use")
            st.markdown("""
            1. Upload your Excel file using the file uploader
            2. Review the file preview
            3. Select columns to translate
            4. Adjust batch size in settings if needed
            5. Click 'Start Translation'
            6. Download your translated file
            """)
            
            st.markdown("### ℹ️ Note")
            st.info("""
            - Files are processed securely
            - Translations are powered by OpenAI's GPT API
            - Large files may take longer to process
            - Adjust batch size for better performance
            """)

if __name__ == "__main__":
    main()