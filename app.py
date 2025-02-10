import streamlit as st
from excel_translation import translate_text, process_excel
import pandas as pd
from io import BytesIO
import os
from openai import OpenAI
import time

# Page configuration
st.set_page_config(
    page_title="Excel Translator Pro",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced aesthetics
st.markdown("""
    <style>
        /* Global styles */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        }
        
        /* Main container */
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header and text styles */
        .stTitle {
            background: linear-gradient(120deg, #1E88E5, #1976D2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 2rem !important;
        }
        
        /* Card containers */
        .stCard {
            background-color: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            transition: transform 0.3s ease;
        }
        
        .stCard:hover {
            transform: translateY(-5px);
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #1E88E5, #1976D2);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #1976D2, #1565C0);
            box-shadow: 0 4px 15px rgba(30, 136, 229, 0.3);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #1E88E5, #1976D2);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed #1E88E5;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            background-color: rgba(30, 136, 229, 0.05);
        }
        
        /* Metrics container */
        .metrics-container {
            background-color: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: white;
            padding: 2rem 1rem;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: white;
            border-radius: 10px;
        }
        
        /* DataFrames */
        [data-testid="stDataFrame"] {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

def create_metrics_card(title, value):
    """Create a styled metric card"""
    st.markdown(f"""
        <div class="metrics-container">
            <h4 style="color: #666; margin-bottom: 0.5rem; font-size: 0.9rem;">{title}</h4>
            <h2 style="color: #1E88E5; margin: 0; font-size: 1.8rem;">{value}</h2>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Get API key from secrets
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error('OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml')
        return
    
    # Initialize OpenAI client with secret key
    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Excel Translator Pro")
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        🔄 This application translates Excel files from German to English using OpenAI's advanced language models.
        
        ✨ Features:
        - Multiple column translation
        - Batch processing
        - Progress tracking
        - Preview capabilities
        """)
        st.markdown("---")
        st.markdown("### Settings")
        show_original = st.toggle("📊 Show original text", value=True)
        batch_size = st.slider("📦 Batch Size", 5, 50, 10)

    # Main content
    st.title("🌐 Excel Translator Pro")
    st.markdown("""
        <p style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>
        Transform your German Excel documents into English with advanced AI translation
        </p>
    """, unsafe_allow_html=True)

    # File upload section
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📂 Choose your Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file containing German text to translate"
        )

    if uploaded_file:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Display file statistics
        with col2:
            create_metrics_card("📊 Total Rows", f"{len(df):,}")
            create_metrics_card("📑 Total Columns", len(df.columns))

        # Column selection
        st.markdown("### 🎯 Select Columns to Translate")
        columns_to_translate = st.multiselect(
            "Choose the columns containing German text:",
            options=df.columns.tolist(),
            default=['MeasureDescription', 'CauseDescription', 'TechnicalObjectDescription']
        )

        if columns_to_translate:
            # Preview original data
            if show_original:
                st.markdown("### 📊 Original Data Preview")
                st.dataframe(
                    df[columns_to_translate].head(),
                    use_container_width=True,
                    height=200
                )

            # Translation process
            if st.button("🚀 Start Translation", use_container_width=True):
                try:
                    progress_placeholder = st.empty()
                    
                    with st.spinner("🔄 Translation in progress..."):
                        start_time = time.time()
                        translated_df = process_excel(
                            df=df,
                            translate_col=columns_to_translate,
                            client=client,
                            batch_size=batch_size,
                            progress_callback=lambda x: progress_placeholder.progress(x)
                        )
                        end_time = time.time()

                    st.success(f"""
                        ✨ Translation completed successfully!
                        ⏱️ Time taken: {end_time - start_time:.2f} seconds
                        📊 Columns translated: {len(columns_to_translate)}
                    """)

                    # Preview translated data
                    st.markdown("### 🎉 Translation Preview")
                    preview_columns = []
                    for col in columns_to_translate:
                        preview_columns.extend([col, f"{col}_EN"])
                    st.dataframe(
                        translated_df[preview_columns].head(),
                        use_container_width=True,
                        height=300
                    )

                    # Download section
                    st.markdown("### 💾 Download Results")
                    buffer = BytesIO()
                    translated_df.to_excel(buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Translated Excel",
                        data=buffer.getvalue(),
                        file_name="translated_document.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
