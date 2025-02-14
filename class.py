import os
import sys
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import logging
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    filename='error_classification.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_openai() -> OpenAI:
    """Initialize OpenAI client with error handling."""
    load_dotenv("OPENAI_API_KEY.env")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in OPENAI_API_KEY.env")
    return OpenAI(api_key=api_key)

def create_classification_prompt(measure: str, cause: str, tech_obj: str) -> str:
    """Create a detailed prompt for GPT to classify the error type."""
    return f"""As an industrial maintenance expert, analyze these error descriptions and classify if the issue is primarily MECHANICAL or ELECTRICAL. Respond with ONLY ONE WORD: either 'MECHANICAL' or 'ELECTRICAL'.

Technical details:
1. Measure taken: {measure}
2. Cause description: {cause}
3. Technical object: {tech_obj}

Consider:
- Mechanical issues involve physical components, movement, wear, structural problems
- Electrical issues involve power, circuits, sensors, controls, signals

Classify as MECHANICAL or ELECTRICAL:"""

def classify_error(client: OpenAI, measure: str, cause: str, tech_obj: str) -> str:
    """
    Classify an error as mechanical or electrical using GPT.
    Returns 'UNKNOWN' if classification fails.
    """
    # Handle empty or invalid inputs
    if not any(isinstance(x, str) and x.strip() for x in [measure, cause, tech_obj]):
        return 'UNKNOWN'
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in industrial maintenance and error classification."},
                {"role": "user", "content": create_classification_prompt(
                    str(measure) if pd.notna(measure) else "",
                    str(cause) if pd.notna(cause) else "",
                    str(tech_obj) if pd.notna(tech_obj) else ""
                )}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        classification = response.choices[0].message.content.strip().upper()
        return classification if classification in ['MECHANICAL', 'ELECTRICAL'] else 'UNKNOWN'
    
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        return 'UNKNOWN'

def create_cache_key(measure: str, cause: str, tech_obj: str) -> str:
    """Create a unique key for caching classifications."""
    return f"{measure}|||{cause}|||{tech_obj}"

def process_file(input_path: str, output_path: str) -> None:
    """
    Process the input file and create a new one with error classifications.
    """
    try:
        # Initialize OpenAI client
        client = setup_openai()
        
        # Load the data
        logging.info(f"Loading data from {input_path}")
        df = pd.read_stata(input_path)
        
        # Create classification cache
        cache: Dict[str, str] = {}
        
        # Prepare for classification
        total_rows = len(df)
        classifications = []
        
        logging.info(f"Starting classification of {total_rows} rows")
        
        # Process each row with progress bar
        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Classifying errors"):
            # Create cache key
            cache_key = create_cache_key(
                str(row.get('MeasureDescription_EN', '')),
                str(row.get('CauseDescription_EN', '')),
                str(row.get('TechnicalObjectDescription_EN', ''))
            )
            
            # Check cache first
            if cache_key in cache:
                classifications.append(cache[cache_key])
            else:
                # Classify and cache
                classification = classify_error(
                    client,
                    row.get('MeasureDescription_EN', ''),
                    row.get('CauseDescription_EN', ''),
                    row.get('TechnicalObjectDescription_EN', '')
                )
                cache[cache_key] = classification
                classifications.append(classification)
        
        # Add classifications to dataframe
        df['ErrorType'] = classifications
        
        # Save results
        df.to_stata(output_path, version=118)
        
        # Log statistics
        mechan_count = sum(1 for x in classifications if x == 'MECHANICAL')
        elect_count = sum(1 for x in classifications if x == 'ELECTRICAL')
        unknown_count = sum(1 for x in classifications if x == 'UNKNOWN')
        
        logging.info(f"""
Classification complete:
- Total records: {total_rows}
- Mechanical: {mechan_count} ({mechan_count/total_rows*100:.1f}%)
- Electrical: {elect_count} ({elect_count/total_rows*100:.1f}%)
- Unknown: {unknown_count} ({unknown_count/total_rows*100:.1f}%)
        """)
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise

def main():
    # File paths
    input_path = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation/Maintenance_Full_Translated.dta"
    output_path = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation/Maintenance_Full_Classified.dta"
    
    start_time = datetime.now()
    
    try:
        process_file(input_path, output_path)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\nClassification completed successfully in {elapsed_time:.1f} seconds")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()