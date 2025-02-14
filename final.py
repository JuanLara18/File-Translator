import pandas as pd
import os
import glob
from datetime import datetime

def get_latest_classification_file(directory):
    """Find the most recent classification file in the directory."""
    pattern = os.path.join(directory, "Maintenance_Full_Translated_Clustered_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No classification files found matching pattern: {pattern}")
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Found latest classification file: {latest_file}")
    return latest_file

def merge_classifications():
    base_dir = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation"
    
    try:
        # Load original translated data
        original_file = os.path.join(base_dir, "Maintenance_Full_Translated.dta")
        print(f"Loading original data from: {original_file}")
        df_original = pd.read_stata(original_file)
        print(f"Original data shape: {df_original.shape}")
        
        # Find and load the most recent classifications file
        classifications_file = get_latest_classification_file(base_dir)
        print(f"Loading classifications from: {classifications_file}")
        # Add low_memory=False to avoid dtype warnings
        df_classifications = pd.read_csv(classifications_file, low_memory=False)
        print(f"Classifications data shape: {df_classifications.shape}")
        
        # Verify dimensions
        if len(df_original) != len(df_classifications):
            raise ValueError(f"Number of rows doesn't match: Original={len(df_original)}, Classifications={len(df_classifications)}")
        
        # Create new dataframe with all columns at once to avoid fragmentation
        classification_cols = ['coarse_cluster', 'coarse_label', 'fine_cluster', 'fine_label']
        new_cols = df_classifications[classification_cols]
        
        # Efficiently combine dataframes
        df_final = pd.concat([df_original, new_cols], axis=1)
        
        # Generate output filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV (all data)
        csv_output = os.path.join(base_dir, f"Maintenance_Full_Translated_Final_{timestamp}.csv")
        print(f"Saving complete dataset as CSV: {csv_output}")
        df_final.to_csv(csv_output, index=False)
        
        # For Stata output, handle string length limitations
        stata_output = os.path.join(base_dir, f"Maintenance_Full_Translated_Final_{timestamp}.dta")
        print(f"Saving Stata-compatible dataset: {stata_output}")
        
        # Convert long string columns to shorter versions for Stata compatibility
        df_stata = df_final.copy()
        for col in df_stata.select_dtypes(include=['object']).columns:
            # Check if column contains strings longer than 244 characters
            max_len = df_stata[col].str.len().max() if not df_stata[col].isna().all() else 0
            if max_len > 244:
                print(f"Truncating column {col} from max length {max_len} to 244 characters")
                df_stata[col] = df_stata[col].str.slice(0, 244)
        
        # Save as Stata file with version 117 (Stata 13+)
        df_stata.to_stata(stata_output, version=117, write_index=False)
        
        # Print summary
        print("\nOperation Summary:")
        print(f"- Original rows: {len(df_final)}")
        print(f"- Added columns: {classification_cols}")
        print(f"- Complete CSV saved as: {os.path.basename(csv_output)}")
        print(f"- Stata-compatible file saved as: {os.path.basename(stata_output)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please verify that all paths are correct and files exist.")
        raise

if __name__ == "__main__":
    merge_classifications()