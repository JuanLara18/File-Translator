import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import json
import os
from time import sleep
import pickle
from datetime import datetime
from dotenv import load_dotenv
import openai
from openai import OpenAI

class FaultClassificationPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.checkpoint_dir = os.path.join(data_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load environment variables and set up OpenAI client
        load_dotenv("OPENAI_API_KEY.env")
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def load_data(self, file_path):
        """Load and preprocess the data."""
        print("Loading data...")
        df = pd.read_stata(file_path)
        
        cols = ["MeasureDescription_EN", "CauseDescription_EN", "TechnicalObjectDescription_EN"]
        df[cols] = df[cols].fillna("")
        
        # Combine text fields
        df["text_combined"] = df[cols].apply(
            lambda row: " ".join([cell.strip() for cell in row if cell.strip() != ""]),
            axis=1
        )
        
        return df
    
    def get_cluster_labels_with_retry(self, terms, cluster_num, k_size="small", max_retries=3):
        """Get cluster labels from GPT with multiple retries and specific prompts based on k size."""
        base_prompt = {
            "small": (
                f"You are an expert in industrial maintenance and fault classification. "
                f"Given these key terms from a cluster of automotive manufacturing faults: {', '.join(terms)}. "
                f"Provide a single broad category name (like 'Mechanical', 'Electrical', 'Software', etc.) "
                f"followed by a short description. Format: 'Category: <name>\\nDescription: <description>'"
            ),
            "large": (
                f"You are an expert in industrial maintenance and fault classification. "
                f"Given these key terms from a cluster of automotive manufacturing faults: {', '.join(terms)}. "
                f"Provide a specific fault category name (2-3 words max) "
                f"followed by a short description. Format: 'Category: <name>\\nDescription: <description>'"
            )
        }
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a precise industrial fault classification expert."},
                        {"role": "user", "content": base_prompt[k_size]}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                label = response.choices[0].message.content.strip()
                
                # Extract category name
                category = label.split('\n')[0].split(': ')[1].strip()
                
                return category
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep(2)
                else:
                    return f"Cluster_{cluster_num}"
    
    def perform_clustering(self, df, k, checkpoint_prefix):
        """Perform clustering and save results."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_prefix}_k{k}.pkl")
        
        if os.path.exists(checkpoint_path):
            print(f"Loading existing clustering for k={k}...")
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"Performing clustering for k={k}...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df["text_combined"])
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Get top terms for each cluster
        cluster_terms = {}
        cluster_labels = {}
        
        for i in range(k):
            center = kmeans.cluster_centers_[i]
            top_indices = center.argsort()[-10:][::-1]
            terms = [vectorizer.get_feature_names_out()[j] for j in top_indices]
            cluster_terms[i] = terms
            
            # Get label based on k size
            k_size = "small" if k <= 3 else "large"
            label = self.get_cluster_labels_with_retry(terms, i, k_size)
            cluster_labels[i] = label
        
        results = {
            'labels': labels,
            'cluster_labels': cluster_labels
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(results, f)
        
        return results
    
    def run_pipeline(self, input_file, k_small=2, k_large=15):
        """Run the complete classification pipeline."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data
        df = self.load_data(input_file)
        
        # Perform two-level clustering
        small_clusters = self.perform_clustering(df, k_small, f"small_clusters_{timestamp}")
        large_clusters = self.perform_clustering(df, k_large, f"large_clusters_{timestamp}")
        
        # Create new columns all at once to avoid fragmentation
        new_columns = pd.DataFrame({
            'coarse_cluster': small_clusters['labels'],
            'coarse_label': pd.Series(small_clusters['labels']).map(small_clusters['cluster_labels']),
            'fine_cluster': large_clusters['labels'],
            'fine_label': pd.Series(large_clusters['labels']).map(large_clusters['cluster_labels'])
        })
        
        # Combine with original dataframe efficiently
        df = pd.concat([df, new_columns], axis=1)
        
        # Save results
        output_file = os.path.join(
            self.data_dir,
            f"Maintenance_Full_Translated_Clustered_{timestamp}.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Save cluster mappings
        mappings = {
            'coarse_clusters': small_clusters['cluster_labels'],
            'fine_clusters': large_clusters['cluster_labels']
        }
        
        with open(os.path.join(self.data_dir, f"cluster_mappings_{timestamp}.json"), 'w') as f:
            json.dump(mappings, f, indent=2)
        
        return df

def main():
    # Configuration
    DATA_DIR = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation"
    INPUT_FILE = os.path.join(DATA_DIR, "Maintenance_Full_Translated.dta")
    
    # Initialize and run pipeline
    pipeline = FaultClassificationPipeline(DATA_DIR)
    df_results = pipeline.run_pipeline(
        input_file=INPUT_FILE,
        k_small=2,  # For broad categories (e.g., mechanical vs electrical)
        k_large=15  # For specific fault types
    )

if __name__ == "__main__":
    main()