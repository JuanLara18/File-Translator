import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from collections import Counter
import umap
import warnings
import os
import shutil
from joblib import Parallel, delayed, dump, load
from datetime import datetime
import tempfile
from pathlib import Path
warnings.filterwarnings('ignore')

class ClusteringPipeline:
    def __init__(self, input_file, n_jobs=-1):
        """Initialize the pipeline with input file and number of parallel jobs."""
        self.input_file = input_file
        self.n_jobs = n_jobs
        self.temp_dir = Path(tempfile.mkdtemp(prefix='clustering_'))
        print(f"Temporary directory created at: {self.temp_dir}")
        
    def __del__(self):
        """Cleanup temporary directory when the object is destroyed."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("Temporary directory cleaned up")

    def save_checkpoint(self, data, filename):
        """Save a checkpoint to the temporary directory."""
        filepath = self.temp_dir / filename
        dump(data, filepath)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Load a checkpoint from the temporary directory."""
        filepath = self.temp_dir / filename
        if filepath.exists():
            return load(filepath)
        return None

    def load_and_preprocess(self):
        """Load and preprocess the data with parallel processing."""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_stata(self.input_file)
        
        # Define columns of interest
        cols = [
            "MeasureDescription_EN",
            "CauseDescription_EN", 
            "TechnicalObjectDescription_EN",
            "CauseLongText_EN",
            "DamagePatternLongText_EN",
            "LongText_EN",
            "MeasureLongText_EN"
        ]
        df[cols] = df[cols].fillna("")
        
        # Parallel processing for text combination
        def process_row(row):
            return " ".join([str(row[col]).strip() for col in cols if str(row[col]).strip() != ""])

        # Filter empty rows and combine text in parallel
        df["all_empty"] = df[cols].apply(lambda row: all(str(cell).strip() == "" for cell in row), axis=1)
        df_filtered = df[~df["all_empty"]].copy()
        
        # Process text combination in parallel
        df_filtered["text_combined"] = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(row) for _, row in df_filtered[cols].iterrows()
        )
        
        print(f"Processed {len(df_filtered)} valid records out of {len(df)} total records")
        
        # Save checkpoint
        self.save_checkpoint(df_filtered, "preprocessed_data.joblib")
        return df_filtered

    def vectorize_text(self, df, max_features=1000):
        """Vectorize text using TF-IDF with parallel processing."""
        print("Vectorizing text...")
        
        checkpoint = self.load_checkpoint("vectorized_data.joblib")
        if checkpoint is not None:
            return checkpoint

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            n_jobs=self.n_jobs  # Parallel processing for vectorization
        )
        
        X = vectorizer.fit_transform(df["text_combined"])
        print(f"Created {X.shape[1]} features from text")
        
        # Save checkpoint
        self.save_checkpoint((X, vectorizer), "vectorized_data.joblib")
        return X, vectorizer

    def calculate_metrics_parallel(self, X, k):
        """Calculate clustering metrics for a specific k in parallel."""
        kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=self.n_jobs)
        labels = kmeans.fit_predict(X)
        
        # Reduce dimensionality for silhouette analysis
        svd = TruncatedSVD(n_components=min(100, X.shape[1]-1))
        X_reduced = svd.fit_transform(X)
        
        return {
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_reduced, labels),
            'ch_score': calinski_harabasz_score(X_reduced, labels)
        }

    def determine_optimal_clusters(self, X, max_k=20):
        """Determine optimal number of clusters using parallel processing."""
        print("Determining optimal number of clusters...")
        
        checkpoint = self.load_checkpoint("optimal_k.joblib")
        if checkpoint is not None:
            return checkpoint

        # Calculate metrics in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.calculate_metrics_parallel)(X, k)
            for k in range(2, max_k + 1)
        )
        
        # Extract metrics
        ks = [r['k'] for r in results]
        inertias = [r['inertia'] for r in results]
        silhouette_scores = [r['silhouette'] for r in results]
        ch_scores = [r['ch_score'] for r in results]
        
        # Find optimal k using different methods
        kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
        k_elbow = kn.elbow
        k_silhouette = ks[silhouette_scores.index(max(silhouette_scores))]
        k_ch = ks[ch_scores.index(max(ch_scores))]
        
        # Plot results
        self.plot_metrics(ks, inertias, silhouette_scores, ch_scores, k_elbow, k_silhouette, k_ch)
        
        # Calculate consensus k
        optimal_k = Counter([k_elbow, k_silhouette, k_ch]).most_common(1)[0][0]
        
        # Save checkpoint
        self.save_checkpoint(optimal_k, "optimal_k.joblib")
        return optimal_k

    def plot_metrics(self, ks, inertias, silhouette_scores, ch_scores, k_elbow, k_silhouette, k_ch):
        """Plot clustering metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Elbow curve
        axes[0].plot(ks, inertias)
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('Inertia')
        axes[0].axvline(x=k_elbow, color='r', linestyle='--')
        
        # Silhouette score
        axes[1].plot(ks, silhouette_scores)
        axes[1].set_title('Silhouette Score')
        axes[1].set_xlabel('k')
        axes[1].set_ylabel('Score')
        axes[1].axvline(x=k_silhouette, color='r', linestyle='--')
        
        # Calinski-Harabasz score
        axes[2].plot(ks, ch_scores)
        axes[2].set_title('Calinski-Harabasz Score')
        axes[2].set_xlabel('k')
        axes[2].set_ylabel('Score')
        axes[2].axvline(x=k_ch, color='r', linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.temp_dir / 'metrics_plot.png')
        plt.close()

    def perform_clustering(self, X, optimal_k, vectorizer):
        """Perform clustering with optimal k using parallel processing."""
        print(f"\nPerforming clustering with k={optimal_k}...")
        
        checkpoint = self.load_checkpoint("clustering_results.joblib")
        if checkpoint is not None:
            return checkpoint

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_jobs=self.n_jobs)
        labels = kmeans.fit_predict(X)
        
        # Get top terms per cluster in parallel
        def get_top_terms_parallel(center, terms, n_terms=10):
            top_indices = center.argsort()[-n_terms:][::-1]
            return [terms[i] for i in top_indices]

        terms = vectorizer.get_feature_names_out()
        top_terms = Parallel(n_jobs=self.n_jobs)(
            delayed(get_top_terms_parallel)(center, terms)
            for center in kmeans.cluster_centers_
        )
        
        print("\nTop terms per cluster:")
        for i, terms in enumerate(top_terms):
            print(f"\nCluster {i}:")
            print(", ".join(terms))
        
        # Save checkpoint
        self.save_checkpoint(labels, "clustering_results.joblib")
        return labels

    def visualize_clusters(self, X, labels):
        """Create UMAP visualization of clusters."""
        print("\nCreating cluster visualization...")
        
        # Reduce dimensionality for visualization
        reducer = umap.UMAP(random_state=42, n_neighbors=15, n_components=2)
        X_reduced = reducer.fit_transform(X.toarray())
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab20')
        plt.colorbar(scatter)
        plt.title('UMAP visualization of clusters')
        
        # Save plot
        plt.savefig(self.temp_dir / 'cluster_visualization.png')
        plt.close()

def main():
    # File paths
    input_file = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/02 Translation/Maintenance_Full_Translated_Complete.dta"
    output_dir = Path("/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/02 Translation")
    
    # Create pipeline instance
    pipeline = ClusteringPipeline(input_file)
    
    try:
        # Execute pipeline
        df = pipeline.load_and_preprocess()
        X, vectorizer = pipeline.vectorize_text(df)
        optimal_k = pipeline.determine_optimal_clusters(X)
        labels = pipeline.perform_clustering(X, optimal_k, vectorizer)
        pipeline.visualize_clusters(X, labels)
        
        # Add cluster labels to dataframe
        df['cluster'] = labels
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"Maintenance_Full_Translated_Clustered_Optimal_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Check temporary directory for intermediate results:", pipeline.temp_dir)
        raise
    
if __name__ == "__main__":
    main()