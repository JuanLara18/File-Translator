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
warnings.filterwarnings('ignore')

def load_and_preprocess(file_path):
    """Load and preprocess the data."""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_stata(file_path)
    
    # Define columns of interest
    cols = ["MeasureDescription_EN", "CauseDescription_EN", "TechnicalObjectDescription_EN"]
    df[cols] = df[cols].fillna("")
    
    # Filter rows where all columns are empty
    df["all_empty"] = df[cols].apply(lambda row: all(cell.strip() == "" for cell in row), axis=1)
    df_filtered = df[~df["all_empty"]].copy()
    
    # Combine text fields
    df_filtered["text_combined"] = df_filtered.apply(
        lambda row: " ".join([row[col].strip() for col in cols if row[col].strip() != ""]), 
        axis=1
    )
    
    print(f"Processed {len(df_filtered)} valid records out of {len(df)} total records")
    return df_filtered

def vectorize_text(df, max_features=1000):
    """Vectorize text using TF-IDF."""
    print("Vectorizing text...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=5,  # Minimum document frequency
        max_df=0.95  # Maximum document frequency
    )
    
    X = vectorizer.fit_transform(df["text_combined"])
    print(f"Created {X.shape[1]} features from text")
    return X, vectorizer

def determine_optimal_clusters(X, max_k=20):
    """Determine optimal number of clusters using multiple metrics."""
    print("Determining optimal number of clusters...")
    
    # Initialize metrics
    inertias = []
    silhouette_scores = []
    ch_scores = []
    
    # Reduce dimensionality for silhouette analysis
    svd = TruncatedSVD(n_components=100)
    X_reduced = svd.fit_transform(X)
    
    # Calculate metrics for different k values
    for k in range(2, max_k + 1):
        print(f"Testing k={k}...", end='\r')
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_reduced, labels))
        ch_scores.append(calinski_harabasz_score(X_reduced, labels))
    
    # Find optimal k using elbow method
    kn = KneeLocator(
        range(2, max_k + 1), inertias, 
        curve='convex', direction='decreasing'
    )
    k_elbow = kn.elbow
    
    # Find optimal k using silhouette score
    k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Find optimal k using Calinski-Harabasz score
    k_ch = ch_scores.index(max(ch_scores)) + 2
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Elbow curve
    axes[0].plot(range(2, max_k + 1), inertias)
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].axvline(x=k_elbow, color='r', linestyle='--')
    
    # Silhouette score
    axes[1].plot(range(2, max_k + 1), silhouette_scores)
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score')
    axes[1].axvline(x=k_silhouette, color='r', linestyle='--')
    
    # Calinski-Harabasz score
    axes[2].plot(range(2, max_k + 1), ch_scores)
    axes[2].set_title('Calinski-Harabasz Score')
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('Score')
    axes[2].axvline(x=k_ch, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    print("\nOptimal number of clusters:")
    print(f"Elbow method: {k_elbow}")
    print(f"Silhouette score: {k_silhouette}")
    print(f"Calinski-Harabasz score: {k_ch}")
    
    # Return consensus k (mode of the three methods)
    optimal_k = Counter([k_elbow, k_silhouette, k_ch]).most_common(1)[0][0]
    return optimal_k

def perform_clustering(X, optimal_k, vectorizer):
    """Perform clustering with optimal k."""
    print(f"\nPerforming clustering with k={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Get top terms per cluster
    def get_top_terms(cluster_center, n_terms=10):
        terms = vectorizer.get_feature_names_out()
        top_indices = cluster_center.argsort()[-n_terms:][::-1]
        return [terms[i] for i in top_indices]
    
    print("\nTop terms per cluster:")
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"\nCluster {i}:")
        print(", ".join(get_top_terms(center)))
    
    return labels

def visualize_clusters(X, labels):
    """Create UMAP visualization of clusters."""
    print("\nCreating cluster visualization...")
    
    # Reduce dimensionality for visualization
    reducer = umap.UMAP(random_state=42)
    X_reduced = reducer.fit_transform(X.toarray())
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('UMAP visualization of clusters')
    plt.show()

def main():
    # File path
    file_path = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation/Maintenance_Full_Translated.dta"
    
    # Load and preprocess data
    df = load_and_preprocess(file_path)
    
    # Vectorize text
    X, vectorizer = vectorize_text(df)
    
    # Determine optimal number of clusters
    optimal_k = determine_optimal_clusters(X)
    
    # Perform clustering
    labels = perform_clustering(X, optimal_k, vectorizer)
    
    # Visualize results
    visualize_clusters(X, labels)
    
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    # Save results
    output_path = "/export/projects1/rsadun_bmw/03 Workplace/Clean Data/Production/Translation/Maintenance_Full_Translated_Clustered_Optimal.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()