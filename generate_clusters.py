import numpy as np
import hdbscan
import os
from sklearn.cluster import KMeans

def generate_hdbscan_clusters():
    """Generate HDBSCAN clusters from t-SNE data and save to numpy file."""
    
    # Load t-SNE data
    tsne_file = "./music-clips-tsne.npy"
    if not os.path.exists(tsne_file):
        print(f"Error: {tsne_file} not found!")
        return
    
    print(f"Loading t-SNE data from {tsne_file}...")
    tsne_data = np.load(tsne_file)
    print(f"Loaded t-SNE data shape: {tsne_data.shape}")
    
    # Apply HDBSCAN clustering
    print("Applying HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200,  # Minimum points in a cluster
        min_samples=3,       # Minimum samples for core points
        #cluster_selection_epsilon=0.1,  # Distance threshold for cluster selection
        #cluster_selection_method='eom'  # Excess of Mass algorithm
    )
    
    clusters = clusterer.fit_predict(tsne_data)
    
    print(f"Clustering complete!")
    print(f"Number of clusters: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
    print(f"Number of noise points: {np.sum(clusters == -1)}")
    print(f"Cluster distribution: {np.bincount(clusters + 1)}")  # +1 to handle -1 noise points
    
    # Calculate cluster centroids
    print("Calculating cluster centroids...")
    unique_clusters = sorted(set(clusters))
    cluster_centroids = {}
    
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # Skip noise points
            cluster_points = tsne_data[clusters == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            cluster_centroids[cluster_id] = centroid
    
    # Save clusters to numpy file
    output_file = "./music-clips-hdbscan.npy"
    np.save(output_file, clusters)
    print(f"Saved clusters to {output_file}")
    
    # Save cluster centroids
    centroids_file = "./music-clips-centroids.npy"
    np.save(centroids_file, cluster_centroids)
    print(f"Saved cluster centroids to {centroids_file}")
    
    return clusters, cluster_centroids

if __name__ == "__main__":
    generate_hdbscan_clusters()
