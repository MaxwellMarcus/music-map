#!/usr/bin/env python3
"""
Script to generate names for music clusters using the MusicGemma model.
Queries the model with cluster centroids and saves the generated names.
"""

import os
import sys
import numpy as np
import json
import torch

from textmodel.MusicGemma import get_model_and_such
from textmodel.query_with_music import query_with_music

def load_cluster_data():
    """Load cluster assignments and original embeddings."""
    clusters_file = "./music-clips-hdbscan.npy"
    embeddings_file = "./music-clips-embeddings.npy"
    
    if not os.path.exists(clusters_file):
        print(f"Error: {clusters_file} not found!")
        print("Please run generate_clusters.py first to create clusters.")
        return None, None
    
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found!")
        print("Please ensure you have the original embeddings file.")
        return None, None
    
    print(f"Loading clusters from {clusters_file}...")
    clusters = np.load(clusters_file)
    
    print(f"Loading original embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    
    print(f"Loaded clusters shape: {clusters.shape}")
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    return clusters, embeddings

def calculate_embedding_centroids(clusters, embeddings):
    """Calculate centroids from original embeddings for each cluster."""
    print("Calculating embedding centroids for clusters...")
    
    unique_clusters = sorted(set(clusters))
    embedding_centroids = {}
    
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # Skip noise points
            # Get indices of points belonging to this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Get the original embeddings for these points
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate the centroid (mean) of the embeddings
            centroid = np.mean(cluster_embeddings, axis=0)
            embedding_centroids[cluster_id] = centroid
            
            print(f"Cluster {cluster_id}: {len(cluster_indices)} points, centroid shape: {centroid.shape}")
    
    return embedding_centroids

def generate_cluster_names(model, processor, embedding_centroids):
    """Generate names for each cluster using the MusicGemma model."""
    cluster_names = {}
    query = "Describe this music."
    
    print("Generating names for clusters...")
    
    for cluster_id, centroid in embedding_centroids.items():
        print(f"Processing cluster {cluster_id}...")
        
        try:
            # Ensure the centroid is in the right format (512D float32)
            music_data = centroid.astype(np.float32)
            
            # Query the model
            generated_text = query_with_music(model, processor, query, music_data, [])
            
            # Clean up the generated text (remove the prompt part)
            generated_text = generated_text.split("model")[-1].strip()
            
            # Limit the length and clean up
            if generated_text:
                cluster_names[int(cluster_id)] = generated_text
                print(f"Cluster {cluster_id}: {generated_text}")
            else:
                cluster_names[int(cluster_id)] = f"Cluster {cluster_id}"
                print(f"Cluster {cluster_id}: Generated empty text, using default name")
                
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            cluster_names[int(cluster_id)] = f"Cluster {cluster_id}"
    
    return cluster_names

def save_cluster_names(cluster_names):
    """Save cluster names to a JSON file."""
    output_file = "./cluster_names.json"
    with open(output_file, 'w') as f:
        json.dump(cluster_names, f, indent=2)
    print(f"Saved cluster names to {output_file}")
    return output_file

def main():
    """Main function to generate cluster names."""
    print("Loading MusicGemma model...")
    
    # try:
        # Load the model and processor
    model, processor, tokenizer = get_model_and_such()
    print("Model loaded successfully!")
    
    # Load cluster data and original embeddings
    clusters, embeddings = load_cluster_data()
    if clusters is None or embeddings is None:
        return
    
    # Calculate centroids from original embeddings
    embedding_centroids = calculate_embedding_centroids(clusters, embeddings)
    
    # Generate names for clusters
    cluster_names = generate_cluster_names(model, processor, embedding_centroids)
    
    # Save the names
    output_file = save_cluster_names(cluster_names)
    
    print(f"\nCluster naming complete!")
    print(f"Generated names for {len(cluster_names)} clusters")
    print(f"Names saved to: {output_file}")
    
    # Display the generated names
    print("\nGenerated cluster names:")
    for cluster_id, name in sorted(cluster_names.items()):
        print(f"Cluster {cluster_id}: {name}")
            
    # except Exception as e:
    #     print(f"Error: {e}")
    #     print("Make sure you have the required dependencies and model files.")
    #     return

if __name__ == "__main__":
    main()
