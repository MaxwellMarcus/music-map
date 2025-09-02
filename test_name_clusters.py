#!/usr/bin/env python3
"""
Test script to verify the cluster naming system.
"""

import os
import json
import numpy as np

def test_cluster_names_file():
    """Test if cluster names file exists and can be loaded."""
    cluster_names_file = "./cluster_names.json"
    
    if not os.path.exists(cluster_names_file):
        print(f"❌ Cluster names file not found: {cluster_names_file}")
        print("   Please run name_clusters.py first to generate cluster names.")
        return False
    
    try:
        with open(cluster_names_file, 'r') as f:
            cluster_names = json.load(f)
        
        print(f"✅ Cluster names file loaded successfully")
        print(f"   Found {len(cluster_names)} cluster names")
        
        # Display some example names
        print("\nExample cluster names:")
        for i, (cluster_id, name) in enumerate(list(cluster_names.items())[:5]):
            print(f"   Cluster {cluster_id}: {name}")
        
        if len(cluster_names) > 5:
            print(f"   ... and {len(cluster_names) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading cluster names: {e}")
        return False

def test_cluster_data_files():
    """Test if required cluster data files exist."""
    required_files = [
        "./music-clips-hdbscan.npy",
        "./music-clips-embeddings.npy",
        "./music-clips-centroids.npy"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("Testing cluster naming system...\n")
    
    # Test cluster data files
    print("1. Checking cluster data files:")
    data_files_ok = test_cluster_data_files()
    print()
    
    # Test cluster names file
    print("2. Checking cluster names file:")
    names_file_ok = test_cluster_names_file()
    print()
    
    # Summary
    if data_files_ok and names_file_ok:
        print("🎉 All tests passed! The cluster naming system is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if not data_files_ok:
            print("\nTo fix data file issues:")
            print("   - Run generate_clusters.py to create cluster data")
            print("   - Ensure music-clips-embeddings.npy exists")
        
        if not names_file_ok:
            print("\nTo fix cluster names issues:")
            print("   - Run name_clusters.py to generate cluster names")
            print("   - Ensure the MusicGemma model is properly set up")

if __name__ == "__main__":
    main()
