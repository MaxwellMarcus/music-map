#!/usr/bin/env python3
"""
Test script to verify the similarity index functionality.
"""

import requests
import json
import sys
import time

def test_similarity_index_creation():
    """Test that similarity search creates a proper index."""
    print("Testing similarity index creation...\n")
    
    # Test queries
    test_queries = [
        "jazz",
        "rock music",
        "electronic dance"
    ]
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        
        try:
            # Make request to similarity search API
            response = requests.get(f'http://localhost:8000/api/similarity-search?query={query}')
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'success':
                    print(f"  ✅ Success!")
                    print(f"     Index name: {result.get('index_name')}")
                    print(f"     Display name: {result.get('display_name')}")
                    print(f"     Query: {result.get('query')}")
                    print(f"     Similarities: {len(result.get('similarities', []))} points")
                    
                    # Check if the index was created by looking at available indexes
                    time.sleep(1)  # Wait a bit for the index to be created
                    
                    # Test getting color data to see if the new index is active
                    color_response = requests.get('http://localhost:8000/api/color-data')
                    if color_response.status_code == 200:
                        color_result = color_response.json()
                        print(f"     Color source: {color_result.get('color_source')}")
                        
                        if 'similarity' in color_result.get('color_source', ''):
                            print(f"     ✅ Index is now active as color source!")
                        else:
                            print(f"     ⚠️  Index not yet active as color source")
                    else:
                        print(f"     ❌ Failed to get color data")
                    
                else:
                    print(f"  ❌ No success status in response")
                    print(f"     Response: {result}")
                    
            else:
                print(f"  ❌ HTTP {response.status_code}")
                print(f"     Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection error - make sure the server is running on localhost:8000")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
        
        # Wait between queries to avoid overwhelming the server
        time.sleep(2)

def test_color_source_switching():
    """Test switching between different color sources."""
    print("Testing color source switching...\n")
    
    try:
        # Test switching back to cluster colors
        response = requests.post('http://localhost:8000/api/set-color-source', 
                               json={
                                   'color_source': 'index:hdbscan_clusters',
                                   'source_name': 'hdbscan_clusters',
                                   'source_data_manager': 'primary'
                               })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully switched to: {result.get('current_color_source')}")
        else:
            print(f"❌ Failed to switch color source: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing color source switching: {e}")

def test_server_status():
    """Test if the server is running."""
    try:
        response = requests.get('http://localhost:8000/api/status')
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print("❌ Server responded with error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running - start with: python app.py")
        return False

def main():
    """Run all tests."""
    print("Similarity Index Test Suite\n")
    
    # Check if server is running
    if not test_server_status():
        print("\nPlease start the server first:")
        print("python app.py")
        return
    
    print()
    
    # Test similarity index creation
    test_similarity_index_creation()
    
    print()
    
    # Test color source switching
    test_color_source_switching()
    
    print("\nTest complete!")
    print("\nTo use similarity search in the web interface:")
    print("1. Open http://localhost:8000")
    print("2. Click the search bar")
    print("3. Use the dropdown to select 'Vibe Search' mode")
    print("4. Type a query like 'jazz' or 'rock music'")
    print("5. Press Enter to perform the similarity search")
    print("6. The similarity index will be created and applied as a color source")
    print("7. All points will be colored automatically by Plotly based on similarity scores")

if __name__ == "__main__":
    main()
