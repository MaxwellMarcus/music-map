#!/usr/bin/env python3
"""
Test script to verify the vibe search functionality.
"""

import requests
import json
import sys

def test_similarity_search_api():
    """Test the similarity search API endpoint."""
    print("Testing vibe search functionality...\n")
    
    # Test queries
    test_queries = [
        "jazz",
        "rock music",
        "electronic dance",
        "classical piano",
        "hip hop beats"
    ]
    
    for query in test_queries:
        print(f"Testing query: '{query}'")
        
        try:
            # Make request to similarity search API
            response = requests.get(f'http://localhost:8000/api/similarity-search?query={query}')
            
            if response.status_code == 200:
                result = response.json()
                
                if 'similarities' in result:
                    similarities = result['similarities']
                    max_sim = max(similarities)
                    min_sim = min(similarities)
                    avg_sim = sum(similarities) / len(similarities)
                    
                    print(f"  ✅ Success!")
                    print(f"     Similarities: {len(similarities)} points")
                    print(f"     Max similarity: {max_sim:.3f}")
                    print(f"     Min similarity: {min_sim:.3f}")
                    print(f"     Avg similarity: {avg_sim:.3f}")
                    
                    # Show top 3 most similar points
                    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
                    print(f"     Top 3 indices: {top_indices}")
                    
                else:
                    print(f"  ❌ No similarities in response")
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
    print("Vibe Search Test Suite\n")
    
    # Check if server is running
    if not test_server_status():
        print("\nPlease start the server first:")
        print("python app.py")
        return
    
    print()
    
    # Test similarity search
    test_similarity_search_api()
    
    print("Test complete!")
    print("\nTo use vibe search in the web interface:")
    print("1. Open http://localhost:8000")
    print("2. Click the search bar")
    print("3. Click the 'Artist Search' button to switch to 'Vibe Search'")
    print("4. Type a query like 'jazz' or 'rock music'")
    print("5. Watch the colors change based on similarity!")

if __name__ == "__main__":
    main()
