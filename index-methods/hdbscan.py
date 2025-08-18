import hdbscan

def index(data, **kwargs):
    """Create clusters using HDBSCAN and return as index values"""
    cluster_labels = hdbscan.HDBSCAN(**kwargs).fit_predict(data)
    
    # Convert cluster labels to index format
    # HDBSCAN returns -1 for noise points, we'll keep those as -1
    return {"index": cluster_labels}