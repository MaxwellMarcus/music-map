import numpy as np

def index(data, start, end, name):
    """Create an index that names points in a specific range"""
    n_points = len(data)
    
    # Validate range
    if start < 0 or end > n_points or start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, data length={n_points}")
    
    # Create index array: 1 for points in range, 0 for others
    index_values = np.zeros(n_points, dtype=int)
    index_values[start:end] = 1
    
    return {"index": index_values, "name": name}
