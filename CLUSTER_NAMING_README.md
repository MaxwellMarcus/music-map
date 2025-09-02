# Cluster Naming System

This system uses the MusicGemma model to automatically generate descriptive names for music clusters and displays them on the scatter plot visualization.

## Overview

The cluster naming system consists of:
1. **`name_clusters.py`** - Script to generate cluster names using MusicGemma
2. **API endpoint** - `/api/cluster-names` to serve cluster names
3. **Frontend integration** - Displays names as annotations on the scatter plot

## Prerequisites

1. **Required files:**
   - `music-clips-embeddings.npy` - Original music embeddings (512D)
   - `music-clips-hdbscan.npy` - Cluster assignments from HDBSCAN
   - `music-clips-centroids.npy` - Cluster centroids (optional, for reference)

2. **Dependencies:**
   ```bash
   pip install torch transformers peft
   ```

3. **MusicGemma model:**
   - Ensure the model files are in `text-model/` directory
   - The model should be properly configured for your system

## Usage

### 1. Generate Cluster Names

Run the cluster naming script:

```bash
python name_clusters.py
```

This will:
- Load the MusicGemma model
- Load cluster assignments and original embeddings
- Calculate centroids from original embeddings (not t-SNE)
- Query the model with "Describe this music." for each cluster
- Save generated names to `cluster_names.json`

### 2. Test the System

Verify everything is working:

```bash
python test_name_clusters.py
```

### 3. View in Web Interface

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open your browser to `http://localhost:8000`

3. The cluster names will appear as annotations on the scatter plot

## How It Works

### Backend Process

1. **Data Loading:**
   - Loads cluster assignments from HDBSCAN output
   - Loads original 512D embeddings (not t-SNE projections)
   - Maps cluster IDs to embedding indices

2. **Centroid Calculation:**
   - For each cluster, finds all points belonging to that cluster
   - Calculates the mean (centroid) of the original embeddings
   - This gives a representative 512D vector for each cluster

3. **Model Querying:**
   - Uses MusicGemma model with the prompt "Describe this music."
   - Inputs the 512D centroid embedding as music data
   - Generates descriptive text for each cluster

4. **Name Storage:**
   - Saves generated names to `cluster_names.json`
   - Maps cluster IDs to descriptive names

### Frontend Integration

1. **API Endpoint:**
   - `/api/cluster-names` serves the generated names
   - Returns JSON with cluster ID to name mapping

2. **Plot Annotations:**
   - Cluster names appear as text annotations on the scatter plot
   - Positioned at cluster centroids (from t-SNE coordinates)
   - Styled with white text on semi-transparent black background

3. **Hover Information:**
   - Point hover shows the cluster name instead of generic "Cluster X"
   - Integrates with existing metadata display

## File Structure

```
web-viewer/
├── name_clusters.py              # Main cluster naming script
├── test_name_clusters.py         # Test script
├── cluster_names.json            # Generated cluster names (output)
├── text-model/
│   ├── MusicGemma.py            # MusicGemma model implementation
│   └── query_with_music.py      # Music querying functions
├── app.py                        # Web server with cluster names API
└── templates/
    └── index.html               # Frontend with cluster name display
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors:**
   - Check that all required dependencies are installed
   - Verify MusicGemma model files are in place
   - Check system compatibility (MPS, CUDA, CPU)

2. **Missing Data Files:**
   - Run `generate_clusters.py` first to create cluster data
   - Ensure `music-clips-embeddings.npy` exists

3. **Empty Cluster Names:**
   - Check model output format in `query_with_music.py`
   - Verify the model is generating responses correctly

4. **Names Not Displaying:**
   - Check browser console for JavaScript errors
   - Verify `cluster_names.json` is properly formatted
   - Check that cluster centroids are loaded

### Debug Mode

Enable verbose logging by modifying the scripts:

```python
# In name_clusters.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Customization

### Changing the Prompt

Modify the query in `name_clusters.py`:

```python
query = "Describe this music in 3 words."  # Custom prompt
```

### Styling Annotations

Modify the annotation style in `templates/index.html`:

```javascript
annotations.push({
    // ... existing properties
    font: {
        size: 16,                    // Larger text
        color: '#ff0000'            // Red text
    },
    bgcolor: 'rgba(255, 255, 0, 0.8)',  // Yellow background
});
```

### Filtering Clusters

Add logic to skip certain clusters:

```python
# In name_clusters.py
if cluster_id < 0 or cluster_id > 10:  # Skip noise and large clusters
    continue
```

## Performance Considerations

- **Model Loading:** MusicGemma model is loaded once at startup
- **Batch Processing:** Consider processing clusters in batches for large datasets
- **Caching:** Generated names are cached in JSON file
- **Memory:** Ensure sufficient RAM for model and embeddings

## Future Enhancements

- **Batch Processing:** Process multiple clusters simultaneously
- **Name Validation:** Filter out inappropriate or low-quality names
- **User Feedback:** Allow users to edit or regenerate names
- **Multiple Languages:** Support for different language prompts
- **Name History:** Track changes and versions of cluster names
