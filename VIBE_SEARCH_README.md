# Vibe Search Feature

The "Search on Vibes" feature allows users to search for music by describing the mood, genre, or style they're looking for, using the MusicGemma model to find similar music through semantic similarity.

## 🎯 Overview

Instead of searching for specific artist names, vibe search uses natural language descriptions to find music that matches the described mood or style. The system:

1. **Embeds your query** using MusicGemma's text embedding capabilities
2. **Compares it** to all music embeddings using cosine similarity
3. **Colors the scatter plot** based on similarity scores
4. **Retains cluster names** throughout the search process

## 🚀 How to Use

### 1. Access Vibe Search
1. Open the web interface at `http://localhost:8000`
2. Click the search bar to open it
3. Use the **dropdown menu** to select **"Vibe Search"** mode
4. The placeholder text will change to indicate vibe search mode

### 2. Search by Vibes
Type natural language descriptions and **press Enter** to search:
- `"jazz"` - Find jazz music
- `"energetic rock"` - Find energetic rock music
- `"chill electronic"` - Find relaxing electronic music
- `"classical piano"` - Find classical piano pieces
- `"upbeat dance music"` - Find danceable tracks
- `"melancholic indie"` - Find sad indie music

**Note**: Vibe search only triggers when you press Enter, not while typing.

### 3. Visual Results
- **Plotly automatically colors** all points based on similarity scores
- **Color scale** automatically adjusts to show the full range of similarities
- **Cluster names** remain visible as annotations
- **Hover over points** to see details and similarity scores

### 4. Reset Colors
- **Clear the search** to return to original cluster colors
- **Close the search bar** to reset everything
- **Switch back to Artist Search** mode

## 🔧 Technical Implementation

### Backend (API)
- **Endpoint**: `/api/similarity-search?query=<text>`
- **Model**: Uses MusicGemma's CLAP module for text embedding
- **Algorithm**: Cosine similarity between query and music embeddings
- **Response**: Creates a similarity index and sets it as the active color source
- **Index Integration**: Automatically adds similarity data to the data manager's indexes

### Frontend (JavaScript)
- **Dropdown Selection**: Choose between Artist Search and Vibe Search modes
- **Enter Key Trigger**: Vibe search only activates when Enter is pressed
- **Automatic Plotly Coloring**: Raw similarity scores passed to Plotly for automatic color generation
- **Full Color Spectrum**: All points colored automatically based on similarity scores
- **Smart Reset**: Automatically returns to cluster colors when search is cleared

### Color Scheme
```javascript
// Raw similarity scores passed to Plotly
Plotly.restyle(plotDiv, {
    'marker.color': [similarityScores]
});
```

**Automatic Plotly Coloring**: The new system:
- Passes **raw similarity scores** directly to Plotly
- Lets Plotly **automatically generate** the optimal color scale
- Creates **dynamic color mapping** that adjusts to the data range
- Provides **professional color schemes** with automatic legend generation

## 🧪 Testing

### Test the API
```bash
python test_vibe_search.py
```

This will test various queries and show similarity scores.

### Manual Testing
1. Start the server: `python app.py`
2. Open browser: `http://localhost:8000`
3. Try different vibe queries
4. Check console for similarity statistics

## 📊 Example Queries

### Genres
- `"jazz"`
- `"rock"`
- `"electronic"`
- `"classical"`
- `"hip hop"`
- `"country"`
- `"blues"`

### Moods
- `"energetic"`
- `"chill"`
- `"melancholic"`
- `"upbeat"`
- `"relaxing"`
- `"intense"`
- `"peaceful"`

### Instruments
- `"piano"`
- `"guitar"`
- `"drums"`
- `"violin"`
- `"saxophone"`

### Styles
- `"dance music"`
- `"ambient"`
- `"orchestral"`
- `"acoustic"`
- `"electronic dance"`

## 🔍 How It Works

### 1. Text Embedding
```python
# Get text embedding from MusicGemma
text_embedding = model.model.music_model.get_text_embedding([query])
```

### 2. Similarity Calculation
```python
# Calculate cosine similarity
similarity = np.dot(text_embedding, music_embedding) / (norm1 * norm2)
```

### 3. Color Assignment
```javascript
// Map similarity to color
const colors = similarities.map(similarity => {
    const normalized = (similarity + 1) / 2;
    const r = Math.round(255 * normalized);
    const b = Math.round(255 * (1 - normalized));
    return `rgb(${r}, 0, ${b})`;
});
```

## 🎨 UI Features

### Search Mode Toggle
- **Blue button**: Artist Search mode
- **Green button**: Vibe Search mode
- **Click to switch** between modes

### Dynamic Placeholder
- Artist Search: `"Search for an artist..."`
- Vibe Search: `"Search by vibes, mood, genre..."`

### Color Feedback
- **Real-time coloring** as you type
- **Smooth transitions** between colors
- **Preserved cluster names** during search

## 🐛 Troubleshooting

### Common Issues

1. **No color changes**
   - Check browser console for errors
   - Verify MusicGemma model is loaded
   - Ensure embeddings file exists

2. **Slow response**
   - Model loading takes time on first query
   - Subsequent queries should be faster
   - Check server logs for errors

3. **No similarity results**
   - Verify query is not empty
   - Check API endpoint is working
   - Test with simple queries like "jazz"

### Debug Mode
Enable console logging to see similarity scores:
```javascript
// In browser console
console.log('Similarity scores:', currentSimilarities);
```

## 🔮 Future Enhancements

- **Query suggestions** based on popular searches
- **Multiple query support** (e.g., "jazz AND piano")
- **Advanced filtering** by similarity threshold
- **Search history** for repeated queries
- **Export results** of similar music
- **Batch processing** for multiple queries

## 📁 Files Modified

- `app.py` - Added similarity search API endpoint
- `templates/index.html` - Added vibe search UI and functionality
- `test_vibe_search.py` - Test script for the feature
- `requirements.txt` - Added requests dependency

## 🎵 Use Cases

- **Music Discovery**: Find new music by describing what you want
- **Playlist Creation**: Identify tracks with similar vibes
- **Genre Exploration**: Discover music within specific styles
- **Mood Matching**: Find music for specific moods or activities
- **Research**: Analyze music similarity patterns

The vibe search feature transforms the scatter plot from a static visualization into an interactive music discovery tool, allowing users to explore their music collection through natural language descriptions!
