# Music Visualization Web Application with AI Chatbot

A sophisticated web application built with Python Flask that provides an interactive visualization of musical artists and their relationships, featuring an AI-powered chatbot for music discussions and analysis.

## 🎵 Features

### **Interactive Music Visualization**
- **2D Interactive Plot**: Explore musical artists in a t-SNE or UMAP projected space
- **Cluster Analysis**: HDBSCAN-based clustering with customizable color schemes
- **Point Selection**: Click on any artist to view detailed information
- **Zoom & Pan**: Navigate through the visualization with intuitive controls
- **Dynamic Coloring**: Switch between different data sources for visualization

### **AI-Powered Music Chatbot**
- **Music Analysis**: Start conversations about specific musical artists and their work
- **Contextual Chat**: AI remembers conversation history for coherent discussions
- **Dual Model System**: 
  - Music-aware model for initial music analysis
  - Standard language model for ongoing conversation
- **Real-time Interaction**: Seamless chat interface integrated into the point info panel

### **Data Management & Processing**
- **Multi-dimensional Data**: Handle high-dimensional music embeddings
- **Augmentation Methods**: t-SNE, UMAP, and custom transformation pipelines
- **Index Management**: Flexible clustering and categorization systems
- **Metadata Integration**: Wikipedia data for artist information

### **Spotify Integration**
- **Music Playback**: Direct Spotify integration for listening to artist tracks
- **OAuth Authentication**: Secure Spotify account connection
- **Track Search**: Automatic music discovery for artists

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Spotify account (optional, for music playback)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd music-map
```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

## 📁 Project Structure

```
web-viewer/
├── app.py                          # Main Flask application with API endpoints
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── DataManager.py                 # Data management and processing
├── ViewerManager.py               # Visualization and interaction logic
├── templates/
│   └── index.html                # Interactive web interface
├── textmodel/
│   ├── MusicGemma.py             # AI model for music analysis
│   ├── query_with_music.py       # Chatbot query functions
│   └── music_audioset_epoch_15_esc_90.14.pt  # Pre-trained music model
├── augmentation-methods/          # Data transformation pipelines
│   ├── tsne.py                   # t-SNE dimensionality reduction
│   ├── umap.py                   # UMAP dimensionality reduction
│   └── methods.json              # Method configurations
├── index-methods/                 # Clustering and indexing
│   ├── hdbscan.py                # HDBSCAN clustering
│   └── methods.json              # Index configurations
├── cluster-methods/               # Cluster analysis tools
├── projection-methods/            # Additional projection methods
└── tests/                         # Test suite
```

## 🔧 Configuration

### Environment Variables
```bash
# Spotify API credentials (optional)
export SPOTIFY_CLIENT_ID="your_spotify_client_id"
export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
```

### Data Files
The application expects the following data files in the root directory:
- `music-clips-embeddings.npy` - High-dimensional music embeddings
- `music-clips-tsne.npy` - Pre-computed t-SNE projections
- `music-clips-hdbscan.npy` - Cluster assignments
- `wikipedia_musicians.csv` - Artist metadata

## 🎯 Usage Guide

### **Exploring the Visualization**
1. **Navigate**: Use mouse to zoom and pan around the visualization
2. **Select Points**: Click on any artist point to view detailed information
3. **Change Views**: Switch between different projection methods (t-SNE, UMAP)
4. **Customize Colors**: Use different data sources for color coding

### **Using the AI Chatbot**
1. **Start Conversation**: Click on an artist point, then click "Start Conversation About This Music"
2. **Chat Naturally**: Ask questions about the music, artist, or any related topics
3. **Context Awareness**: The AI remembers your conversation history
4. **Music Analysis**: Get AI-generated insights about musical characteristics

### **Spotify Integration**
1. **Connect Account**: Click "Connect Spotify Account" in the music section
2. **Listen to Music**: Play tracks directly from the web interface
3. **Discover Music**: Find related tracks and albums

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test files
pytest tests/test_integration.py
```

## 🔌 API Endpoints

### **Core Endpoints**
- `GET /` - Main web interface
- `GET /api/status` - Application status
- `GET /api/data.bin` - Visualization data
- `GET /api/metadata` - Artist metadata

### **Chatbot Endpoints**
- `POST /api/initiate-conversation` - Start AI conversation about music
- `POST /api/chat` - Continue AI conversation

### **Visualization Endpoints**
- `GET /api/color-data` - Color scheme data
- `GET /api/cluster-centroids` - Cluster information
- `GET /api/similarity-search` - Similarity search functionality

### **Spotify Endpoints**
- `GET /login` - Spotify OAuth login
- `GET /callback` - OAuth callback handler
- `GET /api/spotify-search` - Music search

## 🛠️ Development

### **Adding New Features**
- **New Augmentation Methods**: Add to `augmentation-methods/` directory
- **New Index Methods**: Add to `index-methods/` directory
- **API Endpoints**: Extend the Flask app in `app.py`
- **UI Components**: Modify `templates/index.html`

### **Model Customization**
- **Music Model**: Modify `textmodel/MusicGemma.py`
- **Chatbot Logic**: Update `textmodel/query_with_music.py`
- **Data Processing**: Extend `DataManager.py`

## 📊 Performance Considerations

- **Large Datasets**: Optimized for datasets with 100K+ data points
- **Memory Usage**: Efficient numpy-based data handling
- **Model Loading**: Lazy loading of AI models for faster startup
- **Caching**: Intelligent caching of processed data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- **LAION-CLAP**: Audio understanding model
- **Google Gemma**: Language model foundation
- **Hugging Face**: Transformers and PEFT libraries
- **Spotify**: Music streaming integration

## 📞 Support

For questions, issues, or contributions, please:
- Check existing issues in the repository
- Create a new issue with detailed information
- Contact the development team

---

**Built with ❤️ for music exploration and AI research**
