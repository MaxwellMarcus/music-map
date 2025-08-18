# Python Web Server

A simple web server built with Python Flask, featuring a modern web interface and RESTful API endpoints.

## Features

- 🐍 **Python Flask Backend**: Lightweight and fast web framework
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 🔌 **RESTful API**: JSON-based API endpoints
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Real-time Interaction**: Dynamic frontend with JavaScript

## API Endpoints

- `GET /` - Main web interface
- `GET /api/hello` - Returns a hello message
- `GET /api/data` - Returns sample data
- `POST /api/data` - Accepts and returns JSON data

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5001`

### 3. Access the Web Interface

Open your browser and navigate to `http://localhost:5001`

## Project Structure

```
web-viewer/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Web interface template
```

## Usage

1. **Hello API**: Click the "Get Hello Message" button to test the basic API
2. **Get Data**: Click "Get Data" to retrieve sample data from the server
3. **Send Data**: Enter text in the input field and click "Send Data" to post data to the server

## Development

The server runs in debug mode by default, which means:
- Code changes will automatically reload the server
- Detailed error messages are displayed
- The server is accessible from other devices on your network

## Customization

You can easily extend this server by:
- Adding new routes in `app.py`
- Creating additional API endpoints
- Modifying the HTML template in `templates/index.html`
- Adding database integration
- Implementing user authentication

## Requirements

- Python 3.7+
- Flask 2.3.3
- Werkzeug 2.3.7
