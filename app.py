from flask import Flask, render_template, request, jsonify, Response, redirect, session, make_response
import os
import numpy as np
import pandas as pd
import json
import requests
from urllib.parse import quote_plus, urlencode
from functools import lru_cache
from ViewerManager import ViewerManager
from DataManager import DataManager, Index
import argparse
import secrets
import sys
from textmodel.MusicGemma import get_model_and_such
from textmodel.query_with_music import query_with_music, query


class ManagerManager:
    # Do not change the fucking port.
    def __init__(self, run_app: bool=True, port: int=8000, small_data: bool=False):
        self.data_manager = DataManager()
        
        # Load data
        if small_data:
            # Create small test data
            n = 200
            data = np.random.rand(n, 10).astype(np.float32)
            self.data_manager = DataManager(data=data, name="primary")
        else:
            embeddings_file = "./music-clips-embeddings.npy"
            tsne_file = "./music-clips-tsne.npy"
            metadata_file = "./wikipedia_musicians.csv"

            self.data_manager.load_data(file_path=embeddings_file)
            # Load jittered t-SNE data (pre-processed to prevent overlap)
            jittered_tsne_file = "./music-clips-tsne-jittered.npy"
            if os.path.exists(jittered_tsne_file):
                print(f"Loading jittered t-SNE data from {jittered_tsne_file}")
                tsne_data = np.load(jittered_tsne_file)
                self.data_manager.augmentations.add_data_manager(DataManager(data=tsne_data, name="tsne"))
                print(f"Loaded jittered t-SNE data with shape: {tsne_data.shape}")
            elif os.path.exists(tsne_file):
                print(f"Jittered t-SNE file not found, loading original from {tsne_file}")
                tsne_data = np.load(tsne_file)
                self.data_manager.augmentations.add_data_manager(DataManager(data=tsne_data, name="tsne"))
                print(f"Loaded original t-SNE data with shape: {tsne_data.shape}")
            else:
                print(f"Neither jittered nor original t-SNE file found")
            if os.path.exists( metadata_file ):
                print(f"Loading metadata from {metadata_file}")
                metadata = pd.read_csv(metadata_file)
                print(f"Loaded metadata shape: {metadata.shape}")
                print(f"Metadata columns: {metadata.columns.tolist()}")
                self.data_manager.metadata = metadata
            else:
                print(f"Metadata file {metadata_file} not found")
            
            # Load HDBSCAN clusters as an index
            hdbscan_file = "./music-clips-hdbscan.npy"
            centroids_file = "./music-clips-centroids.npy"
            if os.path.exists(hdbscan_file):
                print(f"Loading HDBSCAN clusters from {hdbscan_file}")
                clusters = np.load(hdbscan_file)
                print(f"Loaded clusters shape: {clusters.shape}")
                
                # Load cluster centroids if available
                cluster_centroids = None
                if os.path.exists(centroids_file):
                    print(f"Loading cluster centroids from {centroids_file}")
                    cluster_centroids = np.load(centroids_file, allow_pickle=True).item()
                    print(f"Loaded centroids for {len(cluster_centroids)} clusters")
                
                # Create an index from the clusters
                from DataManager import Index
                cluster_index = Index(clusters, {"name": "hdbscan_clusters", "display_name": "HDBSCAN Clusters"})
                self.data_manager.indexes.add_index(cluster_index)
                
                # Store centroids in the data manager for color assignment
                if cluster_centroids:
                    self.data_manager.cluster_centroids = cluster_centroids
                
                print(f"Added HDBSCAN clusters as index")
            else:
                print(f"HDBSCAN clusters file {hdbscan_file} not found")
        
        self.viewer_manager = ViewerManager(self.data_manager, "data")
        # Only set t-SNE as viewed data manager if it exists
        if self.data_manager.augmentations.has_data_manager("tsne"):
            self.viewer_manager.set_viewed_data_manager(self.data_manager.augmentations.get_data_manager("tsne"))
            
            # Set HDBSCAN clusters as the default color source if available
            if "hdbscan_clusters" in self.data_manager.indexes.indexes:
                self.viewer_manager.set_color_source("index:hdbscan_clusters", "hdbscan_clusters", "primary")
                print("Set HDBSCAN clusters as default color source")
        
        # Spotify OAuth configuration
        self.spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID', 'your_spotify_client_id')
        self.spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', 'your_spotify_client_secret')
        self.spotify_redirect_uri = 'http://127.0.0.1:8000/callback'
        self.spotify_scopes = 'user-read-private user-read-email user-read-playback-state user-modify-playback-state streaming'
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.secret_key = secrets.token_hex(16)
        
        self.model, self.processor, self.tokenizer = get_model_and_such()
        
        # Add routes
        self.app.route('/')(self.index)
        self.app.route('/api/status')(self.status_api)
        self.app.route('/api/data.bin')(self.data_api)
        self.app.route('/api/metadata')(self.metadata_api)
        self.app.route('/api/color-data')(self.color_data_api)
        self.app.route('/api/cluster-centroids')(self.cluster_centroids_api)
        self.app.route('/api/cluster-names')(self.cluster_names_api)
        self.app.route('/api/similarity-search')(self.similarity_search_api)
        self.app.route('/api/set-color-source')(self.set_color_source_api)
        self.app.route('/api/cluster-background')(self.cluster_background_api)
        self.app.route('/api/data-bounds')(self.data_bounds_api)
        self.app.route('/api/spotify-search')(self.spotify_search_api)
        self.app.route('/api/initiate-conversation', methods=['POST'])(self.initiate_conversation_api)
        self.app.route('/api/chat', methods=['POST'])(self.chat_api)
        self.app.route('/login')(self.spotify_login)
        self.app.route('/callback')(self.spotify_callback)
        self.app.route('/logout')(self.spotify_logout)
        
        self._default_port = port
        
        if run_app:
            self.app.run(port=port, host='0.0.0.0')
    
    def index(self):
        return render_template('index.html')
    
    def spotify_login(self):
        """Initiate Spotify OAuth login"""
        auth_url = 'https://accounts.spotify.com/authorize?' + urlencode({
            'client_id': self.spotify_client_id,
            'response_type': 'code',
            'redirect_uri': self.spotify_redirect_uri,
            'scope': self.spotify_scopes,
            'state': secrets.token_hex(16)
        })
        return redirect(auth_url)
    
    def spotify_callback(self):
        """Handle Spotify OAuth callback"""
        code = request.args.get('code')
        if code:
            # Exchange code for access token
            token_url = 'https://accounts.spotify.com/api/token'
            data = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': self.spotify_redirect_uri,
                'client_id': self.spotify_client_id,
                'client_secret': self.spotify_client_secret
            }
            
            response = requests.post(token_url, data=data)
            if response.status_code == 200:
                token_data = response.json()
                
                # Store tokens in cookies
                resp = make_response(redirect('/'))
                resp.set_cookie('spotify_access_token', token_data['access_token'], max_age=3600)
                if 'refresh_token' in token_data:
                    resp.set_cookie('spotify_refresh_token', token_data['refresh_token'], max_age=31536000)
                
                return resp
        
        return redirect('/')
    
    def spotify_logout(self):
        """Logout from Spotify"""
        resp = make_response(redirect('/'))
        resp.delete_cookie('spotify_access_token')
        resp.delete_cookie('spotify_refresh_token')
        return resp
    
    def get_spotify_token(self):
        """Get valid Spotify access token"""
        access_token = request.cookies.get('spotify_access_token')
        if not access_token:
            return None
        
        # Check if token is still valid
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get('https://api.spotify.com/v1/me', headers=headers)
        
        if response.status_code == 401:
            # Token expired, try to refresh
            refresh_token = request.cookies.get('spotify_refresh_token')
            if refresh_token:
                token_url = 'https://accounts.spotify.com/api/token'
                data = {
                    'grant_type': 'refresh_token',
                    'refresh_token': refresh_token,
                    'client_id': self.spotify_client_id,
                    'client_secret': self.spotify_client_secret
                }
                
                response = requests.post(token_url, data=data)
                if response.status_code == 200:
                    token_data = response.json()
                    return token_data['access_token']
        
        return access_token if response.status_code == 200 else None
    
    def spotify_search_api(self):
        """Search for Spotify tracks by artist name"""
        try:
            artist_name = request.args.get('artist', '')
            if not artist_name:
                return jsonify({'error': 'No artist name provided'}), 400
            
            # Check if user is authenticated
            access_token = self.get_spotify_token()
            if not access_token:
                return jsonify({
                    'error': 'Not authenticated',
                    'needs_auth': True,
                    'login_url': '/login'
                }), 401
            
            # Search for tracks by artist
            headers = {'Authorization': f'Bearer {access_token}'}
            search_url = 'https://api.spotify.com/v1/search'
            params = {
                'q': f'artist:"{artist_name}"',
                'type': 'track',
                'limit': 5,
                'market': 'US'
            }
            
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                tracks = data.get('tracks', {}).get('items', [])
                
                if tracks:
                    # Get the first track
                    track = tracks[0]
                    track_info = {
                        'artist': artist_name,
                        'track_name': track['name'],
                        'track_id': track['id'],
                        'album_name': track['album']['name'],
                        'album_image': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'preview_url': track['preview_url'],
                        'external_url': track['external_urls']['spotify'],
                        'embed_url': f"https://open.spotify.com/embed/track/{track['id']}?utm_source=generator",
                        'found': True,
                        'message': f"Found track: {track['name']} by {artist_name}"
                    }
                    
                    # Add alternative tracks
                    if len(tracks) > 1:
                        track_info['alternative_tracks'] = tracks[1:5]
                    
                    return jsonify(track_info)
                else:
                    return jsonify({
                        'artist': artist_name,
                        'found': False,
                        'message': f"No tracks found for {artist_name}",
                        'search_url': f"https://open.spotify.com/search/{quote_plus(artist_name)}"
                    })
            else:
                return jsonify({'error': 'Spotify API error'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def run(self, port: int=None, debug: bool=False):
        """Helper to run the Flask app (used by tests)."""
        self.app.run(debug=debug, port=self._default_port if port is None else port, use_reloader=False)
               
    def home(self):
        return render_template('index.html')

    def status_api(self):
        """Get current status of data managers and available methods."""
        try:
            
            # Get current data manager name
            current_dm_name = 'primary'
            if self.viewer_manager.viewed_data_manager != self.viewer_manager.primary_data_manager:
                # Find the name of the current data manager
                for name, dm in self.viewer_manager.get_data_managers().items():
                    if dm == self.viewer_manager.viewed_data_manager:
                        current_dm_name = name
                        break
            
            # Get indexes info (one-dimensional arrays for coloring/clustering)
            indexes = self.viewer_manager.get_indexes()
            available_indexes = list(indexes.keys()) if indexes else []
            
            # Get augmentations info (multi-dimensional data for visualization)
            augmentations = self.viewer_manager.get_data_managers()
            available_augmentations = list(augmentations.keys()) if augmentations else []
            
            # Build parent-child relationship map
            augmentation_hierarchy = {}
            for name, dm in augmentations.items():
                if name == 'primary':
                    # Primary data manager's children are its direct augmentations
                    children = []
                    for child_name, child_dm in dm.augmentations.data_managers.items():
                        children.append(child_name)
                    augmentation_hierarchy[name] = children
                else:
                    # Other data managers' children are their direct augmentations
                    children = []
                    for child_name, child_dm in dm.augmentations.data_managers.items():
                        children.append(child_name)
                    augmentation_hierarchy[name] = children
            
            # Get current data information
            current_data = self.viewer_manager.get_data()
            data_info = {
                'shape': current_data.shape if current_data is not None else None,
                'dimensions': current_data.shape[1] if current_data is not None and len(current_data.shape) > 1 else None,
                'is_2d': current_data is not None and len(current_data.shape) == 2 and current_data.shape[1] == 2
            }
            
            # Get shape information for all data managers
            data_manager_shapes = {}
            for name, dm in augmentations.items():
                data = dm.get_data()
                if data is not None:
                    data_manager_shapes[name] = data.shape
                else:
                    data_manager_shapes[name] = None
            
            # Find 2D augmentations specifically (for visualization)
            two_d_augmentations = []
            for name, dm in augmentations.items():
                if name != 'primary':  # Skip the primary data manager
                    data = dm.get_data()
                    if data is not None and len(data.shape) == 2 and data.shape[1] == 2:
                        two_d_augmentations.append(name)
            
            return jsonify({
                'current_data_manager': current_dm_name,
                'current_index': self.viewer_manager.active_viewport_metadata["viewed_index"],
                'available_indexes': available_indexes,
                'available_augmentations': available_augmentations,
                'augmentation_hierarchy': augmentation_hierarchy,
                'two_d_augmentations': two_d_augmentations,
                'data_info': data_info,
                'data_manager_shapes': data_manager_shapes,
                'augmentation_methods': self.viewer_manager.get_augmentation_methods(),
                'index_methods': self.viewer_manager.get_index_methods(),
                'current_color_source': self.viewer_manager.active_viewport_metadata.get("color_source", "index:hdbscan_clusters")
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def data_api(self):
        """Serve current data as raw binary float32 interleaved [x0,y0,x1,y1,...]."""
        try:
        
            # Check if the current data exists
            current_data = self.viewer_manager.get_data()

            if current_data is None:
                return jsonify({'error': 'No data available'}), 404
            
            # Validate data dimensions
            if not current_data.ndim == 2:
                return jsonify({
                    "error": f"Data must be 2-dimensional for visualization. Current data has {current_data.ndim} dimensions.",
                    "data_shape": current_data.shape,
                    "suggestion": "Use an augmentation method like t-SNE or UMAP to create 2D data."
                }), 404
            
            if not current_data.shape[1] == 2:
                return jsonify({
                    "error": f"Data must have exactly 2 columns for 2D visualization. Current data has {current_data.shape[1]} columns.",
                    "data_shape": current_data.shape,
                    "suggestion": f"Use an augmentation method to reduce {current_data.shape[1]}-dimensional data to 2D."
                }), 404

            coords = np.ascontiguousarray(current_data.astype(np.float32, copy=False))
            data_bytes = coords.tobytes(order='C')
            resp = Response(data_bytes, mimetype='application/octet-stream')
            resp.headers['Content-Disposition'] = 'inline; filename="data.bin"'
            resp.headers['X-Dtype'] = 'float32'
            resp.headers['X-Columns'] = '2'
            resp.headers['X-Count'] = str(coords.shape[0])
            return resp
            
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def indexes_api(self):
        """Get index data for a specific data manager or current data manager."""
        try:
            # Get data manager name from query parameter
            data_manager_name = request.args.get('data_manager', 'viewed')
            
            # Get the target data manager
            if data_manager_name == 'viewed':
                target_dm = self.viewer_manager.viewed_data_manager
            elif data_manager_name == 'primary':
                target_dm = self.viewer_manager.primary_data_manager
            else:
                # Try to find data manager by name
                managers = self.viewer_manager.get_data_managers()
                if data_manager_name in managers:
                    target_dm = managers[data_manager_name]
                else:
                    return jsonify({'error': f'Data manager {data_manager_name} not found'}), 404
            
            # Get indexes for the target data manager
            indexes = target_dm.indexes
            index_names = target_dm.get_index_names()
            
            if not indexes:
                return jsonify({'indexes': {}, 'index_names': {}})
            
            # Convert to serializable format
            serializable_indexes = {}
            if hasattr(indexes, 'get_all_indexes'):
                all_indexes = indexes.get_all_indexes()
                for name, index_obj in all_indexes.items():
                    if hasattr(index_obj, 'get_data'):
                        data = index_obj.get_data()
                        if hasattr(data, 'tolist'):
                            serializable_indexes[name] = data.tolist()
                        else:
                            serializable_indexes[name] = list(data) if data else []
                    else:
                        if hasattr(index_obj, 'tolist'):
                            serializable_indexes[name] = index_obj.tolist()
                        else:
                            serializable_indexes[name] = list(index_obj) if index_obj else []
            
            return jsonify({
                'indexes': serializable_indexes,
                'index_names': index_names
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def augment_api(self):
        """Create data augmentation using specified method."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            # Check if managers are properly initialized
            if self.viewer_manager is None:
                return jsonify({'error': 'Viewer manager not initialized'}), 500
            
            method = data.get('method')
            augmentation_type = data.get('augmentation_type', method)
            parameters = data.get('parameters', {})
            data_source = data.get('data_source', 'viewed')  # New parameter
            
            if not method:
                return jsonify({'error': 'Method is required'}), 400
            
            # Get current data shape for matrix validation
            current_data = self.viewer_manager.viewed_data_manager.get_data() # Should not be referencing viewed_data_manager for anything here because there is no guarantee the viewed_data_manager is the one being operated on
            data_shape = current_data.shape if current_data is not None else None
            
            # Validate and process parameters
            processed_params = self.viewer_manager.validate_and_process_parameters(
                method, 'augmentation', parameters, data_shape
            )
            
            self.viewer_manager.augment(method, augmentation_type, data_source=data_source, **processed_params)
            
            return jsonify({
                'status': 'success',
                'message': f'Augmentation {augmentation_type} created using {method}',
                'available_augmentations': list(self.viewer_manager.get_data_managers().keys())
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def switch_index_api(self):
        """Switch to a different augmentation for visualization."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            index_name = data.get('index')
            if not index_name:
                return jsonify({'error': 'Index name is required'}), 400
            
            # Check if the augmentation exists (t-SNE/UMAP are augmentations, not indexes)
            augmentations = self.viewer_manager.get_data_managers()
            if index_name not in augmentations:
                available_augmentations = list(augmentations.keys()) if augmentations else []
                return jsonify({'error': f'Augmentation {index_name} not found. Available: {available_augmentations}'}), 400
            
            # Switch to the augmentation DataManager
            target_dm = augmentations[index_name]
            self.viewer_manager.set_viewed_data_manager(target_dm)
            
            return jsonify({
                'status': 'success',
                'message': f'Switched to augmentation {index_name}',
                'current_index': index_name
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def switch_data_manager_api(self):
        """Switch to a different data manager."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            data_manager_name = data.get('data_manager')
            if not data_manager_name:
                return jsonify({'error': 'Data manager name is required'}), 400
            
            # Check if the data manager exists
            data_managers = self.viewer_manager.get_data_managers()
            if data_manager_name not in data_managers:
                available_managers = list(data_managers.keys())
                return jsonify({'error': f'Data manager {data_manager_name} not found. Available: {available_managers}'}), 400
            
            # Switch to the data manager
            target_dm = data_managers[data_manager_name]
            self.viewer_manager.set_viewed_data_manager(target_dm)
            
            return jsonify({
                'status': 'success',
                'message': f'Switched to data manager {data_manager_name}',
                'current_data_manager': data_manager_name
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def metadata_api(self):
        """Get metadata for current data."""
        try:
            # Always get metadata from the primary data manager since that's where CSV data is loaded
            metadata = self.viewer_manager.primary_data_manager.get_metadata()
            print(f"Metadata API: metadata = {metadata}")
            if metadata is None:
                print("Metadata API: metadata is None")
                return jsonify({'metadata': None})
            
            records = metadata.to_dict('records')
            print(f"Metadata API: records = {records[:2] if records else None}")  # Show first 2 records
            return jsonify({
                'metadata': records
            })
        except Exception as exc:
            print(f"Metadata API error: {exc}")
            return jsonify({'error': str(exc)}), 500


    def method_parameters_api(self):
        """Get parameters for a specific method."""
        try:
            method_type = request.args.get('type')
            method_name = request.args.get('method')
            
            if not method_type or not method_name:
                return jsonify({'error': 'Both type and method parameters are required'}), 400
            
            if method_type == 'augmentation':
                methods = self.viewer_manager.get_augmentation_methods()
            elif method_type == 'index':
                methods = self.viewer_manager.get_index_methods()
            else:
                return jsonify({'error': 'Invalid method type. Must be "augmentation" or "index"'}), 400
            
            if method_name not in methods:
                return jsonify({'error': f'Method {method_name} not found in {method_type} methods'}), 404
            
            method_def = methods[method_name]
            return jsonify({
                'method': method_name,
                'type': method_type,
                'parameters': method_def.get('parameters', {})
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def index_api(self):
        """Create index using specified method."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            # Check if managers are properly initialized
            if self.viewer_manager is None:
                return jsonify({'error': 'Viewer manager not initialized'}), 500
            
            method = data.get('method')
            index_type = data.get('index_type', method)
            parameters = data.get('parameters', {})
            data_source = data.get('data_source', 'viewed')
            
            if not method:
                return jsonify({'error': 'Method is required'}), 400
            
            # Get current data shape for matrix validation
            current_data = self.viewer_manager.viewed_data_manager.get_data()
            data_shape = current_data.shape if current_data is not None else None
            
            # Validate and process parameters
            processed_params = self.viewer_manager.validate_and_process_parameters(
                method, 'index', parameters, data_shape
            )
            
            self.viewer_manager.index(method, index_type, data_source=data_source, **processed_params)
            
            return jsonify({
                'status': 'success',
                'message': f'Index {index_type} created using {method}',
                'available_indexes': list(self.viewer_manager.get_indexes().keys())
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def color_sources_api(self):
        """Get available color sources."""
        try:
            color_sources = self.viewer_manager.get_available_color_sources()
            current_color_source = self.viewer_manager.active_viewport_metadata.get("color_source", "index:hdbscan_clusters")
            
            print(f"Available color sources: {color_sources}")
            print(f"Current color source: {current_color_source}")
            
            return jsonify({
                'color_sources': color_sources,
                'current_color_source': current_color_source
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def set_color_source_api(self):
        """Set the active color source."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            color_source = data.get('color_source')
            source_name = data.get('source_name')
            source_data_manager = data.get('source_data_manager')
            
            if not color_source:
                return jsonify({'error': 'Color source is required'}), 400
            
            # Set the color source
            self.viewer_manager.set_color_source(color_source, source_name, source_data_manager)
            
            return jsonify({
                'status': 'success',
                'message': f'Color source set to {color_source}',
                'current_color_source': color_source
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def color_data_api(self):
        """Get color data for current color source."""
        try:
            color_data = self.viewer_manager.get_color_data()
            
            # Convert to list if it's a numpy array, or return empty list if None
            if color_data is None:
                color_list = []
            elif hasattr(color_data, 'tolist'):
                color_list = color_data.tolist()
            else:
                color_list = list(color_data) if color_data else []
            
            return jsonify({
                'color_data': color_list,
                'color_source': self.viewer_manager.active_viewport_metadata.get("color_source", "index:tsne")
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def cluster_centroids_api(self):
        """Get cluster centroids for spatial color assignment."""
        try:
            centroids = getattr(self.data_manager, 'cluster_centroids', None)
            if centroids is None:
                return jsonify({'centroids': None})
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_centroids = {}
            for cluster_id, centroid in centroids.items():
                serializable_centroids[int(cluster_id)] = centroid.tolist()
            
            return jsonify({
                'centroids': serializable_centroids
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def cluster_names_api(self):
        """Get cluster names for display on the scatter plot."""
        try:
            cluster_names_file = "./cluster_names.json"
            if not os.path.exists(cluster_names_file):
                return jsonify({'cluster_names': None})
            
            with open(cluster_names_file, 'r') as f:
                cluster_names = json.load(f)
            
            return jsonify({
                'cluster_names': cluster_names
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def similarity_search_api(self):
        """Perform similarity search using MusicGemma text embeddings and create an index."""
        try:
            # Get query from request
            query = request.args.get('query', '').strip()
            if not query:
                return jsonify({'error': 'Query parameter is required'}), 400
            
            # Import MusicGemma model
            sys.path.append('./text-model')
            
            # Load model and get text embedding
            if self.model is None:
                self.model, self.processor, self.tokenizer = get_model_and_such()
            
            # Get text embedding from the music model
            if hasattr(self.model.model.model, 'music_model') and self.model.model.model.music_model:
                # Get text embedding - try different method names
                music_model = self.model.model.model.music_model
                text_embedding = None
                
                if hasattr(music_model, 'get_text_embedding'):
                    text_embedding = music_model.get_text_embedding([query])
                elif hasattr(music_model, 'get_text_embedding_from_data'):
                    text_embedding = music_model.get_text_embedding_from_data([query])
                elif hasattr(music_model, 'encode_text'):
                    text_embedding = music_model.encode_text([query])
                else:
                    return jsonify({'error': 'Text embedding method not found in music model'}), 500
                
                if text_embedding is None:
                    return jsonify({'error': 'Failed to get text embedding'}), 500
                
                # Load original embeddings
                embeddings_file = "./music-clips-embeddings.npy"
                if not os.path.exists(embeddings_file):
                    return jsonify({'error': 'Embeddings file not found'}), 404
                
                embeddings = np.load(embeddings_file)
                
                # Calculate cosine similarities
                similarities = embeddings @ text_embedding.T
                
                index_name = f"similarity_{query.replace(' ', '_').lower()}"
                display_name = f"Similarity: {query}"
                
                # Create the index
                similarity_index = Index(similarities.flatten(), {
                    "name": index_name,
                    "display_name": display_name,
                    "query": query,
                    "type": "similarity_search"
                })
                
                # Add to data manager indexes
                self.data_manager.indexes.add_index(similarity_index)
                
                # Set as active color source
                self.viewer_manager.set_color_source(f"index:{index_name}", index_name, "primary")
                
                # Convert numpy array to list for JSON serialization
                similarities_list = similarities.flatten().tolist()
                
                return jsonify({
                    'similarities': similarities_list,
                    'query': query,
                    'index_name': index_name,
                    'display_name': display_name,
                    'status': 'success'
                })
            else:
                return jsonify({'error': 'Music model not available'}), 500
                
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def set_color_source_api(self):
        """Set the active color source."""
        try:
            # Handle invalid JSON
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            if data is None:
                return jsonify({'error': 'Invalid JSON data'}), 400
            
            color_source = data.get('color_source')
            source_name = data.get('source_name')
            source_data_manager = data.get('source_data_manager')
            
            if not color_source:
                return jsonify({'error': 'Color source is required'}), 400
            
            # Set the color source
            self.viewer_manager.set_color_source(color_source, source_name, source_data_manager)
            
            return jsonify({
                'status': 'success',
                'message': f'Color source set to {color_source}',
                'current_color_source': color_source
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500
    
    def initiate_conversation( self, music_data: np.ndarray ):
        """Initiate a conversation with the music model."""
        try:
            if self.model is None:
                self.model, self.processor, self.tokenizer = get_model_and_such()
            output = query_with_music(self.model, self.processor, "Describe this music.", music_data, max_new_tokens=16) # Do not change this you dum dum.
            model_response = output.split("model")[-1].strip()
            out_text = f"This music has {model_response}."
            return jsonify({
                'status': 'success',
                'message': f'Conversation initiated with {model_response}',
                'model_response': model_response,
                "conversation_history": [ { "role": "system", "content": [ { "type": "text", "text": out_text } ] } ]
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500
        
    def chat( self, conversation_history: list[dict]=[], max_new_tokens: int=128 ):
        """Chat with the music model."""
        try:
            if self.model is None:
                self.model, self.processor, self.tokenizer = get_model_and_such()
            
            output = query(self.model, self.processor, conversation_history=conversation_history, max_new_tokens=max_new_tokens) # Do not change this you dum dum.
            model_response = output.split("model")[-1].strip()
            return jsonify({
                'status': 'success',
                'message': f'Conversation continued with {model_response}',
                'model_response': model_response,
                "conversation_history": conversation_history + [ { "role": "system", "content": [ { "type": "text", "text": model_response } ] } ]
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def cluster_background_api(self):
        """Serve the cluster background SVG."""
        try:
            svg_path = "./cluster_background.svg"
            if os.path.exists(svg_path):
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
                return Response(svg_content, mimetype='image/svg+xml')
            else:
                return jsonify({'error': 'SVG background not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def data_bounds_api(self):
        """Get data bounds for zoom constraints."""
        try:
            bounds_path = "./data_bounds.json"
            if os.path.exists(bounds_path):
                with open(bounds_path, 'r') as f:
                    bounds = json.load(f)
                return jsonify(bounds)
            else:
                return jsonify({'error': 'Bounds file not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def dataset_info_api(self):
        """Get detailed information about the current dataset."""
        try:
            data_manager = self.viewer_manager.viewed_data_manager
            current_data = data_manager.get_data()
            
            # Get data information
            data_info = {
                'shape': current_data.shape if current_data is not None else None,
                'dimensions': current_data.shape[1] if current_data is not None and len(current_data.shape) > 1 else None,
                'is_2d': current_data is not None and len(current_data.shape) == 2 and current_data.shape[1] == 2,
                'total_points': current_data.shape[0] if current_data is not None else 0
            }
            
            # Get available indexes (one-dimensional arrays for coloring/clustering)
            available_indexes = data_manager.get_index_names()
            
            # Get available augmentations (multi-dimensional data for visualization)
            augmentations = self.viewer_manager.get_data_managers()
            available_augmentations = list(augmentations.keys()) if augmentations else []
            
            # Get 2D augmentations specifically (for visualization)
            two_d_augmentations = []
            for name, dm in augmentations.items():
                if name != 'primary':  # Skip the primary data manager
                    data = dm.get_data()
                    if data is not None and len(data.shape) == 2 and data.shape[1] == 2:
                        two_d_augmentations.append(name)
            
            # Generate suggestions based on data characteristics
            suggestions = []
            if current_data is not None:
                if data_info['is_2d']:
                    suggestions.append("Your data is already 2D and ready for visualization.")
                elif data_info['dimensions'] == 1:
                    suggestions.append("You can create a histogram or line plot of your 1D data.")
                elif data_info['dimensions'] == 3:
                    suggestions.append("You can create a 3D scatter plot or reduce to 2D using t-SNE/UMAP.")
                elif data_info['dimensions'] > 3:
                    suggestions.append(f"You can reduce your {data_info['dimensions']}-dimensional data to 2D using dimensionality reduction methods.")
            
            if two_d_augmentations:
                suggestions.append(f"You have {len(two_d_augmentations)} 2D augmentations available: {', '.join(two_d_augmentations)}")
            else:
                suggestions.append("No 2D augmentations available. Create a t-SNE or UMAP augmentation to generate 2D visualization data.")
            
            if available_indexes:
                suggestions.append(f"You have {len(available_indexes)} indexes available for coloring/clustering: {', '.join(available_indexes.keys())}")
            
            return jsonify({
                'data_info': data_info,
                'available_indexes': available_indexes,
                'available_augmentations': available_augmentations,
                'two_d_augmentations': two_d_augmentations,
                'suggestions': suggestions,
                'current_data_manager': 'primary'
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def initiate_conversation_api(self):
        """API endpoint to initiate a conversation with music data."""
        try:
            data = request.get_json()
            if not data or 'point_index' not in data:
                return jsonify({'error': 'Missing point_index parameter'}), 400
            
            point_index = data['point_index']
            
            # Get the music data for this point from the primary data manager
            primary_data = self.data_manager.get_data()
            if primary_data is not None and point_index < len(primary_data):
                music_data = primary_data[point_index]
                return self.initiate_conversation(music_data)
            else:
                return jsonify({'error': 'Invalid point index or no data available'}), 400
                
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def chat_api(self):
        """API endpoint to continue a chat conversation."""
        try:
            data = request.get_json()
            if not data or 'conversation_history' not in data:
                return jsonify({'error': 'Missing conversation_history parameter'}), 400
            
            conversation_history = data['conversation_history']
            print( conversation_history )
            max_new_tokens = 128 #data.get('max_new_tokens', 128)
            
            return self.chat(conversation_history, max_new_tokens)
                
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500


if __name__ == "__main__":
    mm = ManagerManager()
    
   
