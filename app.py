from flask import Flask, render_template, request, jsonify, Response
import os
import numpy as np
from functools import lru_cache
from ViewerManager import ViewerManager
from DataManager import DataManager, Index
import argparse


class ManagerManager:
    # Do not change the fucking port.
    def __init__(self, run_app: bool=True, port: int=5000, small_data: bool=False):
        self.data_manager = DataManager()

        if small_data:
            # Create small test data
            n = 200
            data = np.random.rand(n, 10).astype(np.float32)
            self.data_manager = DataManager(data=data, name="primary")
        else:
            embeddings_file = "./music-clips-embeddings.npy"
            tsne_file = "./music-clips-tsne.npy"
            
            self.data_manager.load_data(file_path=embeddings_file)
            # Only load t-SNE if the file exists
            if os.path.exists(tsne_file):
                tsne_data = np.load(tsne_file)
                self.data_manager.augmentations.add_data_manager(DataManager(data=tsne_data, name="tsne"))
        
        self.viewer_manager = ViewerManager(self.data_manager, "data")
        # Only set t-SNE as viewed data manager if it exists
        if self.data_manager.augmentations.has_data_manager("tsne"):
            self.viewer_manager.set_viewed_data_manager(self.data_manager.augmentations.get_data_manager("tsne"))

        self.app = Flask(__file__)

        self.app.route("/")(self.home)
        self.app.route("/api/status")(self.status_api)
        self.app.route("/api/data.bin")(self.data_api_binary)
        self.app.route("/api/indexes")(self.indexes_api)
        self.app.route("/api/augment", methods=["POST"])(self.augment_api)
        self.app.route('/api/switch-index', methods=['POST'])(self.switch_index_api)
        self.app.route('/api/switch-data-manager', methods=['POST'])(self.switch_data_manager_api)
        self.app.route('/api/metadata')(self.metadata_api)
        self.app.route('/api/method-parameters')(self.method_parameters_api)
        self.app.route('/api/index', methods=['POST'])(self.index_api)
        self.app.route('/api/color-sources')(self.color_sources_api)
        self.app.route('/api/set-color-source', methods=['POST'])(self.set_color_source_api)
        self.app.route('/api/color-data')(self.color_data_api)
        self.app.route('/api/dataset-info')(self.dataset_info_api)
        
        self._default_port = port
        if run_app:
            # Normal interactive run
            self.app.run( port=port )

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
                'current_color_source': self.viewer_manager.active_viewport_metadata.get("color_source", "index:tsne")
            })
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    def data_api_binary(self):
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
            metadata = self.viewer_manager.get_metadata()
            if metadata is None:
                return jsonify({'metadata': None})
            
            return jsonify({
                'metadata': metadata.to_dict('records')
            })
        except Exception as exc:
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
            current_color_source = self.viewer_manager.active_viewport_metadata.get("color_source", "index:tsne")
            
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


if __name__ == "__main__":
    mm = ManagerManager()
    
   
