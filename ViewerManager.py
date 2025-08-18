from DataManager import DataManager
import os
import importlib
import json
import numpy as np
import re
import importlib.util



class ViewerManager:
    def __init__(self, primary_data_manager, viewed_index="data"):
        self.primary_data_manager = primary_data_manager
        self.viewed_data_manager = primary_data_manager
        self.viewport_metadata = {}
        self.active_viewport_metadata = {"viewed_index": viewed_index, "color_source": None}
        self.index_methods = None
        self.augmentation_methods = None
        self.update_methods()
        
        # Set a default color source if indexes are available
        indexes = self.primary_data_manager.get_indexes()
        if hasattr(indexes, 'get_names') and indexes.get_names():
            self.active_viewport_metadata["color_source"] = f"index:{indexes.get_names()[0]}"
        elif indexes and len(indexes) > 0:
            first_index = list(indexes.keys())[0]
            self.active_viewport_metadata["color_source"] = f"index:{first_index}"

    def get_data(self):
        """Get the current data from the viewed data manager."""
        return self.viewed_data_manager.get_data()

    def get_metadata(self):
        return self.viewed_data_manager.get_metadata()

    def get_indexes(self):
        """Get all indexes from the viewed data manager."""
        indexes = self.viewed_data_manager.get_indexes()
        if hasattr(indexes, 'get_all_indexes'):
            return indexes.get_all_indexes()
        return indexes

    def get_color_data(self):
        """Get the current color data based on active color source."""
        color_source = self.active_viewport_metadata.get("color_source", "index:tsne")
        
        # Handle case where color_source is None
        if color_source is None:
            return None
        
        if color_source.startswith("index:"):
            index_name = color_source.replace("index:", "")
            
            # Check if we have a specific source data manager
            source_data_manager = self.active_viewport_metadata.get("source_data_manager")
            if source_data_manager:
                # Get the specified data manager
                all_managers = self.get_data_managers()
                if source_data_manager in all_managers:
                    target_dm = all_managers[source_data_manager]
                    indexes = target_dm.get_indexes()
                    if hasattr(indexes, 'get_index'):
                        index_obj = indexes.get_index(index_name)
                        if index_obj:
                            return index_obj.get_data()
                    else:
                        if index_name in indexes:
                            return indexes[index_name]
            
            # Fallback: try to get index from viewed data manager
            indexes = self.viewed_data_manager.get_indexes()
            if hasattr(indexes, 'get_index'):
                index_obj = indexes.get_index(index_name)
                if index_obj:
                    return index_obj.get_data()
            else:
                if index_name in indexes:
                    return indexes[index_name]
            
            # If not found in viewed data manager, try primary data manager
            if self.viewed_data_manager != self.primary_data_manager:
                primary_indexes = self.primary_data_manager.get_indexes()
                if hasattr(primary_indexes, 'get_index'):
                    index_obj = primary_indexes.get_index(index_name)
                    if index_obj:
                        return index_obj.get_data()
                else:
                    if index_name in primary_indexes:
                        return primary_indexes[index_name]
            
            return None
        elif color_source == "metadata":
            metadata_column = self.active_viewport_metadata.get("metadata_column", None)
            if metadata_column:
                metadata = self.get_metadata()
                if metadata is not None and metadata_column in metadata.columns:
                    return metadata[metadata_column].values
        else:  # default to first available index
            indexes = self.viewed_data_manager.get_indexes()
            if hasattr(indexes, 'get_names') and indexes.get_names():
                first_index = indexes.get_names()[0]
                index_obj = indexes.get_index(first_index)
                return index_obj.get_data() if index_obj else None
            return self.get_data()
        
        return None

    def set_color_source(self, color_source, source_name=None, source_data_manager=None):
        """Set the active color source for the plot."""
        self.active_viewport_metadata["color_source"] = color_source
        
        if color_source.startswith("index:"):
            # Default active_index to the name in the color_source if not provided
            index_name = color_source.replace("index:", "")
            self.active_viewport_metadata["active_index"] = source_name or index_name
            if source_data_manager:
                self.active_viewport_metadata["source_data_manager"] = source_data_manager
        elif color_source == "metadata" and source_name:
            self.active_viewport_metadata["metadata_column"] = source_name
            # Clear index-specific settings
            self.active_viewport_metadata.pop("active_index", None)
            self.active_viewport_metadata.pop("source_data_manager", None)
        else:
            # Clear any specific settings
            self.active_viewport_metadata.pop("active_index", None)
            self.active_viewport_metadata.pop("source_data_manager", None)
            self.active_viewport_metadata.pop("metadata_column", None)

    def get_available_color_sources(self):
        """Get all available color sources (indexes, metadata columns) from all data managers."""
        sources = {}
        
        # Get all data managers
        all_managers = self.get_data_managers()
        
        # Add indexes from all data managers
        for dm_name, dm in all_managers.items():
            indexes = dm.get_indexes()
            if hasattr(indexes, 'get_names'):
                for index_name in indexes.get_names():
                    index_obj = indexes.get_index(index_name)
                    display_name = index_obj.get_display_name() if index_obj else index_name
                    # Include data manager name in the key to distinguish between same-named indexes
                    sources[f"index:{index_name}"] = f"Index: {display_name} ({dm_name})"
            elif indexes:
                for index_name in indexes.keys():
                    sources[f"index:{index_name}"] = f"Index: {index_name} ({dm_name})"
        
        # Add available metadata columns
        metadata = self.get_metadata()
        if metadata is not None:
            for column in metadata.columns:
                sources[f"metadata:{column}"] = f"Metadata: {column}"
        
        return sources

    def get_data_managers(self):
        """Get all available data managers including augmentations."""
        managers = {"primary": self.primary_data_manager}
        
        # Recursively include all augmentations under the primary data manager
        managers.update(self._recurse_data_managers(self.primary_data_manager))
        
        return managers

    def _recurse_data_managers(self, data_manager):
        managers = {}
        if hasattr(data_manager, 'augmentations') and hasattr(data_manager.augmentations, 'get_all_data_managers'):
            for sub_name, sub_dm in data_manager.augmentations.get_all_data_managers().items():
                managers[sub_name] = sub_dm
                managers.update(self._recurse_data_managers(sub_dm))
        return managers

    def set_viewed_data_manager(self, data_manager):
        self.viewed_data_manager = data_manager
        # Ensure viewport metadata exists for this manager
        if data_manager not in self.viewport_metadata:
            # Default viewed index if available, else keep existing key
            default_index = None
            indexes = data_manager.get_indexes()
            if hasattr(indexes, 'get_names') and indexes.get_names():
                default_index = indexes.get_names()[0]
            self.viewport_metadata[data_manager] = {"viewed_index": default_index}
        self.active_viewport_metadata = self.viewport_metadata[data_manager]

    def set_viewed_index(self, index_name):
        self.active_viewport_metadata["viewed_index"] = index_name

    def update_methods(self):
        """Load method definitions from JSON files."""
        self.index_methods = self._load_methods("index-methods/methods.json")
        self.augmentation_methods = self._load_methods("augmentation-methods/methods.json")

    def _load_methods(self, file_path):
        """Load method definitions from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing {file_path}: {e}")
            return {}

    def get_index_methods(self):
        """Get available index methods."""
        if self.index_methods is None:
            self.update_methods()
        return self.index_methods

    def get_augmentation_methods(self):
        """Get available augmentation methods."""
        if self.augmentation_methods is None:
            self.update_methods()
        return self.augmentation_methods

    def _load_function_from(self, directory: str, file_name: str, func_name: str):
        module_path = os.path.join(directory, file_name)
        spec = importlib.util.spec_from_file_location(f"{directory.replace('-', '_')}_{file_name}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name)

    def augment(self, method, type=None, data_source="viewed", **kwargs):
        """Apply augmentation to data from specified source."""
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            user_kwargs = kwargs.pop('kwargs')
            kwargs.update(user_kwargs)
        
        method_obj = self.augmentation_methods[method]
        fn = self._load_function_from("augmentation-methods", method_obj['file'], method_obj["function"])
        
        # Get the target data manager based on data source
        if data_source == "viewed":
            target_dm = self.viewed_data_manager
        elif data_source == "primary":
            target_dm = self.primary_data_manager
        else:
            # Try to find data manager by name
            managers = self.get_data_managers()
            if data_source in managers:
                target_dm = managers[data_source]
            else:
                raise ValueError(f"Unknown data source: {data_source}")
        
        if target_dm.data is None:
            raise ValueError("No data available for augmentation")
        
        # Call augment_data on the target data manager
        target_dm.augment_data(fn, method if type is None else type, **kwargs)

    def index(self, method, type=None, data_source="viewed", **kwargs):
        """Create index from data from specified source."""
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            user_kwargs = kwargs.pop('kwargs')
            kwargs.update(user_kwargs)
        
        method_obj = self.index_methods[method]
        fn = self._load_function_from("index-methods", method_obj['file'], method_obj["function"])
        
        # Get the target data manager based on data source
        if data_source == "viewed":
            target_dm = self.viewed_data_manager
        elif data_source == "primary":
            target_dm = self.primary_data_manager
        else:
            # Try to find data manager by name
            managers = self.get_data_managers()
            if data_source in managers:
                target_dm = managers[data_source]
            else:
                raise ValueError(f"Unknown data source: {data_source}")
        
        if target_dm.data is None:
            raise ValueError("No data available for indexing")
        
        # Call index_data on the target data manager
        target_dm.index_data(fn, method if type is None else type, **kwargs)

    def validate_and_process_parameters(self, method_name, method_type, parameters, data_shape=None):
        """Validate and process parameters for a method."""
        if method_type == "augmentation":
            method_def = self.augmentation_methods.get(method_name)
        elif method_type == "index":
            method_def = self.index_methods.get(method_name)
        else:
            raise ValueError(f"Unknown method type: {method_type}")
        
        if not method_def:
            raise ValueError(f"Method '{method_name}' not found in {method_type} methods")
        
        method_params = method_def.get('parameters', {})
        processed_params = {}
        
        # Check for missing required parameters
        for param_name, param_def in method_params.items():
            required = param_def.get('required', False)
            if required and param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' is missing")
        
        for param_name, param_value in parameters.items():
            if param_name not in method_params:
                raise ValueError(f"Unknown parameter '{param_name}' for method '{method_name}'")
            
            param_def = method_params[param_name]
            param_type = param_def.get('type')
            

            
            if param_type == 'kwargs':
                processed_params[param_name] = self._process_kwargs_parameter(param_name, param_value)
            elif isinstance(param_type, str) and 'matrix' in param_type:

                processed_params[param_name] = self._process_matrix_parameter(param_name, param_value, param_type, data_shape)
            elif param_type == 'int':
                processed_params[param_name] = int(param_value)
            elif param_type == 'float':
                processed_params[param_name] = float(param_value)
            elif param_type == 'string':
                processed_params[param_name] = str(param_value)
            elif param_type == 'index':
                processed_params[param_name] = param_value
            elif isinstance(param_type, list):
                # Handle union types like [string, index]
                processed_params[param_name] = self._process_union_parameter(param_name, param_value, param_type, data_shape)
            else:
                processed_params[param_name] = param_value
        
        return processed_params

    def _process_kwargs_parameter(self, param_name, param_value):
        """Process kwargs parameters."""
        if isinstance(param_value, str):
            try:
                return json.loads(param_value)
            except json.JSONDecodeError:
                raise ValueError(f"Parameter '{param_name}' must be valid JSON")
        elif isinstance(param_value, dict):
            return param_value
        else:
            raise ValueError(f"Parameter '{param_name}' must be a JSON string or dictionary")

    def _process_matrix_parameter(self, param_name, param_value, param_type, data_shape):
        """Process matrix parameters.

        Accepts either:
        - A literal matrix string like "[1,2;3,4]"
        - A data manager name (e.g., 'primary', 'viewed', or an augmentation name), in which case
          the underlying data from that manager is used as the matrix.
        """

        # Helper to validate shape against (n, m, i, j) spec
        def _validate_against_spec(array_shape: tuple, spec: str, current_shape: tuple):
            if 'matrix' not in spec:
                return  # Nothing to validate
            dim_match = re.search(r'\(([^\)]*)\)\s*matrix', spec)
            if not dim_match:
                return
            tokens = [t.strip() for t in dim_match.group(1).split(',') if t.strip()]
            # Normalize shapes to 2D when applicable
            arr_rows = array_shape[0] if len(array_shape) > 0 else None
            arr_cols = array_shape[1] if len(array_shape) > 1 else None
            cur_n = current_shape[0] if current_shape and len(current_shape) > 0 else None
            cur_m = current_shape[1] if current_shape and len(current_shape) > 1 else None

            # Single-dimension case like (i,)matrix
            if len(tokens) == 1 and tokens[0].endswith(','):
                tok = tokens[0].replace(',', '')
                # tok can be n/m/i/j or a number
                if tok == 'n' and cur_n is not None and arr_rows != cur_n:
                    raise ValueError(f"Parameter '{param_name}' must have length n={cur_n}")
                if tok == 'm' and cur_m is not None and arr_rows != cur_m:
                    raise ValueError(f"Parameter '{param_name}' must have length m={cur_m}")
                if tok not in ('n', 'm', 'i', 'j'):
                    try:
                        expected = int(tok)
                        if arr_rows != expected:
                            raise ValueError(f"Parameter '{param_name}' must have length {expected}")
                    except ValueError:
                        pass
                return

            # Two-dimension case like (n, i)matrix
            if len(tokens) >= 1:
                # rows
                row_tok = tokens[0]
                if row_tok == 'n' and cur_n is not None and arr_rows != cur_n:
                    raise ValueError(f"Parameter '{param_name}' must have n={cur_n} rows")
                if row_tok == 'm' and cur_m is not None and arr_rows != cur_m:
                    raise ValueError(f"Parameter '{param_name}' must have m={cur_m} rows")
                if row_tok not in ('n', 'm', 'i', 'j'):
                    try:
                        expected = int(row_tok)
                        if arr_rows != expected:
                            raise ValueError(f"Parameter '{param_name}' must have {expected} rows")
                    except ValueError:
                        pass
            if len(tokens) >= 2:
                col_tok = tokens[1]
                if col_tok == 'n' and cur_n is not None and arr_cols != cur_n:
                    raise ValueError(f"Parameter '{param_name}' must have n={cur_n} columns")
                if col_tok == 'm' and cur_m is not None and arr_cols != cur_m:
                    raise ValueError(f"Parameter '{param_name}' must have m={cur_m} columns")
                if col_tok not in ('n', 'm', 'i', 'j'):
                    try:
                        expected = int(col_tok)
                        if arr_cols != expected:
                            raise ValueError(f"Parameter '{param_name}' must have {expected} columns")
                    except ValueError:
                        pass

        # 1) If a string is provided, it could be a literal matrix or a data manager name
        if isinstance(param_value, str):
            matrix_str = param_value.strip()
            # Literal matrix string
            if matrix_str.startswith('[') and matrix_str.endswith(']'):
                try:
                    # Remove outer brackets and split by rows
                    inner = matrix_str[1:-1]
                    rows = [row.strip() for row in inner.split(';') if row.strip()]
                    matrix = []
                    for row in rows:
                        row_values = [float(val.strip()) for val in row.split(',') if val.strip()]
                        matrix.append(row_values)
                    matrix = np.array(matrix)
                    if data_shape:
                        _validate_against_spec(matrix.shape, param_type, data_shape)
                    return matrix
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Parameter '{param_name}' could not be converted to matrix: {str(e)}")

            # Check if this is a point selection (e.g., "point:123:primary")
            if matrix_str.startswith('point:'):
                parts = matrix_str.split(':')
                if len(parts) >= 3:
                    try:
                        point_index = int(parts[1])
                        source_dm_name = parts[2]
                        
                        # Get the source data manager
                        if source_dm_name in ('viewed', 'current'):
                            source_dm = self.viewed_data_manager
                        elif source_dm_name == 'primary':
                            source_dm = self.primary_data_manager
                        else:
                            managers = self.get_data_managers()
                            if source_dm_name in managers:
                                source_dm = managers[source_dm_name]
                            else:
                                raise ValueError(f"Unknown data manager '{source_dm_name}' for point selection")
                        
                        # Get the point data
                        data = source_dm.get_data()
                        if data is None:
                            raise ValueError(f"Data manager '{source_dm_name}' has no data")
                        
                        if point_index >= len(data):
                            raise ValueError(f"Point index {point_index} is out of range for data manager '{source_dm_name}'")
                        
                        # Extract the point (row) from the data
                        point_data = data[point_index]
                        
                        # For vector parameters, ensure correct shape
                        if 'matrix' in param_type:
                            # Check if we need to transpose for vector parameters
                            if param_type.startswith('(1,') or param_type.startswith('(i,') or param_type.startswith('(j,'):
                                # Need row vector (1, n)
                                if len(point_data.shape) == 1:
                                    point_data = point_data.reshape(1, -1)
                            elif param_type.startswith('(,1)') or param_type.startswith('(,i)') or param_type.startswith('(,j)'):
                                # Need column vector (n, 1)
                                if len(point_data.shape) == 1:
                                    point_data = point_data.reshape(-1, 1)
                        
                        return point_data
                        
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Invalid point selection format '{matrix_str}': {str(e)}")
            
            # Otherwise treat as data manager identifier
            # Map name to data manager object
            if matrix_str in ('viewed', 'current'):
                target_dm = self.viewed_data_manager
            elif matrix_str == 'primary':
                target_dm = self.primary_data_manager
            else:
                managers = self.get_data_managers()
                if matrix_str in managers:
                    target_dm = managers[matrix_str]
                else:
                    raise ValueError(f"Unknown data manager '{matrix_str}' for parameter '{param_name}'")

            data = target_dm.get_data()
            if data is None:
                raise ValueError(f"Data manager '{matrix_str}' has no data for parameter '{param_name}'")
            if hasattr(data, 'shape') and data_shape:
                _validate_against_spec(data.shape, param_type, data_shape)
            return data

        # 2) If an array-like was provided
        if isinstance(param_value, (list, tuple, np.ndarray)):
            arr = np.array(param_value)
            if data_shape:
                _validate_against_spec(arr.shape, param_type, data_shape)
            return arr

        # 3) Otherwise fail
        raise ValueError(f"Parameter '{param_name}' could not be converted to matrix: Unsupported value type")

    def _process_union_parameter(self, param_name, param_value, param_types, data_shape):
        """Process union type parameters."""
        for param_type in param_types:
            try:
                if param_type == 'string':
                    return str(param_value)
                elif param_type == 'int':
                    return int(param_value)
                elif param_type == 'float':
                    return float(param_value)
                elif param_type == 'index':
                    return param_value
                elif isinstance(param_type, str) and 'matrix' in param_type:
                    return self._process_matrix_parameter(param_name, param_value, param_type, data_shape)
            except (ValueError, TypeError):
                continue
        
        raise ValueError(f"Parameter '{param_name}' could not be converted to any of the allowed types: {param_types}")

