import numpy as np
import os
import pandas as pd
import json

class DataManager:
    def __init__(self, data=None, metadata=None, name="unnamed", augmentations=None, indexes=None):
        self.data = data
        self.metadata = metadata
        self.name = name

        # Initialize collections
        self.augmentations = DataManagerCollection(augmentations or {}) # Augmentations are sub data managers that are built off of the data
        self.indexes = IndexCollection(indexes or {}) # Indexes are one dimensional arrays that have the same number of elements as the data

    def load_data(self, file_path=None, data=None):
        self.data = np.load(file_path) if not file_path is None else data

    def augment_data(self, augmentation_function, augmentation_type, **kwargs):
        """Apply augmentation function to data and store result."""
        input_data = self.data.copy()

        # Apply numpy function to the input data or get attributes from the input data
        if "numpy_pre_ops" in kwargs:
            numpy_pre_ops = kwargs.pop( "numpy_pre_ops", [] )
            for op in numpy_pre_ops:
                op = op.split( "." )
                if op[ 0 ] == "np":
                    op_func = np
                    for o in op[ 1: ]:
                        op_func = getattr( op_func, o )
                    input_data = op_func( input_data )
                else:
                    for o in op:
                        input_data = getattr( input_data, o )

        # Apply the augmentation function to the input data
        result = augmentation_function(input_data, **kwargs)

        # Apply numpy function to the result or get attributes from the result
        if "numpy_post_ops" in kwargs:
            numpy_post_ops = kwargs.pop( "numpy_post_ops", [] )
            for op in numpy_post_ops:
                op = op.split( "." )
                if op[ 0 ] == "np":
                    op_func = np
                    for o in op[ 1: ]:
                        op_func = getattr( op_func, o )
                    result = op_func( result )
                else:
                    for o in op:
                        result = getattr( result, o )
        
        # Handle different return formats
        if isinstance(result, dict):
            if "data" in result:
                # Store the augmented data directly
                new_dm = DataManager(data=result["data"], name=augmentation_type)
                self.augmentations.add_data_manager(new_dm)
            else:
                # Assume result contains data directly
                new_dm = DataManager(data=result, name=augmentation_type)
                self.augmentations.add_data_manager(new_dm)
        else:
            # Direct array result
            new_dm = DataManager(data=result, name=augmentation_type)
            self.augmentations.add_data_manager(new_dm)
        
        return new_dm

    def index_data(self, index_function, index_type, **kwargs):
        """Apply index function to data and store result."""
        input_data = self.data.copy()

        # Apply numpy function to the input data or get attributes from the input data
        if "numpy_pre_ops" in kwargs:
            numpy_pre_ops = kwargs.pop( "numpy_pre_ops", [] )
            for op in numpy_pre_ops:
                op = op.split( "." )
                if op[ 0 ] == "np":
                    op_func = np
                    for o in op[ 1: ]:
                        op_func = getattr( op_func, o )
                    input_data = op_func( input_data )
                else:
                    for o in op:
                        input_data = getattr( input_data, o )
        result = index_function(input_data, **kwargs)

        # Apply numpy function to the result or get attributes from the result
        if "numpy_post_ops" in kwargs:
            numpy_post_ops = kwargs.pop( "numpy_post_ops", [] )
            for op in numpy_post_ops:
                op = op.split( "." )
                if op[ 0 ] == "np":
                    op_func = np
                    for o in op[ 1: ]:
                        op_func = getattr( op_func, o )
                    result = op_func( result )
                else:
                    for o in op:
                        result = getattr( result, o )
        
        # Handle different return formats
        if isinstance(result, dict):
            if "index" in result:
                # Store the index values
                index_obj = Index(result["index"], {"name": index_type})
                if "name" in result:
                    index_obj.names["display_name"] = result["name"]
                self.indexes.add_index(index_obj)
            else:
                # Assume result contains index directly
                self.indexes.add_index(Index(result, {"name": index_type}))
        else:
            # Direct array result
            self.indexes.add_index(Index(result, {"name": index_type}))
        
        return self.indexes.get_index(index_type)

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data
        if self.data is not None:
            np.save(os.path.join(file_path, "data.npy"), self.data)

        # Save metadata
        if self.metadata is not None:
            self.metadata.to_csv(os.path.join(file_path, "metadata.csv"), index=False)

        # Save augmentations
        augmentation_dir = os.path.join(file_path, "augmentations")
        os.makedirs(augmentation_dir, exist_ok=True)
        for name, dm in self.augmentations.data_managers.items():
            dm.save(os.path.join(augmentation_dir, name))

        # Save indexes
        index_dir = os.path.join(file_path, "indexes")
        os.makedirs(index_dir, exist_ok=True)
        for name, index_obj in self.indexes.indexes.items():
            np.save(os.path.join(index_dir, f"{name}.npy"), index_obj.index)
            with open(os.path.join(index_dir, f"{name}.json"), "w") as f:
                json.dump(index_obj.names, f)

    @staticmethod
    def load(file_path):
        # Load data
        data = np.load(os.path.join(file_path, "data.npy"))
        
        # Load metadata if it exists
        metadata = None
        metadata_path = os.path.join(file_path, "metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
        
        # Load augmentations
        augmentations = {}
        augmentation_dir = os.path.join(file_path, "augmentations")
        if os.path.exists(augmentation_dir):
            for augmentation_type in os.listdir(augmentation_dir):
                augmentations[augmentation_type] = DataManager.load(os.path.join(augmentation_dir, augmentation_type))
        
        # Load indexes
        indexes = {}
        index_dir = os.path.join(file_path, "indexes")
        if os.path.exists(index_dir):
            for index_file in os.listdir(index_dir):
                if index_file.endswith(".npy"):
                    index_type = index_file[:-4]
                    index_data = np.load(os.path.join(index_dir, index_file))
                    
                    # Load index names
                    names = {}
                    names_path = os.path.join(index_dir, f"{index_type}.json")
                    if os.path.exists(names_path):
                        with open(names_path, "r") as f:
                            names = json.load(f)
                    
                    indexes[index_type] = Index(index_data, names)
        
        dm = DataManager(data, metadata, augmentations=augmentations, indexes=indexes)
        return dm

    def get_metadata(self):
        return self.metadata

    def get_data(self):
        return self.data

    def get_augmentations(self):
        return self.augmentations
    
    def get_indexes(self):
        return self.indexes

    def get_index_names(self):
        """Get a mapping of index names to their display names."""
        names = {}
        for name, index_obj in self.indexes.indexes.items():
            display_name = index_obj.names.get("display_name", name)
            names[name] = display_name
        return names


class DataManagerCollection:
    def __init__(self, data_managers=None):
        self.data_managers = data_managers or {}

    def add_data_manager(self, data_manager):
        """Add a data manager to the collection."""
        self.data_managers[data_manager.name] = data_manager

    def get_data_manager(self, name):
        """Get a data manager by name."""
        return self.data_managers.get(name)

    def get_all_data_managers(self):
        """Get all data managers as a dictionary."""
        return self.data_managers

    def remove_data_manager(self, name):
        """Remove a data manager by name."""
        if name in self.data_managers:
            del self.data_managers[name]

    def has_data_manager(self, name):
        """Check if a data manager exists."""
        return name in self.data_managers

    def get_names(self):
        """Get all data manager names."""
        return list(self.data_managers.keys())

    def __getitem__(self, name):
        """Allow dictionary-style access."""
        return self.data_managers[name]

    def __setitem__(self, name, data_manager):
        """Allow dictionary-style assignment."""
        data_manager.name = name
        self.data_managers[name] = data_manager

    def __contains__(self, name):
        """Allow 'in' operator."""
        return name in self.data_managers

    def __iter__(self):
        """Allow iteration over data managers."""
        return iter(self.data_managers.values())

    def __len__(self):
        """Return number of data managers."""
        return len(self.data_managers)


class Index:
    def __init__(self, index, names=None):
        self.index = index
        self.names = names or {}
        self.name = names.get("name", "unnamed") if names else "unnamed"

    def get_display_name(self):
        """Get the display name for this index."""
        return self.names.get("display_name", self.name)

    def get_data(self):
        """Get the index data."""
        return self.index

    def set_display_name(self, display_name):
        """Set the display name for this index."""
        self.names["display_name"] = display_name

    def add_name(self, key, value):
        """Add a name/key-value pair to the index."""
        self.names[key] = value

    def get_name(self, key, default=None):
        """Get a name value by key."""
        return self.names.get(key, default)


class IndexCollection:
    def __init__(self, indexes=None):
        self.indexes = indexes or {}

    def add_index(self, index):
        """Add an index to the collection."""
        self.indexes[index.name] = index

    def get_index(self, name):
        """Get an index by name."""
        return self.indexes.get(name)

    def get_all_indexes(self):
        """Get all indexes as a dictionary."""
        return self.indexes

    def remove_index(self, name):
        """Remove an index by name."""
        if name in self.indexes:
            del self.indexes[name]

    def has_index(self, name):
        """Check if an index exists."""
        return name in self.indexes

    def get_names(self):
        """Get all index names."""
        return list(self.indexes.keys())

    def get_by_type(self, index_type):
        """Get all indexes of a specific type."""
        result = {}
        for name, index_obj in self.indexes.items():
            if index_type in name.lower():
                result[name] = index_obj
        return result

    def __getitem__(self, name):
        """Allow dictionary-style access."""
        return self.indexes[name]

    def __setitem__(self, name, index):
        """Allow dictionary-style assignment."""
        index.name = name
        self.indexes[name] = index

    def __contains__(self, name):
        """Allow 'in' operator."""
        return name in self.indexes

    def __iter__(self):
        """Allow iteration over indexes."""
        return iter(self.indexes.values())

    def __len__(self):
        """Return number of indexes."""
        return len(self.indexes)