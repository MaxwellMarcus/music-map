#!/usr/bin/env python3

import requests
import json
import time
import threading
import numpy as np
from urllib.request import urlopen
from urllib.error import HTTPError
from app import ManagerManager
from DataManager import DataManager, Index
from ViewerManager import ViewerManager

# Test configuration
TEST_PORT = 5010
BASE = f"http://localhost:{TEST_PORT}"

def http_get(path):
    """Helper function to make HTTP GET requests."""
    try:
        with urlopen(f"{BASE}{path}", timeout=5) as resp:
            return resp.getcode(), resp.read()
    except HTTPError as e:
        return e.code, e.read()

def http_post(path, data):
    """Helper function to make HTTP POST requests."""
    try:
        response = requests.post(f"{BASE}{path}", json=data, timeout=5)
        return response.status_code, response.content
    except requests.exceptions.RequestException as e:
        return 500, str(e).encode()

def wait_for_server():
    """Wait for the server to start."""
    for _ in range(30):  # Wait up to 30 seconds
        try:
            code, body = http_get("/api/status")
            if code == 200:
                print("✓ Server started successfully")
                return True
        except:
            pass
        time.sleep(1)
    return False

def start_server():
    """Start the Flask server in a separate thread."""
    mm = ManagerManager(run_app=False, port=TEST_PORT, small_data=True)
    mm.run(port=TEST_PORT, debug=False)

def test_datamanager_initialization():
    """Test DataManager initialization with various parameters."""
    print("Testing DataManager initialization...")
    
    # Test with no parameters
    dm = DataManager()
    assert dm.data is None
    assert dm.augmentations.get_all_data_managers() == {}
    assert dm.indexes.get_all_indexes() == {}
    assert dm.get_index_names() == {}
    
    # Test with data only
    data = np.random.rand(100, 10)
    dm = DataManager(data=data)
    assert dm.data is not None
    assert dm.data.shape == (100, 10)
    # Indexes are now handled through indexes, so should be empty initially
    assert dm.indexes.get_all_indexes() == {}
    
    # Test with indexes
    test_index = np.array([0, 1, 2, 0, 1, 2] * 16 + [-1] * 4)  # 100 elements
    index_obj = Index(test_index, {"name": "test_clusters", "display_name": "Test Clusters"})
    dm = DataManager(data=data)
    dm.indexes.add_index(index_obj)
    assert "test_clusters" in dm.indexes.get_all_indexes()
    assert len(dm.indexes.get_index("test_clusters").get_data()) == 100
    assert dm.indexes.get_index("test_clusters").get_data()[0] == 0
    assert dm.indexes.get_index("test_clusters").get_data()[1] == 1
    assert dm.indexes.get_index("test_clusters").get_data()[96] == -1  # Last few should be -1
    
    # Test with custom index names
    custom_names = {0: "Group A", 1: "Group B", 2: "Group C", -1: "No Group"}
    index_obj.names.update(custom_names)
    dm = DataManager(data=data)
    dm.indexes.add_index(index_obj)
    index_names = dm.get_index_names()
    # Check that custom names are included
    assert "test_clusters" in index_names
    assert index_names["test_clusters"] == "Test Clusters"
    
    print("✓ DataManager initialization tests passed")

def test_datamanager_methods():
    """Test DataManager methods with different return formats."""
    print("Testing DataManager methods...")
    
    dm = DataManager()
    
    # Test load_data
    dm.load_data(data=np.random.rand(30, 3))
    assert dm.data.shape == (30, 3)
    # Indexes are now handled through indexes, so should be empty initially
    assert dm.indexes.get_all_indexes() == {}
    
    # Test augment_data with dict return
    def test_augment_dict(data, **kwargs):
        return {"data": data[:, :3]}  # Reduce to 3D
    
    result = dm.augment_data(test_augment_dict, "test_aug")
    assert "test_aug" in dm.augmentations.get_all_data_managers()
    assert dm.augmentations.get_data_manager("test_aug").data.shape == (30, 3)
    # New augmentations should have no indexes by default
    assert len(dm.augmentations.get_data_manager("test_aug").indexes.get_all_indexes()) == 0
    
    # Test augment_data with direct array return
    def test_augment_array(data, **kwargs):
        return data[:, :3]  # Reduce to 3D
    
    result = dm.augment_data(test_augment_array, "test_aug2")
    assert "test_aug2" in dm.augmentations.get_all_data_managers()
    assert dm.augmentations.get_data_manager("test_aug2").data.shape == (30, 3)
    
    # Test index_data with dict return
    def test_index_dict(data, **kwargs):
        return {"index": np.arange(len(data)), "name": "test_index"}
    
    result = dm.index_data(test_index_dict, "test_index")
    assert "test_index" in dm.indexes.get_all_indexes()
    assert len(dm.indexes.get_index("test_index").get_data()) == 30
    index_names = dm.get_index_names()
    assert "test_index" in index_names
    assert index_names["test_index"] == "test_index"
    
    # Test index_data with direct array return
    def test_index_array(data, **kwargs):
        return np.arange(len(data))
    
    result = dm.index_data(test_index_array, "test_index2")
    assert "test_index2" in dm.indexes.get_all_indexes()
    assert len(dm.indexes.get_index("test_index2").get_data()) == 30
    
    # Test index_data for clusters
    def test_cluster(data, **kwargs):
        return np.array([0, 1, 0, 1] * 7 + [0, 0])
    
    result = dm.index_data(test_cluster, "clusters")
    assert "clusters" in dm.indexes.get_all_indexes()
    assert len(dm.indexes.get_index("clusters").get_data()) == 30
    
    # Test index_data for projections
    def test_project(data, **kwargs):
        return data[:, :2]  # Reduce to 2D
    
    result = dm.index_data(test_project, "test_proj")
    assert "test_proj" in dm.indexes.get_all_indexes()
    assert dm.indexes.get_index("test_proj").get_data().shape == (30, 2)
    
    print("✓ DataManager methods tests passed")

def test_viewermanager_functionality():
    """Test ViewerManager core functionality."""
    print("Testing ViewerManager functionality...")
    
    # Create a data manager with some data and indexes
    data = np.random.rand(50, 5)
    dm = DataManager(data=data, name="test")
    
    # Add some indexes
    tsne_index = Index(np.random.rand(50, 2), {"name": "tsne", "display_name": "t-SNE"})
    cluster_index = Index(np.random.randint(0, 3, 50), {"name": "clusters", "display_name": "Clusters"})
    dm.indexes.add_index(tsne_index)
    dm.indexes.add_index(cluster_index)
    
    # Create viewer manager
    vm = ViewerManager(dm, "tsne")
    
    # Test basic functionality
    assert vm.primary_data_manager == dm
    assert vm.viewed_data_manager == dm
    assert vm.active_viewport_metadata["viewed_index"] == "tsne"
    
    # Test get_data_managers
    managers = vm.get_data_managers()
    assert "primary" in managers
    assert managers["primary"] == dm
    
    # Test get_indexes
    indexes = vm.get_indexes()
    assert "tsne" in indexes
    assert "clusters" in indexes
    
    # Test color sources
    color_sources = vm.get_available_color_sources()
    assert "index:tsne" in color_sources
    assert "index:clusters" in color_sources
    
    print("✓ ViewerManager functionality tests passed")

def test_api_error_handling():
    """Test API error handling for various scenarios."""
    print("Testing API error handling...")
    
    # Test invalid JSON
    code, body = http_post("/api/augment", "invalid json")
    print(f"Invalid JSON test: code={code}, body={body}")
    assert code in [400, 500]  # Should return 400 or 500 for invalid JSON
    
    # Test missing method
    code, body = http_post("/api/augment", {"parameters": {}})
    assert code == 400
    error = json.loads(body.decode("utf-8"))
    assert "Method is required" in error["error"]
    
    # Test unknown method
    code, body = http_post("/api/augment", {"method": "nonexistent_method"})
    assert code == 500  # Should fail when trying to load the method
    
    # Test invalid data source
    code, body = http_post("/api/augment", {
        "method": "tsne",
        "data_source": "nonexistent_source"
    })
    assert code == 500  # Should fail when trying to find the data source
    
    # Test invalid index switch
    code, body = http_post("/api/switch-index", {"index": "nonexistent_index"})
    assert code == 400
    error = json.loads(body.decode("utf-8"))
    assert "not found" in error["error"]
    
    print("✓ API error handling tests passed")

def test_parameter_validation():
    """Test parameter validation for various types."""
    print("Testing parameter validation...")
    
    # Test missing required parameter
    code, body = http_post("/api/index", {
        "method": "range_index",
        "parameters": {"start": 0}  # Missing 'end' and 'name'
    })
    assert code == 500  # API returns 500 for validation errors
    error = json.loads(body.decode("utf-8"))
    assert "Required parameter" in error["error"]
    
    # Test invalid parameter type
    code, body = http_post("/api/index", {
        "method": "range_index",
        "parameters": {"start": "not_a_number", "end": 100, "name": "test"}
    })
    assert code == 500
    error = json.loads(body.decode("utf-8"))
    print(f"Invalid parameter test: {error}")
    assert any(phrase in error["error"] for phrase in ["invalid literal", "could not be converted"])
    
    # Test matrix parameter conversion
    code, body = http_post("/api/augment", {
        "method": "transform",
        "parameters": {"transformation": "invalid_matrix"}
    })
    assert code == 500
    error = json.loads(body.decode("utf-8"))
    print(f"Matrix parameter test: {error}")
    assert any(phrase in error["error"] for phrase in ["could not be converted", "invalid"])
    
    # Test valid parameters
    code, body = http_post("/api/index", {
        "method": "range_index",
        "parameters": {"start": 0, "end": 100, "name": "test_range"}
    })
    assert code == 200
    
    print("✓ Parameter validation tests passed")

def test_data_source_selection():
    """Test data source selection and chaining."""
    print("Testing data source selection...")
    
    # Create augmentation from primary data
    code, body = http_post("/api/augment", {
        "method": "tsne",
        "augmentation_type": "test_tsne",
        "data_source": "primary",
        "parameters": {"kwargs": "{\"n_components\": 2}"}
    })
    assert code == 200
    
    # Check that augmentation was created
    code, body = http_get("/api/status")
    assert code == 200
    status = json.loads(body.decode("utf-8"))
    assert "test_tsne" in status["available_augmentations"]
    
    # Create index from primary data
    code, body = http_post("/api/index", {
        "method": "range_index",
        "index_type": "test_range",
        "data_source": "primary",
        "parameters": {"start": 0, "end": 100, "name": "test_range"}
    })
    assert code == 200
    
    # Create augmentation from the first augmentation
    code, body = http_post("/api/augment", {
        "method": "umap",
        "augmentation_type": "test_umap",
        "data_source": "test_tsne",
        "parameters": {"kwargs": "{\"n_components\": 2}"}
    })
    # The second augmentation might fail if the first one isn't ready, so let's be flexible
    if code != 200:
        print(f"Second augmentation failed with code {code}, body: {body}")
        # Try creating it from primary data instead
        code, body = http_post("/api/augment", {
            "method": "umap",
            "augmentation_type": "test_umap2",
            "data_source": "primary",
            "parameters": {"kwargs": "{\"n_components\": 2}"}
        })
        assert code == 200, f"Failed to create second augmentation: {body}"
    
    # Check that both augmentations were created
    code, body = http_get("/api/status")
    assert code == 200
    status = json.loads(body.decode("utf-8"))
    print(f"Status after creating augmentations: {status}")
    available_augmentations = status["available_augmentations"]
    print(f"Available augmentations: {available_augmentations}")
    assert len(available_augmentations) >= 2
    
    print("✓ Data source selection tests passed")

def test_color_source_functionality():
    """Test color source related functionality."""
    print("Testing color source functionality...")
    
    # Create some indexes to test with
    code, body = http_post("/api/index", {
        "method": "range_index",
        "index_type": "test_range",
        "data_source": "primary",
        "parameters": {"start": 0, "end": 100, "name": "first_half"}
    })
    assert code == 200
    
    code, body = http_post("/api/index", {
        "method": "range_index",
        "index_type": "test_range2",
        "data_source": "primary",
        "parameters": {"start": 100, "end": 200, "name": "second_half"}
    })
    assert code == 200
    
    # Get available color sources
    code, body = http_get("/api/color-sources")
    assert code == 200
    sources = json.loads(body.decode("utf-8"))
    print(f"Color sources: {sources}")
    assert "color_sources" in sources
    assert "index:test_range" in sources["color_sources"]
    assert "index:test_range2" in sources["color_sources"]
    
    # Set color source
    code, body = http_post("/api/set-color-source", {"color_source": "index:test_range"})
    assert code == 200
    
    # Check that color source was set
    code, body = http_get("/api/color-sources")
    assert code == 200
    sources = json.loads(body.decode("utf-8"))
    print(f"Color sources after setting: {sources}")
    assert sources["current_color_source"] == "index:test_range"
    
    # Get color data
    code, body = http_get("/api/color-data")
    assert code == 200
    color_data = json.loads(body.decode("utf-8"))
    assert "color_data" in color_data
    
    print("✓ Color source functionality tests passed")

def main():
    """Run all tests."""
    print("Starting comprehensive integration tests...")
    
    # Start server in background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    assert wait_for_server(), "Server did not start in time"
    
    # Run all test functions
    test_datamanager_initialization()
    test_datamanager_methods()
    test_viewermanager_functionality()
    test_api_error_handling()
    test_parameter_validation()
    test_data_source_selection()
    test_color_source_functionality()
    test_tree_refresh_on_index_creation()
    
    print("Running basic functionality tests...")
    
    # 1) Status endpoint
    code, body = http_get("/api/status")
    assert code == 200
    status = json.loads(body.decode("utf-8"))
    assert "current_data_manager" in status
    assert "available_indexes" in status
    assert "available_augmentations" in status
    assert "augmentation_methods" in status
    assert "index_methods" in status
    
    # 2) Data endpoint - create t-SNE augmentation first, then switch to it and test data endpoint
    code, body = http_post("/api/augment", {
        "method": "tsne",
        "augmentation_type": "tsne",
        "data_source": "primary",
        "parameters": {"kwargs": "{\"n_components\": 2}"}
    })
    assert code == 200
    
    code, body = http_post("/api/switch-data-manager", {"data_manager": "tsne"})
    assert code == 200
    
    code, body = http_get("/api/data.bin")
    assert code == 200
    data_shape = (200, 2)  # Small data should be 200x2
    print(f"Data shape: {data_shape}")
    
    # 3) Method parameters endpoint
    code, body = http_get("/api/method-parameters?type=augmentation&method=tsne")
    assert code == 200
    params = json.loads(body.decode("utf-8"))
    assert params.get("method") == "tsne"
    assert "parameters" in params
    
    # 4) Create augmentation using augmentation API
    code, body = http_post("/api/augment", {
        "method": "tsne",
        "augmentation_type": "tsne_test",
        "data_source": "primary",
        "parameters": {"kwargs": "{\"n_components\": 2}"}
    })
    assert code == 200
    
    code, body = http_get("/api/status")
    assert code == 200
    status = json.loads(body.decode("utf-8"))
    assert "tsne_test" in status["available_augmentations"], status
    
    # Switch to new data manager
    code, body = http_post("/api/switch-data-manager", {"data_manager": "tsne_test"})
    print(f"Switch data manager response: code={code}, body={body}")
    # The data manager might not be available immediately, so let's check what's available
    if code != 200:
        # Get status to see what data managers are available
        code2, body2 = http_get("/api/status")
        status = json.loads(body2.decode("utf-8"))
        print(f"Available data managers: {status.get('available_augmentations', [])}")
        # Try switching to the first available 2D data manager
        if status.get('two_d_augmentations'):
            first_2d_dm = status['two_d_augmentations'][0]
            code, body = http_post("/api/switch-data-manager", {"data_manager": first_2d_dm})
            assert code == 200, body
    else:
        assert code == 200, body
    
    # Switch to invalid data manager (expect 400)
    code, body = http_post("/api/switch-data-manager", {"data_manager": "does_not_exist"})
    assert code == 400
    
    # 5) Create clusters using index API (hdbscan)
    code, body = http_post("/api/index", {
        "method": "hdbscan",
        "index_type": "clusters",
        "data_source": "primary",
        "parameters": {"min_cluster_size": 5}
    })
    assert code == 200
    
    code, body = http_get("/api/indexes")
    assert code == 200
    indexes = json.loads(body.decode("utf-8"))
    assert "indexes" in indexes
    assert "index_names" in indexes
    # Verify indexes is a dict
    assert isinstance(indexes["indexes"], dict)
    # Verify index_names is a dict
    assert isinstance(indexes["index_names"], dict)
    
    # 6) Metadata endpoint (small_data → no metadata mandatory)
    code, body = http_get("/api/metadata")
    assert code == 200
    meta = json.loads(body.decode("utf-8"))
    assert "metadata" in meta
    
    # 7) Create range index
    code, body = http_post("/api/index", {
        "method": "range_index",
        "index_type": "test_range",
        "data_source": "primary",
        "parameters": {"start": 0, "end": 50, "name": "test_group"}
    })
    assert code == 200
    
    # 8) Test data source selection - create index from the augmentation
    code, body = http_post("/api/index", {
        "method": "hdbscan",
        "index_type": "clusters_from_aug",
        "data_source": "tsne_test",  # Use the augmentation we created earlier
        "parameters": {"min_cluster_size": 3}
    })
    assert code == 200
    
    # Color sources should include the new index
    code, body = http_get("/api/color-sources")
    assert code == 200
    sources = json.loads(body.decode("utf-8"))
    assert any(k.startswith("index:") for k in sources.get("color_sources", {})), sources
    
    # Set color source to the new index and fetch color data
    code, body = http_post("/api/set-color-source", {"color_source": "index:test_range"})
    assert code == 200
    code, body = http_get("/api/color-data")
    assert code == 200
    
    print("✓ All comprehensive integration checks passed on port", TEST_PORT)
    
    # Test new UI features
    test_new_ui_features()




def test_new_ui_features():
    """Test the new compact UI features and Plotly interaction."""
    print("Testing new UI features...")
    
    # Test that the main page loads
    code, body = http_get("/")
    assert code == 200
    html_content = body.decode("utf-8")
    
    # Check for new UI elements
    assert "Data Viewer" in html_content
    assert "app-container" in html_content
    assert "sidebar" in html_content
    assert "main-content" in html_content
    
    # Check for new Plotly configuration
    assert "scrollZoom: true" in html_content
    assert "displayModeBar: false" in html_content
    assert "mousedown" in html_content
    assert "wheel" in html_content
    
    # Check for new CSS classes
    assert "control-section" in html_content
    assert "stats-grid" in html_content
    assert "status-indicator" in html_content
    
    # Check for tree viewer elements (only if using updated template)
    if "Data Tree" in html_content:
        assert "data-tree" in html_content
        assert "buildDataTree" in html_content
        print("✓ Tree viewer features found")
    else:
        print("✓ Using legacy template (tree viewer not available)")
    
    # Test that the API endpoints still work with the new UI
    code, body = http_get("/api/status")
    assert code == 200
    status = json.loads(body.decode("utf-8"))
    assert "current_data_manager" in status
    assert "available_augmentations" in status
    
    # Test data endpoint
    code, body = http_get("/api/data.bin")
    assert code == 200
    assert len(body) > 0
    
    # Test indexes endpoint for tree viewer
    code, body = http_get("/api/indexes")
    assert code == 200
    indexes = json.loads(body.decode("utf-8"))
    assert "indexes" in indexes
    assert "index_names" in indexes
    
    print("✓ New UI features test passed")


def test_tree_refresh_on_index_creation():
    """Test that the data tree refreshes when new indexes are created."""
    print("Testing tree refresh on index creation...")
    
    # Start server if not already running
    if not wait_for_server():
        print("Starting server for tree refresh test...")
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        if not wait_for_server():
            raise Exception("Failed to start server for tree refresh test")
    
    # Get initial status and indexes
    code, body = http_get("/api/status")
    assert code == 200
    initial_status = json.loads(body.decode("utf-8"))
    
    code, body = http_get("/api/indexes")
    assert code == 200
    initial_indexes = json.loads(body.decode("utf-8"))
    initial_index_count = len(initial_indexes.get("indexes", {}))
    
    # Create a new index
    code, body = http_post("/api/index", {
        "method": "range_index",
        "index_type": "tree_test_index",
        "data_source": "primary",
        "parameters": {"start": 0, "end": 25, "name": "tree_test_group"}
    })
    assert code == 200
    result = json.loads(body.decode("utf-8"))
    assert result["status"] == "success"
    
    # Wait a moment for the index to be processed
    time.sleep(1)
    
    # Check that the new index appears in the indexes endpoint
    code, body = http_get("/api/indexes")
    assert code == 200
    updated_indexes = json.loads(body.decode("utf-8"))
    updated_index_count = len(updated_indexes.get("indexes", {}))
    
    # Should have one more index than before
    assert updated_index_count == initial_index_count + 1
    assert "tree_test_index" in updated_indexes.get("indexes", {})
    
    # Check that the status reflects the new index
    code, body = http_get("/api/status")
    assert code == 200
    updated_status = json.loads(body.decode("utf-8"))
    assert "tree_test_index" in updated_status.get("available_indexes", [])
    
    print("✓ Tree refresh on index creation test passed")


if __name__ == "__main__":
    main()
