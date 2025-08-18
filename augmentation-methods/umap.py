import umap

def augment(data, **kwargs):
    """Convert data to 2D using UMAP projection"""
    projected_data = umap.UMAP(**kwargs).fit_transform(data)
    return {"data": projected_data}
