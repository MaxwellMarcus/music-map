from sklearn.manifold import TSNE

def augment(data, **kwargs):
    """Convert data to 2D using t-SNE projection"""
    n_components = kwargs.get('n_components', 2)
    perplexity = kwargs.get('perplexity', 30.0)
    early_exaggeration = kwargs.get('early_exaggeration', 12.0)
    learning_rate = kwargs.get('learning_rate', 200.0)
    n_iter = kwargs.get('n_iter', 1000)
    n_iter_without_progress = kwargs.get('n_iter_without_progress', 300)
    min_grad_norm = kwargs.get('min_grad_norm', 1e-07)
    metric = kwargs.get('metric', 'euclidean')
    init = kwargs.get('init', 'random')
    verbose = kwargs.get('verbose', 0)
    random_state = kwargs.get('random_state', None)
    method = kwargs.get('method', 'barnes_hut')
    angle = kwargs.get('angle', 0.5)
    
    projected_data = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        n_iter=n_iter,
        n_iter_without_progress=n_iter_without_progress,
        min_grad_norm=min_grad_norm,
        metric=metric,
        init=init,
        verbose=verbose,
        random_state=random_state,
        method=method,
        angle=angle
    ).fit_transform(data)
    
    return {"data": projected_data}
