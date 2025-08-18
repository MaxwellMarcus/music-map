def augment( data, dist_data ):
    return { "data": data @ dist_data.T }