import numpy as np

def augment(data, transformation):
    # data: (n, m), transformation: (m, i)
    return {"data": np.asarray(data) @ np.asarray(transformation)}
