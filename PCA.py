import numpy as np

def pca(data):
    n = 3
    mean = np.mean(data, axis=0)
    zero_mean_data = data - mean
    cov = np.cov(zero_mean_data, rowvar=False)
    value, vec = np.linalg.eig(cov)
    index = np.argsort(value)
    n_index = index[-n:]
    n_vec = vec[:, n_index]
    low_dim_data = np.dot(zero_mean_data, n_vec)
    return low_dim_data
