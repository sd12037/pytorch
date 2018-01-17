import torch
import numpy as np

def time_normalization(x, time_dim):
    return (x-x.mean(dim=time_dim, keepdim=True))/(torch.sqrt(x.var(dim=time_dim, keepdim=True))+1e-8)

def elec_map2d(X):
    batch_size = X.shape[0]
    X_spat = np.zeros((batch_size, X.shape[-1], 1, 5, 7))
    X_spat[:, :, 0, 1, 0] = X[:, 0, :]
    X_spat[:, :, 0, 1, 1] = X[:, 1, :]
    X_spat[:, :, 0, 1, 2] = X[:, 2, :]
    X_spat[:, :, 0, 1, 3] = X[:, 3, :]
    X_spat[:, :, 0, 1, 4] = X[:, 4, :]
    X_spat[:, :, 0, 1, 5] = X[:, 5, :]
    X_spat[:, :, 0, 1, 6] = X[:, 6, :]

    X_spat[:, :, 0, 2, 0] = X[:, 7, :]
    X_spat[:, :, 0, 2, 1] = X[:, 8, :]
    X_spat[:, :, 0, 2, 2] = X[:, 9, :]
    X_spat[:, :, 0, 2, 3] = X[:, 10, :]
    X_spat[:, :, 0, 2, 4] = X[:, 11, :]
    X_spat[:, :, 0, 2, 5] = X[:, 12, :]
    X_spat[:, :, 0, 2, 6] = X[:, 13, :]

    X_spat[:, :, 0, 3, 0] = X[:, 14, :]
    X_spat[:, :, 0, 3, 1] = X[:, 15, :]
    X_spat[:, :, 0, 3, 2] = X[:, 16, :]
    X_spat[:, :, 0, 3, 3] = X[:, 17, :]
    X_spat[:, :, 0, 3, 4] = X[:, 18, :]
    X_spat[:, :, 0, 3, 5] = X[:, 19, :]
    X_spat[:, :, 0, 3, 6] = X[:, 20, :]

    X_spat[:, :, 0, 0, 0] = X[:, 30, :]
    X_spat[:, :, 0, 0, 1] = X[:, 31, :]
    X_spat[:, :, 0, 0, 2] = X[:, 32, :]
    X_spat[:, :, 0, 0, 3] = X[:, 33, :]
    X_spat[:, :, 0, 0, 4] = X[:, 34, :]
    X_spat[:, :, 0, 0, 5] = X[:, 35, :]
    X_spat[:, :, 0, 0, 6] = X[:, 36, :]

    X_spat[:, :, 0, 4, 0] = X[:, 47, :]
    X_spat[:, :, 0, 4, 1] = X[:, 48, :]
    X_spat[:, :, 0, 4, 2] = X[:, 49, :]
    X_spat[:, :, 0, 4, 3] = X[:, 50, :]
    X_spat[:, :, 0, 4, 4] = X[:, 51, :]
    X_spat[:, :, 0, 4, 5] = X[:, 52, :]
    X_spat[:, :, 0, 4, 6] = X[:, 53, :]
    return X_spat

