import numpy as np

def mean_centering(R: np.ndarray) -> np.ndarray:
    mean = np.mean(R, axis=0)
    mean = np.reshape(mean, (R.shape[1], 1))
    ones = np.ones((R.shape[0], 1))
    
    X = R - ones @ mean.T
    
    return X
    

def empirical_cov_matrix(R: np.ndarray) -> np.ndarray:
    X = mean_centering(R)
    return 1/(X.shape[0] - 1) * (X.T @ X)
