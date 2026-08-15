import numpy as np
from pathlib import Path
from src.generator import generate_synthetic_data
from src.covariance import empirical_cov_matrix
from src.pca import extract_eigen, variance_explained, reduce_dimensions

def run_pipeline(load_path=None, seed: int=19, T: int=1000, N: int=5) -> tuple[np.ndarray, np.ndarray]:
    if load_path is None:
        data = generate_synthetic_data(T, N, seed)
    else:
        try:
            data = np.load(load_path)
        except:
            raise ValueError("Loading path no found")

    covariance_matrix = empirical_cov_matrix(data)
    
    eigenVal, eigenVec = extract_eigen(covariance_matrix)
    
    variance = variance_explained(eigenVal)
    
    compressed_eigenVal, compressed_eigenVec = reduce_dimensions(variance, eigenVec)
    
    return (compressed_eigenVal, compressed_eigenVec)