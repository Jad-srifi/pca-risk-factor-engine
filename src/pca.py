import numpy as np

def extract_eigen(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenValues, eigenVectors = np.linalg.eigh(sigma)
    
    sorted_eigenvalues_index = np.argsort(eigenValues)[::-1]
    
    sorted_eigenvectors = eigenVectors[:, sorted_eigenvalues_index]
    sorted_eigenvalues = eigenValues[sorted_eigenvalues_index]
    
    return (sorted_eigenvalues, sorted_eigenvectors)

def variance_explained(eigenvalues: np.ndarray) -> np.ndarray:
    variance_ratio = eigenvalues / np.sum(eigenvalues)
    return variance_ratio

def reduce_dimensions(eigenvalues: np.ndarray, eigenvectors: np.ndarray, threshold=0.95) -> tuple[np.ndarray, np.ndarray]:
    cumulative_variance = np.cumsum(eigenvalues)
    
    cutoff_index = np.argmax(cumulative_variance >= threshold)
    
    compressed_eigenvalues = eigenvalues[:cutoff_index+1]
    compressed_eigenvectors = eigenvectors[:, :cutoff_index+1]
    
    return (compressed_eigenvalues, compressed_eigenvectors)