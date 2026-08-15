from pathlib import Path
import numpy as np

def generate_R(T: int, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mu, cov, T)

def generate_cov(N: int) -> np.ndarray:
    L = np.random.uniform(0.5, 1.5, size=(N, 1))
    
    D_variances = np.random.uniform(0.1, 0.5, size=N)
    D = np.diag(D_variances)
    
    cov = L @ L.T + D
    
    assert np.allclose(cov, cov.T), "Covariance matrix is assymetric"
    
    return cov

def generate_mu(N: int) -> np.ndarray:
    return np.random.uniform(-0.03, 0.03, size=N)

def generate_synthetic_data(T: int, N: int, seed: int=19) -> np.ndarray:
    np.random.seed(seed)
    
    cov = generate_cov(N)
    mu = generate_mu(N)
    
    if T <= 0 or N <= 0:
        raise ValueError(f'Matrix Size should be a positive non zero value')
    
    elif cov.shape != (N, N):
        raise ValueError(f'Covariance matrix shape mismatch. Expected ({N}, {N})')
    
    elif mu.shape[0] != N:
        raise ValueError(f'Mu matrix shape mismatch. Expected ({N}, )')
    
    R = generate_R(T, mu, cov)
    
    if R.shape == (T, N): 
        current_dir = Path(__file__).resolve()
        
        target_dir = current_dir.parent.parent / 'data'
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / f'synthetic_market_{T}X{N}_{seed}.npy'
        
        np.save(file_path, R)
        return R
    
    raise ValueError(f'R matrix output shape mismatch. Expected ({T}, {N})')