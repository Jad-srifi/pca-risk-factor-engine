from pathlib import Path
from pipeline import run_pipeline

if __name__ == '__main__':
    current_dir = Path(__file__).resolve()
    data_path = current_dir.parent / 'data' / 'synthetic_market_1000X5_19.npy'

    nVal, nVec = run_pipeline(load_path=data_path)
    print(f"[SYSTEM] Selected Variance Explained: {nVal}")
    print(f"[SYSTEM] Compressed Factors: {nVec.shape}")