# PCA Risk Factor Engine

A quantitative trading pipeline implementing Principal Component Analysis (PCA) to extract orthogonal market risk factors, enabling macro-neutral statistical arbitrage.

## 🏛 Objective

In quantitative finance, asset returns are driven by overlapping, unobservable forces. This engine ingests raw market data, strips away expected drift, and decomposes the empirical covariance matrix into strictly independent (orthogonal) dimensions of risk. By isolating the "Macro Tide" (the primary eigenvector) from the "Micro Spreads" (subsequent eigenvectors), this architecture allows trading algorithms to mathematically quarantine capital from systemic market crashes.

## ⚙️ Mathematical Architecture

The pipeline is structured into four sequential, vectorized engines:

### 1. Synthetic Market Generation (src/generator.py)

Randomly generated covariance matrices violate the laws of probability (yielding negative variances and crashing PCA solvers). This module physically constructs a strictly Positive Semi-Definite (PSD) matrix utilizing factor loadings and idiosyncratic noise:

$\Sigma = LL^T + D$

Outputs a simulated log-return matrix: $R \in \mathbb{R}^{T \times N}$

![Data Generated](https://github.com/user-attachments/assets/32f87ff0-cefe-4dc7-93ee-5c01e4f173ec)

### 2. The Covariance Engine (src/covariance.py)

Isolates pure kinetic energy (variance) by stripping the expected daily drift (the ocean current) using NumPy broadcasting, bypassing computationally expensive iterative loops.

- Mean Centering: $X = R - \mathbf{1}\mu^T$

![Data Centered](https://github.com/user-attachments/assets/f050edc8-b4f9-4f68-be43-044156342812)

- Empirical Covariance: $\Sigma = \frac{1}{T-1} X^T X$

![Covariance Heatmap](https://github.com/user-attachments/assets/688762c3-2b65-427a-8bc3-98aad7174e6c)

### 3. Spectral Decomposition (src/pca.py)

Utilizes a Hermitian solver (np.linalg.eigh) to extract real eigenvalues and strictly orthogonal eigenvectors, ensuring the mathematical geometry perfectly aligns with the market's risk structure.

$\Sigma = W \Lambda W^T$

### 4. Dimensional Compression

Translates raw kinetic energy into Trace Equivalence (percentage of total system risk). It computes the cumulative sum of the variance via the C-backend and truncates the matrices once a defined threshold (e.g., $95\%$) is breached, dropping statistical noise.

$V_i = \frac{\lambda_i}{\sum_{j=1}^N \lambda_j}$

![PCA Eigenvectors](https://github.com/user-attachments/assets/a01a7137-4728-41ec-8c2a-b48fa8ee64e0)

### 📂 Repository Structure

```pca-risk-factor-engine/
│
├── src/
│   ├── __init__.py
│   ├── generator.py      # PSD Tangle Map & Synthetic Logbook
│   ├── covariance.py     # Vectorized Centering & Empirical Covariance
│   └── pca.py            # Hermitian Eigendecomposition & Compression
│
├── notebooks/            # Interactive Research & Visualization
│   ├── 01_data_generation_and_psd.ipynb
│   ├── 02_covariance_and_centering.ipynb
│   └── 03_pca_and_variance_attribution.ipynb
│
├── data/                 # Binary .npy storage for matrix states
│
├── main.py               # Master orchestration switchboard
├── visualize_pca.py      # 2D visual mapping of the PCA physics
└── README.md
```


### 📓 Interactive Research (Notebooks)

The notebooks/ directory contains the theoretical proofs translated into interactive Socratic Sprints.

**Notebook 01** demonstrates why standard random generation crashes PCA, proving the necessity of the $LL^T + D$ architecture.

**Notebook 02** visually maps the physical difference between raw directional drift ($\mu$) and pure, isolated kinetic energy ($\Sigma$).

**Notebook 03** breaks down the Eigenbasis, isolating the Macro Tide from the Micro Spreads, and tests the variance compression thresholds.

**Notebook 04** provides a 2D visual mapping of the PCA physics utilizing matplotlib to render the geometric projections.

### 🚀 Execution

The pipeline is orchestrated entirely through **main.py**.

**Execute the pipeline with default synthetic generation (T=1000, N=5)
python main.py**

**Expected Output Readout:**
- [SYSTEM] Selected Variance Explained: [0.82, 0.11, 0.05]
- [SYSTEM] Compressed Factors: (5, 3)


### 🧠 Strategic Application

**Eigenvector 1 ($w_1$):** The Macro Tide. Represents systemic market risk (e.g., global interest rate shifts). Yields the highest eigenvalue. Useful for Beta tracking; deadly for statistical arbitrage.

**Eigenvectors 2+ ($w_2, w_3...$):** The Micro Spreads. Independent, orthogonal risk dimensions isolating sector-specific or relative-value relationships. Immune to $w_1$ shocks.