"""
Course 3 — Principal Component Analysis (PCA)
=============================================
Concepts: covariance matrix, eigenvectors/values, dimensionality reduction, reconstruction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 1. Synthetic 3D Data Setup ────────────────────────────────────────
rng = np.random.default_rng(42)
x1 = rng.normal(0, 1, 200)
x2 = 2 * x1 + rng.normal(0, 0.2, 200)
x3 = -0.5 * x2 + rng.normal(0, 0.1, 200)
X_3d = np.column_stack([x1, x2, x3])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_3d)

# ── 2. PCA from Scratch via Covariance & Eigen Decomposition ──────────
cov_matrix = np.cov(X_scaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by descending eigenvalue magnitude
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

var_exp_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio (Scratch):", var_exp_ratio.round(4))

# Project 3D -> 2D
k = 2
U_k = eigenvectors[:, :k]
Z_2d = X_scaled @ U_k

# Reconstruct 3D from 2D
X_rec = Z_2d @ U_k.T

# ── 3. Sklearn PCA Verification ───────────────────────────────────────
pca_sk = PCA(n_components=2)
Z_sk = pca_sk.fit_transform(X_scaled)
print("Explained Variance Ratio (Sklearn):", pca_sk.explained_variance_ratio_.round(4))

# ── 4. Visualizing Explained Variance & Projection ────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, 4), np.cumsum(var_exp_ratio), "ro-", linewidth=2)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.title("Cumulative Explained Variance Knee Plot")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c="purple", alpha=0.6)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("2D Projection of 3D Data via PCA")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • PCA: reduces dimensions by finding orthogonal max-variance axes║
║ • Mean-normalization & scaling MANDATORY before PCA            ║
║ • Eigenvalues give variance retained by each principal component ║
║ • Reconstruction: X_approx = Z @ U_k^T                         ║
╚════════════════════════════════════════════════════════════════╝
""")
