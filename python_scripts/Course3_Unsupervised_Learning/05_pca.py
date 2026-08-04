"""
Course 3 — Principal Component Analysis (PCA)
==============================================
Concepts: dimensionality reduction, covariance matrix, eigenvalues/eigenvectors,
explained variance, data reconstruction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── 1. Generate Synthetic 3D Data ──────────────────────────────────────
# Points that lie mostly on a 2D plane in a 3D space.
rng = np.random.default_rng(42)
n_samples = 150
x1 = rng.uniform(-2, 2, n_samples)
x2 = rng.uniform(-2, 2, n_samples)
# x3 is a linear combination of x1 and x2, with some Gaussian noise added
x3 = 0.8 * x1 + 1.2 * x2 + rng.normal(0, 0.15, n_samples)

X = np.column_stack((x1, x2, x3))

# Mean normalize and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. PCA from Scratch ────────────────────────────────────────────────
print("── PCA from Scratch ──")

# Covariance matrix: Sigma = (1/m) * X^T * X
m = X_scaled.shape[0]
cov_matrix = (1 / m) * (X_scaled.T @ X_scaled)
print(f"Covariance Matrix:\n{cov_matrix.round(4)}")

# Eigen-decomposition (using np.linalg.eigh since covariance matrix is symmetric)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nEigenvalues (Variance explained by each component):")
print(eigenvalues)
print("\nEigenvectors (Principal Components / load vectors):")
print(eigenvectors)

# Project the data to k dimensions (k = 2)
k = 2
U_k = eigenvectors[:, :k]  # top k components
X_projected_scratch = X_scaled @ U_k

# Cumulative explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio_scratch = eigenvalues / total_variance
cumulative_variance_scratch = np.cumsum(explained_variance_ratio_scratch)

print(f"\nExplained Variance Ratio:      {explained_variance_ratio_scratch[:k].round(4)}")
print(f"Cumulative Explained Variance: {cumulative_variance_scratch[k-1]:.4f}")

# Reconstruct the data from projected representation
X_reconstructed_scratch = X_projected_scratch @ U_k.T

# ── 3. PCA with Scikit-Learn ───────────────────────────────────────────
print("\n── PCA with Scikit-Learn ──")

pca_sklearn = PCA(n_components=k)
X_projected_sklearn = pca_sklearn.fit_transform(X_scaled)

print(f"Sklearn Components (eigenvectors):\n{pca_sklearn.components_}")
print(f"Sklearn Explained Variance Ratio: {pca_sklearn.explained_variance_ratio_.round(4)}")
print(f"Sklearn Singular Values:          {pca_sklearn.singular_values_.round(4)}")

# Compare components (note: sign flip is possible and normal in PCA)
diff_components = np.abs(np.abs(U_k.T) - np.abs(pca_sklearn.components_))
print(f"\nMax difference in absolute values of components: {np.max(diff_components):.2e}")

# Compare projection
diff_projection = np.abs(np.abs(X_projected_scratch) - np.abs(X_projected_sklearn))
print(f"Max difference in absolute values of projection: {np.max(diff_projection):.2e}")

# ── 4. Visualization ───────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))

# Plot 1: Original 3D Data & Principal Component Axes
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c='blue', alpha=0.5, label='Data points')

# Draw principal component axes
# Scale eigenvectors for visibility in plot
scale = 2.0
mean_point = np.zeros(3)
for i in range(3):
    v = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
    ax1.plot([mean_point[0], v[0]], [mean_point[1], v[1]], [mean_point[2], v[2]], 
             color='red', linewidth=3, label='PC Axes' if i == 0 else "")

ax1.set_xlabel('X1 (Scaled)')
ax1.set_ylabel('X2 (Scaled)')
ax1.set_zlabel('X3 (Scaled)')
ax1.set_title('Original 3D Data & PC Axes')
ax1.legend()

# Plot 2: 2D Projected Data
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(X_projected_scratch[:, 0], X_projected_scratch[:, 1], c='green', alpha=0.6)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title('Projected 2D Data')
ax2.grid(alpha=0.3)

# Plot 3: Explained Variance
ax3 = fig.add_subplot(1, 3, 3)
components_num = np.arange(1, 4)
ax3.bar(components_num, explained_variance_ratio_scratch, alpha=0.6, align='center',
        label='Individual explained variance')
ax3.step(components_num, cumulative_variance_scratch, where='mid',
         label='Cumulative explained variance', color='red')
ax3.set_xticks(components_num)
ax3.set_xlabel('Principal Component Index')
ax3.set_ylabel('Explained Variance Ratio')
ax3.set_title('Explained Variance vs. Components')
ax3.legend(loc='best')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • PCA: Unsupervised dimensionality reduction method            ║
║ • Purpose: Compress features, visualize high-D data, speed up  ║
║   model training.                                              ║
║ • Scaling: ALWAYS mean-normalize and scale features first so   ║
║   variance isn't dominated by different feature units.         ║
║ • Math: Compute covariance matrix, find eigenvectors/values    ║
║ • Variance Retained: Sum of top k eigenvalues / sum of all     ║
║   eigenvalues. Usually aim for 95-99% variance retained.       ║
║ • Reconstruction: Project back using X_recon = Z * U_k^T       ║
╚════════════════════════════════════════════════════════════════╝
""")
