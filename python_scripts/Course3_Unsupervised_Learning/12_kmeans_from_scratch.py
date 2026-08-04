"""
Course 3 — K-Means Clustering from Scratch & Image Compression
=================================================================
Concepts: K-Means algorithm, centroid update, Elbow method, image quantization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.helpers import make_blobs

# ── 1. K-Means Algorithm from Scratch ─────────────────────────────────
def initialize_centroids(X, K, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), K, replace=False)
    return X[indices]

def find_closest_centroids(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

def run_kmeans_scratch(X, K, max_iters=10, seed=42):
    centroids = initialize_centroids(X, K, seed)
    for _ in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

# Test on 2D synthetic blobs
X, _ = make_blobs(n=300, centers=3, seed=42)
centroids, cluster_idx = run_kmeans_scratch(X, K=3)

# ── 2. Image Color Quantization / Compression Demo ────────────────────
# Create synthetic RGB image (64x64)
rng = np.random.default_rng(42)
image = rng.uniform(0, 1, size=(64, 64, 3))
X_image = image.reshape(-1, 3)

# Compress to K=4 colors using K-Means
kmeans_img = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_img.fit(X_image)
X_compressed = kmeans_img.cluster_centers_[kmeans_img.labels_]
image_compressed = X_compressed.reshape(64, 64, 3)

# ── 3. Visualization ──────────────────────────────────────────────────
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=cluster_idx, cmap="viridis", alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.title("K-Means (Scratch) Clustering")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(image_compressed)
plt.title("Image Color Compression (K=4 Colors)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • K-Means: iterative assignment & centroid update algorithm    ║
║ • Objective: minimize inertia sum ||x - mu_k||^2               ║
║ • Elbow Method: plot inertia vs K to find optimal cluster count║
║ • Image Compression: represent millions of colors with K colors║
╚════════════════════════════════════════════════════════════════╝
""")
