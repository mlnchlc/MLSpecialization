"""
Course 3 — K-Means Clustering
===============================
Concepts: unsupervised learning, centroid initialisation, Elbow method, visualisation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.helpers import make_blobs

# ── Generate Data ─────────────────────────────────────────────────────

X, y_true = make_blobs(n=300, centers=3, seed=42)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", edgecolor="k", s=40)
plt.title("Ground Truth (for reference)")
plt.gca().set_aspect("equal")

# ── K-Means Implementation from Scratch ───────────────────────────────

print("── K-Means from Scratch ──")


def initialize_centroids(X, k, seed=42):
    """Randomly pick k points as centroids."""
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], k, replace=False)]


def assign_clusters(X, centroids):
    """Assign each point to nearest centroid (Euclidean distance)."""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """Compute mean of points in each cluster as new centroid."""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])


def kmeans(X, k, n_iters=100, seed=42):
    """Full K-Means algorithm."""
    centroids = initialize_centroids(X, k, seed)
    for i in range(n_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            print(f"  Converged at iteration {i + 1}")
            break
        centroids = new_centroids
    return labels, centroids


labels, centroids = kmeans(X, k=3)

print(f"  Centroids:\n{centroids}")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolor="k", s=40)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X",
            s=200, linewidths=2, edgecolors="black", label="Centroids")
plt.title("K-Means Clustering (from scratch)")
plt.legend()
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

# ── Elbow Method ──────────────────────────────────────────────────────

print("\n── Elbow Method ──")

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 3.5))
plt.plot(K_range, inertias, "bo-", linewidth=1.5)
plt.axvline(3, color="red", ls="--", alpha=0.6, label="True k=3")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method — choose k where inertia bends")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ── Silhouette Score ──────────────────────────────────────────────────

print("\n── Silhouette Score ──")

for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"  k={k}: Silhouette = {score:.4f}")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • K-Means: iterative centroid-based clustering                ║
║   1. Randomly initialise k centroids                          ║
║   2. Assign each point to nearest centroid                    ║
║   3. Recompute centroids as mean of cluster points            ║
║   4. Repeat until convergence                                 ║
║ • Distortion / Inertia: sum of squared distances to centroid  ║
║ • Elbow Method: plot k vs inertia, pick the "bend" point      ║
║ • Silhouette Score: measures cluster cohesion & separation     ║
║ → ranges [-1, 1]; higher = better                             ║
║ • K-Means is sensitive to initialisation (run multiple times) ║
╚════════════════════════════════════════════════════════════════╝
""")
