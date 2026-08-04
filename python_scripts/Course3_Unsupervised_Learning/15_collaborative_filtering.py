"""
Course 3 — Collaborative Filtering Recommender Systems
=======================================================
Concepts: matrix factorization R ~ U*V^T, joint cost function, GD for recommendations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. User-Item Rating Matrix Setup ──────────────────────────────────
# R matrix: 4 users x 5 items (0 indicates unrated item)
R = np.array([
    [5, 4, 0, 0, 1],
    [0, 0, 4, 5, 0],
    [3, 0, 0, 2, 4],
    [0, 3, 5, 0, 0]
], dtype=float)

num_users, num_items = R.shape
num_features = 3
lambda_reg = 0.1

rng = np.random.default_rng(42)
U = rng.normal(0, 0.5, (num_users, num_features))
V = rng.normal(0, 0.5, (num_items, num_features))

# ── 2. Joint Cost Function & Gradient Descent ─────────────────────────
def compute_cf_cost(U, V, R, lambda_reg):
    mask = (R > 0)
    pred = U @ V.T
    err = (pred - R) * mask
    cost = 0.5 * np.sum(err ** 2) + 0.5 * lambda_reg * (np.sum(U ** 2) + np.sum(V ** 2))
    return cost

print("Training Matrix Factorization Collaborative Filtering:")
alpha = 0.02
for i in range(1001):
    mask = (R > 0)
    pred = U @ V.T
    err = (pred - R) * mask
    
    dU = err @ V + lambda_reg * U
    dV = err.T @ U + lambda_reg * V
    
    U -= alpha * dU
    V -= alpha * dV
    
    if i % 250 == 0:
        cost = compute_cf_cost(U, V, R, lambda_reg)
        print(f"  Iter {i:4d}: Cost = {cost:.4f}")

# ── 3. Predictions & Item Similarity ─────────────────────────────────
predicted_ratings = U @ V.T
print("\nOriginal Ratings (0 = Unrated):\n", R)
print("\nPredicted Full Ratings Matrix:\n", predicted_ratings.round(2))

item_sim = cosine_similarity(V)
print("\nItem-Item Cosine Similarity Matrix:\n", item_sim.round(3))

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Collaborative Filtering: learns latent factors U and V       ║
║ • Rating prediction: r_{ij} ≈ U_i · V_j^T                      ║
║ • Joint cost minimizes squared error on observed ratings       ║
║ • Cold-start problem: new users/items require content features ║
╚════════════════════════════════════════════════════════════════╝
""")
