"""
Course 3 — Anomaly Detection via Gaussian Density Estimation
============================================================
Concepts: Gaussian distribution, threshold epsilon selection, F1 score.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ── 1. Dataset Generation ─────────────────────────────────────────────
rng = np.random.default_rng(42)
X_normal = rng.normal(loc=0.0, scale=1.0, size=(300, 2))
X_anomalies = rng.uniform(low=-4.0, high=4.0, size=(15, 2))

X_train = X_normal[:200]
X_val = np.vstack([X_normal[200:], X_anomalies[:5]])
y_val = np.array([0] * 100 + [1] * 5)

# ── 2. Gaussian Parameters Estimation ────────────────────────────────
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var

def multivariate_gaussian(X, mu, var):
    n = len(mu)
    denom = (2 * np.pi) ** (n / 2) * np.prod(np.sqrt(var))
    exp_term = np.exp(-0.5 * np.sum(((X - mu) ** 2) / var, axis=1))
    return exp_term / denom

mu, var = estimate_gaussian(X_train)
p_val = multivariate_gaussian(X_val, mu, var)

# ── 3. Epsilon Threshold Tuning via F1 Score ─────────────────────────
def select_threshold(y_val, p_val):
    best_epsilon = 0.0
    best_f1 = 0.0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    for eps in np.arange(min(p_val), max(p_val), step_size):
        preds = (p_val < eps).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_epsilon = eps
            
    return best_epsilon, best_f1

best_eps, best_f1 = select_threshold(y_val, p_val)
print(f"Optimal Epsilon (ε): {best_eps:.6f}")
print(f"Validation F1 Score: {best_f1:.4f}")

# ── 4. Visualization ──────────────────────────────────────────────────
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
XX, YY = np.meshgrid(x_vals, y_vals)
grid = np.c_[XX.ravel(), YY.ravel()]
Z = multivariate_gaussian(grid, mu, var).reshape(XX.shape)

plt.figure(figsize=(7, 5))
plt.contour(XX, YY, Z, levels=10, cmap="viridis")
p_train = multivariate_gaussian(X_train, mu, var)
anomalies_detected = X_train[p_train < best_eps]

plt.scatter(X_train[:, 0], X_train[:, 1], color="blue", alpha=0.5, label="Normal Data")
plt.scatter(anomalies_detected[:, 0], anomalies_detected[:, 1], color="red", s=80, facecolors="none", edgecolors="r", label="Anomalies (p < ε)")
plt.title(f"Gaussian Anomaly Detection Contours (ε = {best_eps:.4f})")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Anomaly Detection: models normal data density P(x)           ║
║ • Flag as anomaly if P(x) < ε threshold                        ║
║ • Threshold ε selected to maximize F1 score on validation set  ║
║ • Best for rare anomalies with many unseen defect types        ║
╚════════════════════════════════════════════════════════════════╝
""")
