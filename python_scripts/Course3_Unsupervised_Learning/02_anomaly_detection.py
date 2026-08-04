"""
Course 3 — Anomaly Detection
==============================
Concepts: Gaussian distribution, density estimation, threshold selection,
F1 score for evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ── Generate Data ─────────────────────────────────────────────────────

rng = np.random.default_rng(42)
X_normal = rng.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], size=200)
X_anomalies = rng.uniform(low=-2, high=8, size=(20, 2))

# Split: train on 160 normal data; dev & test get 20 normal + 10 anomalies each
X_train = X_normal[:160]
X_dev = np.vstack([X_normal[160:180], X_anomalies[:10]])
y_dev = np.array([0] * 20 + [1] * 10)

X_test = np.vstack([X_normal[180:200], X_anomalies[10:]])
y_test = np.array([0] * 20 + [1] * 10)

# ── Gaussian Density Estimation ───────────────────────────────────────

def estimate_gaussian(X):
    """Estimate mean and variance for each feature."""
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var


def gaussian_pdf(X, mu, var):
    """Compute P(X | mu, var) for each example."""
    eps = 1e-8
    numerator = np.exp(-((X - mu) ** 2) / (2 * var + eps))
    denominator = np.sqrt(2 * np.pi * var + eps)
    return np.prod(numerator / denominator, axis=1)


mu, var = estimate_gaussian(X_train)
print(f"Feature means:     {mu}")
print(f"Feature variances: {var}")

# ── Threshold Selection ───────────────────────────────────────────────

print("\n── Selecting ε (threshold) on Dev Set ──")

p_dev = gaussian_pdf(X_dev, mu, var)
p_test = gaussian_pdf(X_test, mu, var)

# Try many thresholds, pick best by F1
best_eps = 0
best_f1 = 0
step_size = (p_dev.max() - p_dev.min()) / 1000

for eps in np.arange(p_dev.min(), p_dev.max(), step_size):
    y_pred = (p_dev < eps).astype(int)
    f1 = f1_score(y_dev, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_eps = eps

print(f"Best ε:  {best_eps:.6f}")
print(f"Best F1: {best_f1:.4f}")

# ── Evaluate on Test Set ──────────────────────────────────────────────

y_test_pred = (p_test < best_eps).astype(int)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
print(f"Test F1: {test_f1:.4f}")

# Visualise
X_all = np.vstack([X_normal, X_anomalies])
plt.figure(figsize=(8, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c="blue", alpha=0.5, label="Normal (train)")
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            c="cyan", alpha=0.7, label="Normal (test)")
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            c="red", alpha=0.7, label="Anomaly (test)")

# Contour of Gaussian
x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = gaussian_pdf(np.c_[xx.ravel(), yy.ravel()], mu, var)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=10, cmap="Blues", alpha=0.4)
plt.colorbar(label="Density")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Anomaly Detection — Gaussian Density")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.show()

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Model P(x) = Π P(xⱼ | μⱼ, σⱼ²) for each feature j          ║
║ • Anomaly if P(x) < ε (very low probability)                  ║
║ • Choose ε on dev set using F1 score                          ║
║ • Assumes features are Gaussian → can transform if needed     ║
║ • Works well when anomalies are rare (e.g. < 5% of data)      ║
║ • Compared to supervised:                                     ║
║   - Anomaly detection: very few positive examples, many types  ║
║   - Supervised: enough positive/negative examples              ║
╚════════════════════════════════════════════════════════════════╝
""")
