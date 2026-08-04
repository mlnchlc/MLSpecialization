"""
Course 1 — Logistic Regression
================================
Concepts: sigmoid, binary cross-entropy loss, decision boundary, multiclass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils.helpers import make_logistic_data, plot_decision_boundary

# ── Sigmoid Function ──────────────────────────────────────────────────

def sigmoid(z):
    """Sigmoid activation: g(z) = 1 / (1 + e^{-z})"""
    return 1 / (1 + np.exp(-z))

# Visualisation
z_vals = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 3))
plt.plot(z_vals, sigmoid(z_vals), "b-", linewidth=2)
plt.axhline(0.5, color="gray", ls="--", alpha=0.5)
plt.axvline(0, color="gray", ls="--", alpha=0.5)
plt.xlabel("z")
plt.ylabel("g(z)")
plt.title("Sigmoid Function — maps any real to (0, 1)")
plt.grid(alpha=0.3)
plt.show()

# ── Binary Classification ─────────────────────────────────────────────

X, y = make_logistic_data(n=200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
print(f"Coefficients:      {model.coef_}")
print(f"Intercept:         {model.intercept_}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])}")

plot_decision_boundary(model, X_test, y_test)

# ── Logistic Regression from Scratch ──────────────────────────────────

print("\n── Logistic Regression from Scratch ──")

def compute_loss(X, y, w, b):
    """Binary cross-entropy loss."""
    m = X.shape[0]
    z = X @ w + b
    f = sigmoid(z)
    return -(1 / m) * np.sum(y * np.log(f + 1e-8) + (1 - y) * np.log(1 - f + 1e-8))


def train_logistic(X, y, alpha=0.1, n_iters=5000):
    """Gradient descent for logistic regression."""
    m, n = X.shape
    w, b = np.zeros(n), 0.0
    losses = []
    for i in range(n_iters):
        z = X @ w + b
        f = sigmoid(z)
        error = f - y
        dw = (1 / m) * (X.T @ error)
        db = (1 / m) * np.sum(error)
        w -= alpha * dw
        b -= alpha * db
        if i % 500 == 0:
            losses.append(compute_loss(X, y, w, b))
    return w, b, losses


w_scratch, b_scratch, losses = train_logistic(X_train, y_train)
# Predict with threshold 0.5
probs = sigmoid(X_test @ w_scratch + b_scratch)
y_pred_scratch = (probs >= 0.5).astype(int)
print(f"Scratch accuracy:  {accuracy_score(y_test, y_pred_scratch):.3f}")

# ── Multiclass (One-vs-Rest) ──────────────────────────────────────────

print("\n── Multiclass (OvR) ──")
from sklearn.datasets import make_classification

X_mc, y_mc = make_classification(n_samples=300, n_features=2,
                                  n_classes=3, n_clusters_per_class=1,
                                  n_redundant=0, random_state=42)
X_mc_train, X_mc_test, y_mc_train, y_mc_test = train_test_split(
    X_mc, y_mc, test_size=0.25, random_state=42
)

ovr = LogisticRegression()
ovr.fit(X_mc_train, y_mc_train)
print(f"OvR accuracy:      {accuracy_score(y_mc_test, ovr.predict(X_mc_test)):.3f}")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Sigmoid converts linear output to probability ∈ (0, 1)       ║
║ • Decision boundary: where P(y=1|x) = 0.5  →  z = 0           ║
║ • Cross-entropy loss penalises confident wrong predictions     ║
║ • Multiclass: One-vs-Rest (OvR) or Softmax (multinomial)       ║
║ • Always evaluate with Accuracy, Precision, Recall, F1         ║
╚════════════════════════════════════════════════════════════════╝
""")