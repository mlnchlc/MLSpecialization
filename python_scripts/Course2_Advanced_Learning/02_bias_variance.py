"""
Course 2 — Bias / Variance & Learning Curves
==============================================
Concepts: high bias vs high variance, learning curves, cross-validation,
regularisation to diagnose & fix model performance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── Learning Curve (training set size vs error) ───────────────────────

print("── Learning Curves ──")

train_sizes, train_scores, val_scores = learning_curve(
    SVC(kernel="rbf", gamma=0.1),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    random_state=42,
)

train_mean = np.mean(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_mean, "o-", label="Train accuracy")
plt.plot(train_sizes, val_mean, "o-", label="Validation accuracy")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve — RBF Kernel SVM")
plt.legend()
plt.grid(alpha=0.3)

# ── Validation Curve (hyperparameter vs error) ────────────────────────

print("── Validation Curve (gamma in RBF SVM) ──")

param_range = np.logspace(-3, 2, 20)
train_scores, val_scores = validation_curve(
    SVC(kernel="rbf"),
    X_train, y_train,
    param_name="gamma",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
)

train_mean = np.mean(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)

plt.subplot(1, 2, 2)
plt.semilogx(param_range, train_mean, "o-", label="Train accuracy")
plt.semilogx(param_range, val_mean, "o-", label="Validation accuracy")
plt.xlabel("gamma (RBF kernel)")
plt.ylabel("Accuracy")
plt.title("Validation Curve — Bias/Variance Diagnosis")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── Diagnosis ─────────────────────────────────────────────────────────

best_gamma = param_range[np.argmax(val_mean)]
print(f"Best gamma: {best_gamma:.4f}")

print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • High Bias (Underfitting): both train & val accuracy are low  ║
║   → Solutions: bigger model, add features, decrease L2/L1 reg  ║
║ • High Variance (Overfitting): high train & low val accuracy   ║
║   → Solutions: more data, add regularization, simplify model   ║
║ • Learning Curves: plot training size vs. performance          ║
║   → High Bias: train & val plateau early close to each other   ║
║   → High Variance: gap remains between train & val curves      ║
║ • Validation Curves: sweep hyperparameter (e.g. λ, γ, degree)  ║
║   → Pick parameter value that maximizes validation accuracy    ║
╚════════════════════════════════════════════════════════════════╝
""")
