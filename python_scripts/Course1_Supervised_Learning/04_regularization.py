"""
Course 1 — Regularization
===========================
Concepts: overfitting, L2 (Ridge), L1 (Lasso), ElasticNet, tuning λ.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ── High-degree polynomial (prone to overfitting) ─────────────────────

rng = np.random.default_rng(42)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + rng.normal(0, 0.15, size=20)
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Visualise different degrees
degrees = [1, 3, 9, 15]
plt.figure(figsize=(12, 3.5))
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

for i, deg in enumerate(degrees):
    ax = plt.subplot(1, 4, i + 1)
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_plot)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))

    ax.scatter(X_train, y_train, c="b", s=20, label="Train")
    ax.scatter(X_test, y_test, c="orange", s=20, label="Test")
    ax.plot(X_plot, y_pred, "r-", linewidth=1.5)
    ax.set_title(f"deg={deg}\nTrain MSE={train_mse:.3f}, Test MSE={test_mse:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# ── Ridge vs Lasso vs ElasticNet ──────────────────────────────────────

print("── Regularisation Comparison (degree=15, λ=0.01) ──")

deg = 15
poly = PolynomialFeatures(deg)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

models = {
    "Linear (no reg)": LinearRegression(),
    "Ridge (L2)":      Ridge(alpha=0.01),
    "Lasso (L1)":      Lasso(alpha=0.01),
    "ElasticNet":      ElasticNet(alpha=0.01, l1_ratio=0.5),
}

for name, mdl in models.items():
    mdl.fit(X_train_poly, y_train)
    train_mse = mean_squared_error(y_train, mdl.predict(X_train_poly))
    test_mse  = mean_squared_error(y_test, mdl.predict(X_test_poly))
    nz = np.sum(np.abs(mdl.coef_) > 1e-6)
    print(f"  {name:20s}  Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, "
          f"Non-zero coefs={nz}")

# ── Effect of λ (Regularization Strength) ────────────────────────────

print("\n── Effect of λ (alpha) on Ridge ──")
lambdas = np.logspace(-3, 2, 20)
train_errs, test_errs = [], []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train_poly, y_train)
    train_errs.append(mean_squared_error(y_train, ridge.predict(X_train_poly)))
    test_errs.append(mean_squared_error(y_test, ridge.predict(X_test_poly)))

plt.figure(figsize=(8, 4))
plt.semilogx(lambdas, train_errs, "b-o", label="Train MSE")
plt.semilogx(lambdas, test_errs, "r-o", label="Test MSE")
plt.xlabel("λ (alpha)")
plt.ylabel("MSE")
plt.title("Ridge Regularisation — λ vs Error")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

best_lam = lambdas[np.argmin(test_errs)]
print(f"  Best λ (lowest test MSE): {best_lam:.4f}")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Overfitting: low train error, high test error                ║
║ • L2 (Ridge):    adds λ·||w||² — shrinks weights, no zeros     ║
║ • L1 (Lasso):    adds λ·||w||₁ — drives some weights to 0      ║
║ • ElasticNet:    blends L1 + L2 (l1_ratio controls mix)        ║
║ • λ↑ → more regularization (underfitting risk)                 ║
║ • λ↓ → less regularization (overfitting risk)                  ║
║ • Choose λ via cross-validation on validation set              ║
╚════════════════════════════════════════════════════════════════╝
""")
