"""
Course 1 — Gradient Descent
============================
Concepts: cost function, batch GD, learning rate, feature scaling, convergence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import make_linear_data

X, y = make_linear_data(w=2.0, b=1.0, n=100, noise=0.3)
m = X.shape[0]

# ── Cost Function (MSE) ────────────────────────────────────────────────

def compute_cost(X, y, w, b):
    """Mean Squared Error cost."""
    m = X.shape[0]
    pred = X @ w + b
    return (1 / (2 * m)) * np.sum((pred - y) ** 2)

# ── Batch Gradient Descent ─────────────────────────────────────────────

def gradient_descent(X, y, w_init, b_init, alpha, n_iters, verbose=True):
    """Batch GD: uses ALL examples per step."""
    m = X.shape[0]
    w, b = w_init.copy(), b_init
    cost_history = []

    for i in range(n_iters):
        pred = X @ w + b
        error = pred - y

        dw = (1 / m) * (X.T @ error)     # (n_features,)  ← vectorized
        db = (1 / m) * np.sum(error)

        w -= alpha * dw
        b -= alpha * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if verbose and i % 100 == 0:
            print(f"  Iter {i:4d}  cost={cost:.6f}")

    return w, b, cost_history


# ── Run GD ─────────────────────────────────────────────────────────────

X_b = np.c_[np.ones(X.shape[0]), X]  # for closed-form comparison
w_closed = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
b_closed, w_closed = w_closed[0], w_closed[1]

# Add bias column for GD too (so w and b are separate)
alpha = 0.01
n_iters = 1000
w_init = np.zeros(X.shape[1])
b_init = 0.0

w_final, b_final, cost_hist = gradient_descent(X, y, w_init, b_init,
                                                alpha, n_iters)

print(f"\nFinal parameters:  w={w_final[0]:.4f},  b={b_final:.4f}")
print(f"Normal eqn check:  w={w_closed:.4f},  b={b_closed:.4f}")

# ── Learning Curve ─────────────────────────────────────────────────────

plt.figure(figsize=(8, 4))
plt.plot(cost_hist, "b-", linewidth=1.5)
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE/2)")
plt.title("Learning Curve — Gradient Descent Convergence")
plt.grid(alpha=0.3)
plt.show()

# ── Effect of Learning Rate ────────────────────────────────────────────

print("\n── Effect of Learning Rate ──")
alphas = [0.001, 0.01, 0.05, 0.1]
plt.figure(figsize=(10, 5))

for a in alphas:
    _, _, ch = gradient_descent(X, y, w_init, b_init, a, 200, verbose=False)
    plt.plot(ch, label=f"α = {a}")

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent — Different Learning Rates")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Batch GD: uses ALL m examples per step — stable but slow     ║
║ • Stochastic GD: uses 1 example per step — fast but noisy      ║
║ • Mini-batch GD (32-256): best of both worlds                  ║
║ • α too small → slow convergence; α too large → may diverge    ║
║ • Feature scaling (StandardScaler) → faster convergence        ║
╚════════════════════════════════════════════════════════════════╝
""")
