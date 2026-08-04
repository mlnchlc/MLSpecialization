"""
Course 1 — Neural Networks (from scratch)
===========================================
Concepts: forward propagation, activation functions, layer structure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Activation Functions ──────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

# Visualise activations
z = np.linspace(-5, 5, 200)
plt.figure(figsize=(10, 3))
for i, (fn, name) in enumerate([(sigmoid, "Sigmoid"), (tanh, "Tanh"), (relu, "ReLU")]):
    plt.subplot(1, 3, i + 1)
    plt.plot(z, fn(z), linewidth=2)
    plt.axhline(0, color="gray", ls="--", alpha=0.4)
    plt.title(name)
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 2-Layer Neural Network (forward + backward from scratch) ──────────

print("── Neural Network from Scratch ──")

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalise
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


def initialize_parameters(n_x, n_h, n_y):
    """He initialisation for ReLU, small random for output."""
    rng = np.random.default_rng(42)
    W1 = rng.normal(0, np.sqrt(2 / n_x), (n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = rng.normal(0, 0.5, (n_y, n_h))
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_prop(X, params):
    """Forward pass through 2-layer network."""
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = W1 @ X + b1      # (n_h, m)
    A1 = np.tanh(Z1)      # hidden activation
    Z2 = W2 @ A1 + b2     # (1, m)
    A2 = sigmoid(Z2)      # output activation
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def compute_cost(A2, Y):
    """Binary cross-entropy."""
    m = Y.shape[1]
    return -(1 / m) * np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))


def backward_prop(X, Y, params, cache):
    """Backpropagation for 2-layer network."""
    m = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y                                    # (1, m)
    dW2 = (1 / m) * (dZ2 @ A1.T)                    # (1, n_h)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2                                # (n_h, m)
    dZ1 = dA1 * (1 - np.power(A1, 2))               # tanh derivative
    dW1 = (1 / m) * (dZ1 @ X.T)                     # (n_h, n_x)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def train(X, Y, n_h=10, alpha=0.5, n_iters=3000):
    """Train 2-layer NN."""
    n_x, n_y = X.shape[0], Y.shape[0]
    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(n_iters):
        A2, cache = forward_prop(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_prop(X, Y, params, cache)

        params["W1"] -= alpha * grads["dW1"]
        params["b1"] -= alpha * grads["db1"]
        params["W2"] -= alpha * grads["dW2"]
        params["b2"] -= alpha * grads["db2"]

        if i % 500 == 0:
            print(f"  Iter {i:4d}  cost={cost:.6f}")

    return params


# Transpose for (features, samples) format
X_train_t = X_train.T
y_train_t = y_train.T.reshape(1, -1)
X_test_t = X_test.T

params = train(X_train_t, y_train_t, n_h=10, alpha=0.5, n_iters=3000)

# Evaluate
A2_train, _ = forward_prop(X_train_t, params)
A2_test, _ = forward_prop(X_test_t, params)
train_acc = accuracy_score(y_train, (A2_train.T >= 0.5).astype(int))
test_acc = accuracy_score(y_test, (A2_test.T >= 0.5).astype(int))
print(f"\nTrain accuracy:  {train_acc:.3f}")
print(f"Test accuracy:   {test_acc:.3f}")

# ── Visualize decision boundary ───────────────────────────────────────

from utils.helpers import plot_decision_boundary


class NNSklearnWrapper:
    """Wrap our NN in sklearn-like interface for the plotting helper."""
    def __init__(self, params):
        self.params = params

    def predict(self, X):
        A2, _ = forward_prop(X.T, self.params)
        return (A2.T >= 0.5).astype(int).ravel()


plot_decision_boundary(NNSklearnWrapper(params), X_test, y_test.ravel(),
                       title="Neural Network Decision Boundary")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Forward prop:  Z[l] = W[l]·A[l-1] + b[l],  A[l] = g(Z[l])  ║
║ • Hidden layers learn progressively complex features          ║
║ • ReLU → hidden layers (avoids vanishing gradient)            ║
║ • Sigmoid → binary output (0-1 probability)                   ║
║ • More layers = deeper network = learns more complex funcs    ║
╚════════════════════════════════════════════════════════════════╝
""")
