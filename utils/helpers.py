"""
Shared utilities for ML Specialization quick-reference scripts.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.colors import ListedColormap

# Re-export commonly-used types for downstream convenience
NDArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


# ── Synthetic Data Generators ──────────────────────────────────────────

def make_linear_data(
    w: float = 2.0,
    b: float = 1.0,
    n: int = 100,
    noise: float = 0.3,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Generate y = w*x + b + Gaussian noise.

    Args:
        w: Slope of the true linear relationship.
        b: Intercept of the true linear relationship.
        n: Number of samples to generate.
        noise: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        X: Feature array of shape (n, 1).
        y: Target array of shape (n,).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5, size=n)
    y = w * X + b + rng.normal(0, noise, size=n)
    return X.reshape(-1, 1), y


def make_logistic_data(
    n: int = 200,
    seed: int = 42,
) -> Tuple[NDArray, NDArray]:
    """Two-blob classification dataset.

    Args:
        n: Total number of samples (even split between two classes).
        seed: Random seed for reproducibility.

    Returns:
        X: Feature array of shape (n, 2).
        y: Binary label array of shape (n,).
    """
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal([2, 2], [[1, 0], [0, 1]], n // 2)
    X = np.vstack([X, rng.multivariate_normal([5, 5], [[1, 0], [0, 1]], n // 2)])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def make_blobs(
    n: int = 300,
    centers: int = 3,
    seed: int = 42,
) -> Tuple[NDArray, IntArray]:
    """Multi-blob clustering dataset.

    Args:
        n: Total number of samples (evenly split across centers).
        centers: Number of distinct blobs.
        seed: Random seed for reproducibility.

    Returns:
        X: Feature array of shape (n, 2).
        y: Cluster-index label array of shape (n,).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2))
    y = np.zeros(n, dtype=int)
    pts_per_center = n // centers
    offsets = [(2, 2), (8, 2), (5, 8)]
    for i in range(centers):
        start, end = i * pts_per_center, (i + 1) * pts_per_center
        X[start:end] = rng.multivariate_normal(
            offsets[i], [[0.4, 0], [0, 0.4]], pts_per_center
        )
        y[start:end] = i
    return X, y


# ── Plotting Helpers ───────────────────────────────────────────────────

def plot_regression_line(
    X: NDArray,
    y: NDArray,
    w: float,
    b: float,
    title: str = "Linear Regression",
) -> None:
    """Scatter-plot data points and overlay the fitted regression line.

    Args:
        X: Feature array (n, 1) — must be 2-D for sklearn compatibility.
        y: Target array (n,).
        w: Slope of the fitted line.
        b: Intercept of the fitted line.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, alpha=0.6, label="Data")
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, "r-", linewidth=2, label=f"y = {w:.2f}x + {b:.2f}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_decision_boundary(
    model: Any,
    X: NDArray,
    y: NDArray,
    title: str = "Decision Boundary",
) -> None:
    """Plot 2D data points and overlay a classifier's decision boundary.

    Args:
        model: Fitted classifier with a ``predict`` method.
        X: Feature array of shape (n, 2).
        y: Label array of shape (n,).
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00AA00", "#0000FF"])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=40)
    plt.title(title)
    plt.show()
