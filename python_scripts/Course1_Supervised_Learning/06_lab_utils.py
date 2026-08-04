"""
Course 1 — Lab Utilities
==========================
Extra plotting and visualisation helpers used in the course labs.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(cost_history, title="Learning Curve"):
    """Standard cost-vs-iterations plot."""
    plt.figure(figsize=(8, 4))
    plt.plot(cost_history, "b-", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()


def plot_contour(w_vals, b_vals, cost_fn, X, y, w_star=None, b_star=None):
    """3D contour of cost(w, b) — visualise the optimisation landscape."""
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.array([[cost_fn(X, y, np.array([w]), b) for w in w_vals] for b in b_vals])

    plt.figure(figsize=(8, 5))
    plt.contour(W, B, Z, levels=30, cmap="viridis")
    plt.colorbar(label="Cost")
    if w_star is not None and b_star is not None:
        plt.plot(w_star, b_star, "r*", markersize=15, label="Minimum")
    plt.xlabel("w")
    plt.ylabel("b")
    plt.title("Cost Function Contour")
    plt.legend()
    plt.show()
