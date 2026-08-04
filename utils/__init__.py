"""
ML Specialization — Shared Utilities.

Provides synthetic data generators and plotting helpers used across
all three course reference scripts.
"""

from .helpers import (
    make_linear_data,
    make_logistic_data,
    make_blobs,
    plot_regression_line,
    plot_decision_boundary,
)

__all__ = [
    "make_linear_data",
    "make_logistic_data",
    "make_blobs",
    "plot_regression_line",
    "plot_decision_boundary",
]
