"""
Course 2 — Decision Trees & Ensemble Methods
==============================================
Concepts: entropy, information gain, decision trees, random forest, XGBoost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from utils.helpers import plot_decision_boundary

# ── 1. Entropy & Information Gain ─────────────────────────────────────

def entropy(p):
    """Entropy of a binary split: H(p) = -p log2(p) - (1-p) log2(1-p)"""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

p_vals = np.linspace(0, 1, 100)
plt.figure(figsize=(8, 3))
plt.plot(p_vals, [entropy(p) for p in p_vals], "b-", linewidth=2)
plt.axvline(0.5, color="gray", ls="--", alpha=0.5)
plt.xlabel("p (fraction of class 1)")
plt.ylabel("Entropy H(p)")
plt.title("Entropy — measure of impurity (0 = pure, 1 = max impurity)")
plt.grid(alpha=0.3)
plt.show()

# ── 2. Decision Tree Classifier ───────────────────────────────────────

print("── Decision Tree ──")

X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1,
                            random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(f"Train accuracy:  {accuracy_score(y_train, dt.predict(X_train)):.3f}")
print(f"Test accuracy:   {accuracy_score(y_test, y_pred):.3f}")
print(f"Tree depth:      {dt.get_depth()}")
print(f"Leaf nodes:      {dt.get_n_leaves()}")

# Visualise tree
plt.figure(figsize=(14, 6))
plot_tree(dt, filled=True, feature_names=["X1", "X2"],
          class_names=["Class 0", "Class 1"], rounded=True)
plt.title("Decision Tree (max_depth=3)")
plt.show()

plot_decision_boundary(dt, X_test, y_test, title="Decision Tree Boundary")

# ── 3. Random Forest ──────────────────────────────────────────────────

print("\n── Random Forest ──")

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Train accuracy:  {accuracy_score(y_train, rf.predict(X_train)):.3f}")
print(f"Test accuracy:   {accuracy_score(y_test, y_pred_rf):.3f}")

# Feature importance
print(f"Feature importances: {rf.feature_importances_.round(3)}")

plot_decision_boundary(rf, X_test, y_test, title="Random Forest Boundary")

# ── 4. XGBoost ────────────────────────────────────────────────────────

print("\n── XGBoost ──")
try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3,
                                   learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print(f"Test accuracy:   {accuracy_score(y_test, y_pred_xgb):.3f}")
    print(f"Feature importances: {xgb_model.feature_importances_.round(3)}")
except ImportError:
    print("XGBoost not installed. Skipping.")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Decision Tree: split on feature with highest info gain       ║
║ • Entropy: measures impurity (0=pure, 1=evenly mixed)         ║
║ • Information Gain = H(parent) - weighted avg H(children)     ║
║ • max_depth: limits tree size (prevents overfitting)          ║
║ • Random Forest: bagging + random feature subset              ║
║ → reduces variance without increasing bias much               ║
║ • XGBoost: gradient boosting — sequential trees correct       ║
║   errors of previous trees (SOTA for tabular data)            ║
╚════════════════════════════════════════════════════════════════╝
""")
