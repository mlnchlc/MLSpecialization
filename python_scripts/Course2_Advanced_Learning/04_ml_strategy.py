"""
Course 2 — ML Strategy & Best Practices
=========================================
Concepts: train/dev/test splits, error analysis, evaluation metrics,
class imbalance, data augmentation, transfer learning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# ── 1. Train / Dev / Test Splits ──────────────────────────────────────

print("── Data Splitting Strategy ──")

X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

# 60/20/20 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train set: {X_train.shape[0]:>6} samples")
print(f"Dev set:   {X_dev.shape[0]:>6} samples")
print(f"Test set:  {X_test.shape[0]:>6} samples")

# ── 2. Evaluation Metrics ─────────────────────────────────────────────

print("\n── Classification Metrics ──")

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_dev)
y_prob = model.predict_proba(X_dev)[:, 1]

print(f"Accuracy:  {accuracy_score(y_dev, y_pred):.3f}")
print(f"Precision: {precision_score(y_dev, y_pred):.3f}")
print(f"Recall:    {recall_score(y_dev, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_dev, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_dev, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot()
plt.title("Confusion Matrix — Dev Set")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_dev, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ── 3. Cross-Validation ───────────────────────────────────────────────

print("\n── Cross-Validation (5-fold) ──")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV scores:        {cv_scores.round(3)}")
print(f"Mean ± Std:       {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 4. Error Analysis Template ────────────────────────────────────────

print("\n── Error Analysis Checklist ──")
print("""
When your model makes errors on the dev set:

1. Group errors by category (e.g., blurry image, occluded, low lighting)
2. Count how many fall into each category
3. Prioritise the category with the most errors

Example table:
┌──────────────────────┬───────┬────────────┐
│ Error Category       │ Count │ % of Total │
├──────────────────────┼───────┼────────────┤
│ Blurry image         │    12 │        40% │ ← START HERE
│ Occluded object      │     8 │        27% │
│ Low lighting         │     7 │        23% │
│ Other                │     3 │        10% │
└──────────────────────┴───────┴────────────┘
""")

# ── 5. Class Imbalance Demo ───────────────────────────────────────────

print("── Class Imbalance ──")

X_imb, y_imb = make_classification(
    n_samples=1000, weights=[0.9, 0.1], random_state=42
)
print(f"Class distribution: {np.bincount(y_imb)}  ({(y_imb == 1).mean():.1%} positive)")

X_imb_train, X_imb_test, y_imb_train, y_imb_test = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42
)

imb_model = RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                    random_state=42)
imb_model.fit(X_imb_train, y_imb_train)
y_imb_pred = imb_model.predict(X_imb_test)

print(f"Accuracy:  {accuracy_score(y_imb_test, y_imb_pred):.3f}")
print(f"Precision: {precision_score(y_imb_test, y_imb_pred):.3f}")
print(f"Recall:    {recall_score(y_imb_test, y_imb_pred):.3f}")
print(f"F1 Score:  {f1_score(y_imb_test, y_imb_pred):.3f}")
print("  → For imbalanced data, Accuracy is misleading!")
print("  → Use Precision / Recall / F1 instead.")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Train/Dev/Test: 60/20/20 or 98/1/1 for big data            ║
║ • Dev set = iterate & tune hyperparams                       ║
║ • Test set = final evaluation ONLY (peeking leads to bias)    ║
║ • Human-level performance ≈ Bayes error (upper bound)        ║
║ • Avoidable bias  = train error - human error                ║
║ • Variance        = dev error - train error                  ║
║ • For imbalanced data: use Precision, Recall, F1, ROC-AUC    ║
║ • class_weight='balanced' auto-adjusts for imbalance         ║
╚════════════════════════════════════════════════════════════════╝
""")
