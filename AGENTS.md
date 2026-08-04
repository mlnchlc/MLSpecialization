# 🤖 ML Specialization — Workspace Rules for AI Agents

These guidelines are tailored to the `MLSpecialization` project. Every agent modifying code
or adding reference scripts **must** adhere to the rules below.

---

## 1. Script Self-Containment

* **Run out-of-the-box:** Every script in a course folder (e.g., `Course1_Supervised_Learning/`,
  `Course2_Advanced_Learning/`, `Course3_Unsupervised_Learning/`) must be executable on its own
  with zero setup beyond `pip install -r requirements.txt`.
* **Console & Visual Outputs:** Running a script should print formatted outputs to the console
  **and** display visualisations via `matplotlib.pyplot.show()`.
* **Imports:** Always insert the repository root into `sys.path` **before** importing from
  `utils`, like so:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```
  Add this block even if the script currently imports nothing from `utils` — future edits might.
* **`matplotlib.pyplot`:** Import locally inside plotting functions where feasible, or at the top
  of the script. Always call `plt.show()` so figures appear when run non-interactively.

---

## 2. Standardized Layout & Takeaways

* **Header Docstring:** Every script starts with a triple-quoted docstring:
  ```python
  """
  Course N — Title
  =================
  Concepts: comma-separated list.
  """
  ```
* **Key Takeaways Box:** Every script **must** end with a Unicode double-line box:
  ```python
  print("""
  ╔══ Key Takeaways ───────────────────────────────────────────────╗
  ║ • Point 1                                                      ║
  ║ • Point 2                                                      ║
  ╚════════════════════════════════════════════════════════════════╝
  """)
  ```
  Keep points concise (one line each, ≤ 72 characters).

---

## 3. Reproducibility (Determinism)

* **NumPy random:** NEVER use `np.random.rand()`, `np.random.seed()`, or other bare-module
  functions. Always create a **local generator**:
  ```python
  rng = np.random.default_rng(42)
  ```
  Pass it as an explicit argument when possible.
* **scikit-learn:** Always set `random_state=42` on every estimator, splitter, or transformer
  that accepts it (e.g., `RandomForestClassifier`, `KMeans`, `train_test_split`).
* **TensorFlow:** Set `tf.random.set_seed(42)` at the top of any script using TF.

---

## 4. Code Style & Quality

* **Python version:** Target **3.10+**. Use `from __future__ import annotations` in shared
  modules for cleaner type hints.
* **Type hints:** All public functions in `utils/` must have typed signatures
  (`def foo(x: NDArray, y: int) -> float:`). Inside course scripts, type hints are optional
  but encouraged for new functions.
* **Imports:** Group in order: stdlib → third-party → local (`utils`). Use `isort` or the
  `ruff` formatter (see `pyproject.toml`).
* **Line length:** Aim for ≤ 100 characters.
* **f-strings:** Prefer f-strings over `.format()` or `%` for string formatting.

---

## 5. File Structure & Naming

```
📁 Course{N}_{Topic}/      # Interactive notebooks (.ipynb)
├── NN_short_topic.ipynb

📁 python_scripts/         # Standalone executable scripts (.py)
└── 📁 Course{N}_{Topic}/
    ├── __init__.py          # short docstring describing the course
    ├── NN_short_topic.py    # two-digit numeric prefix + snake_case name
    └── ...

📁 utils/
├── __init__.py          # re-exports public symbols from helpers
└── helpers.py           # shared data generators + plotting helpers
```

* Add a one-line `__init__.py` to every new course directory.
* Utility code shared across ≥ 2 scripts belongs in `utils/helpers.py`.
* Course-specific helpers live alongside the scripts (e.g., `06_lab_utils.py`).

---

## 6. `utils/helpers.py` Conventions

| Function | Returns | Used By |
|----------|---------|---------|
| `make_linear_data(w, b, n, noise, seed)` | `(X, y)` | Linear regression, GD |
| `make_logistic_data(n, seed)` | `(X, y)` | Logistic regression |
| `make_blobs(n, centers, seed)` | `(X, y)` | K-Means |
| `plot_regression_line(X, y, w, b, title)` | `None` | Linear regression |
| `plot_decision_boundary(model, X, y, title)` | `None` | Logistic regression, Trees, NN |

When adding new shared utilities, add the function to `helpers.py`, re-export it in
`__init__.py`, and update this table.

---

## 7. Documentation Consistency

* Keep the **directory map** and **course summary table** in `README.md` up-to-date.
* When adding a new script, update both the ASCII folder tree and the table.
* When renaming or removing a file, update `README.md` and this file simultaneously.
* `walkthrough.md` records historical compliance changes — append new entries for major
  refactors.

---

## 8. Script Checklist (for new scripts)

- [ ] Header docstring with course number, title, and concepts
- [ ] `sys.path.insert(0, ...)` before any `from utils import ...`
- [ ] All random state controlled via `default_rng(seed)` / `random_state=42`
- [ ] At least one visualisation with `plt.show()`
- [ ] Console output with formatted results
- [ ] Key Takeaways box at the bottom
- [ ] Script runs successfully end-to-end
