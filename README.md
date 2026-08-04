# 🧠 Andrew Ng's Machine Learning Specialization — Quick Reference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/deps-numpy%20%3C2-blueviolet)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/deps-scikit--learn-orange)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/deps-TensorFlow-FF6F00)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/deps-XGBoost-brightgreen)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A glanceable, self-contained code reference for all 3 courses in **[Andrew Ng's Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)** on Coursera.

Each script is **runnable out-of-the-box** — run it to see console output and interactive visualisations. Perfect for revision, interview prep, or quickly recalling key algorithms.

---

## 📂 Structure

```
📁 Course1_Supervised_Learning/         # Interactive Jupyter Notebooks (.ipynb)
📁 Course2_Advanced_Learning/
📁 Course3_Unsupervised_Learning/

📁 python_scripts/                        # Standalone Python scripts (.py)
├── 📁 Course1_Supervised_Learning/
│   ├── 01_linear_regression.py
│   ├── 02_gradient_descent.py
│   ├── 03_logistic_regression.py
│   ├── 04_regularization.py
│   ├── 05_neural_networks.py
│   └── 06_lab_utils.py
├── 📁 Course2_Advanced_Learning/
│   ├── 01_tensorflow_intro.py
│   ├── 02_bias_variance.py
│   ├── 03_decision_trees.py
│   └── 04_ml_strategy.py
└── 📁 Course3_Unsupervised_Learning/
    ├── 01_kmeans_clustering.py
    ├── 02_anomaly_detection.py
    ├── 03_recommender_systems.py
    ├── 04_reinforcement_learning.py
    ├── 05_pca.py
    ├── 12_kmeans_from_scratch.py
    ├── 13_anomaly_detection_gaussian.py
    ├── 14_pca.py
    ├── 15_collaborative_filtering.py
    ├── 16_content_based_filtering_nn.py
    └── 17_reinforcement_learning_dqn.py

📁 utils/
├── __init__.py
└── helpers.py                           # Shared utilities, data generators
```

## 🚀 Quick Start

```bash
# 1. Clone & enter
git clone <repo-url> && cd MLSpecialization

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run any standalone script
python python_scripts/Course1_Supervised_Learning/01_linear_regression.py
python python_scripts/Course2_Advanced_Learning/01_tensorflow_intro.py
python python_scripts/Course3_Unsupervised_Learning/01_kmeans_clustering.py
```

Each script prints formatted **Key Takeaways** to the console and displays
**Matplotlib visualisations** — no Jupyter notebook required.

---

## 📖 How to Use This Repo

| Goal | Action |
|------|--------|
| **Revise a specific topic** | Open the corresponding `.py` script and read the docstring + takeaways |
| **Run & explore** | Execute the script, tweak parameters, re-run |
| **Steal the code** | Each function is self-contained; copy into your own projects |
| **Compare implementations** | "From scratch" vs sklearn/TensorFlow side-by-side |

---

## ✅ Coding Standards

Every script adheres to these rules (see [AGENTS.md](AGENTS.md)):

1. **Self-contained** — `sys.path` insert at the top, all imports local
2. **Reproducible** — `np.random.default_rng(seed)` everywhere, `random_state=42`
3. **Key Takeaways box** — double-line Unicode box at the end of every script
4. **Visual output** — `matplotlib.pyplot.show()` called for each figure

---

## 📚 Courses Covered

| # | Course | Topics |
|---|--------|--------|
| 1 | Supervised Learning | Linear/Logistic Regression, Gradient Descent, Regularization, Neural Nets |
| 2 | Advanced Algorithms | TensorFlow, Bias/Variance, Decision Trees, ML Strategy |
| 3 | Unsupervised Learning | K-Means, Anomaly Detection, Recommender Systems, RL, PCA |

---

## 📜 License

This project is licensed under the **[MIT License](LICENSE)** — see the [`LICENSE`](LICENSE) file for details.

