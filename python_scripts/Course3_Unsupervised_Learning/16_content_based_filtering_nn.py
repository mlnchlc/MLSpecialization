"""
Course 3 — Content-Based Filtering with Two-Tower Neural Networks
===================================================================
Concepts: Two-Tower architecture, user & item embeddings, dot product prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)

# ── 1. User & Item Content Features ──────────────────────────────────
# 100 ratings samples, user features (10 dim), item features (8 dim)
n_samples = 200
rng = np.random.default_rng(42)

X_user = rng.uniform(0, 1, size=(n_samples, 10))
X_item = rng.uniform(0, 1, size=(n_samples, 8))
y_ratings = np.sum(X_user[:, :3] * X_item[:, :3], axis=1) + rng.normal(0, 0.1, size=n_samples)

# ── 2. Building Two-Tower Architecture in Keras ───────────────────────
embedding_dim = 16

# User Tower
user_input = tf.keras.layers.Input(shape=(10,), name="user_input")
u_dense1 = tf.keras.layers.Dense(32, activation="relu")(user_input)
user_vec = tf.keras.layers.Dense(embedding_dim, name="user_vector")(u_dense1)

# Item Tower
item_input = tf.keras.layers.Input(shape=(8,), name="item_input")
i_dense1 = tf.keras.layers.Dense(32, activation="relu")(item_input)
item_vec = tf.keras.layers.Dense(embedding_dim, name="item_vector")(i_dense1)

# Dot product prediction
output = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])

two_tower_model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
two_tower_model.compile(optimizer="adam", loss="mse")

history = two_tower_model.fit([X_user, X_item], y_ratings, epochs=40, verbose=0)
print(f"Two-Tower Model Final Loss (MSE): {history.history['loss'][-1]:.4f}")

# ── 3. Predictions Demonstration ─────────────────────────────────────
pred_ratings = two_tower_model.predict([X_user[:5], X_item[:5]], verbose=0)
print("\nSample Predicted Ratings vs True Ratings:")
for i in range(5):
    print(f"  Sample {i+1}: Predicted = {pred_ratings[i][0]:.2f}, True = {y_ratings[i]:.2f}")

# ── Key Takeaways ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Two-Tower Neural Net: separate networks for user and item    ║
║ • Outputs fixed-dimensional embedding vectors v_u and v_m       ║
║ • Rating prediction = dot product v_u · v_m                     ║
║ • Handles NEW items/users by feeding their content features   ║
╚════════════════════════════════════════════════════════════════╝
""")
