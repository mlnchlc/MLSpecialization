"""
Course 3 — Recommender Systems
=================================
Concepts: collaborative filtering, content-based filtering, matrix factorisation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. Content-Based Filtering ────────────────────────────────────────
# Predict rating = w·x + b  where x = features of item, w = user preferences

print("── Content-Based Filtering ──")

# Items: [romance, action]  (2 features)
items = np.array([
    [0.9, 0.1],   # Movie A: very romantic
    [0.1, 0.9],   # Movie B: very action
    [0.5, 0.5],   # Movie C: balanced
    [0.8, 0.3],   # Movie D: mostly romantic
    [0.2, 0.7],   # Movie E: mostly action
])
movie_names = ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"]

# User preferences (w)
user_w = np.array([0.8, 0.3])   # likes romance more than action
user_b = 0.0

ratings = items @ user_w + user_b
print("Predicted ratings:")
for name, rating in zip(movie_names, ratings):
    print(f"  {name}: {rating:.2f}")

# ── 2. Collaborative Filtering ────────────────────────────────────────
# Use similarity between users/items to predict missing ratings

print("\n── Collaborative Filtering (Matrix Factorisation) ──")

# Ratings matrix (4 users × 5 movies), 0 = unknown
R = np.array([
    [5, 4, 0, 0, 1],
    [0, 0, 4, 5, 0],
    [3, 0, 0, 2, 4],
    [0, 3, 5, 0, 0],
], dtype=float)

n_users, n_items = R.shape
n_factors = 2   # latent features

# Initialise latent factors
rng = np.random.default_rng(42)
U = rng.normal(0, 0.1, (n_users, n_factors))   # user factors
V = rng.normal(0, 0.1, (n_items, n_factors))   # item factors


def train_collab(R, U, V, alpha=0.02, reg=0.1, n_iters=2000):
    """Alternating least squares on observed ratings only."""
    observed = R > 0
    for epoch in range(n_iters):
        # Predict
        pred = U @ V.T
        error = (observed * (R - pred))

        # Gradients
        dU = -(error @ V) + reg * U
        dV = -(error.T @ U) + reg * V

        U -= alpha * dU
        V -= alpha * dV

        loss = 0.5 * np.sum(observed * (R - U @ V.T) ** 2) + 0.5 * reg * (np.sum(U ** 2) + np.sum(V ** 2))
        if epoch % 500 == 0:
            print(f"  Iter {epoch:4d}  loss={loss:.4f}")

    return U, V


U, V = train_collab(R, U, V)
predicted_ratings = U @ V.T

print("\nOriginal ratings (0 = unknown):")
print(R)
print("\nPredicted ratings (filled):")
print(predicted_ratings.round(2))

# ── 3. Similarity-Based Recommendation ────────────────────────────────

print("\n── Finding Similar Items (Cosine Similarity) ──")

sim_matrix = cosine_similarity(items)
print(f"Similarity matrix:\n{sim_matrix.round(3)}")

# For Movie A (index 0), find most similar
movie_idx = 0
similarities = sim_matrix[movie_idx]
most_similar = np.argsort(similarities)[-3:][::-1]
print(f"\nMost similar to {movie_names[movie_idx]}:")
for idx in most_similar:
    if idx != movie_idx:
        print(f"  {movie_names[idx]}  (similarity: {similarities[idx]:.3f})")

# ── 4. Deep Learning Recommender (Dual/Siamese Network) ─────────────────
# Predict rating using the dot product of user & item embedding networks

print("\n── Deep Learning Recommender ──")
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# Synthetic features:
# Users: [age_scaled, romance_preference, action_preference] (4 users)
user_features = np.array([
    [0.1, 0.9, 0.1],  # User 0: young, likes romance
    [0.8, 0.1, 0.9],  # User 1: older, likes action
    [0.5, 0.5, 0.5],  # User 2: mid-age, balanced
    [0.2, 0.8, 0.2]   # User 3: young, likes romance
], dtype=np.float32)

# Movies: [year_scaled, romance_genre, action_genre] (5 movies)
movie_features = np.array([
    [0.9, 0.9, 0.1],  # Movie A: modern romance
    [0.1, 0.1, 0.9],  # Movie B: classic action
    [0.5, 0.5, 0.5],  # Movie C: balanced
    [0.8, 0.8, 0.2],  # Movie D: modern romance
    [0.2, 0.2, 0.8]   # Movie E: classic action
], dtype=np.float32)

# Generate a synthetic training dataset: user index, movie index, rating
# We train the networks to predict these observed ratings:
user_indices, movie_indices = np.where(R > 0)
y_train = R[R > 0].astype(np.float32)

# Extract features for training pairs
X_user_train = user_features[user_indices]
X_movie_train = movie_features[movie_indices]

# Dual network architecture (functional API)
user_input = tf.keras.layers.Input(shape=(3,), name="user_input")
user_net = tf.keras.layers.Dense(16, activation="relu")(user_input)
user_vec = tf.keras.layers.Dense(8, activation="linear", name="user_embedding")(user_net)

movie_input = tf.keras.layers.Input(shape=(3,), name="movie_input")
movie_net = tf.keras.layers.Dense(16, activation="relu")(movie_input)
movie_vec = tf.keras.layers.Dense(8, activation="linear", name="movie_embedding")(movie_net)

# Dot product of embeddings to predict rating
pred_rating = tf.keras.layers.Dot(axes=1, name="dot_product")([user_vec, movie_vec])

dl_model = tf.keras.Model(inputs=[user_input, movie_input], outputs=pred_rating)
dl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss="mean_squared_error")

# Train
dl_model.fit([X_user_train, X_movie_train], y_train, epochs=200, verbose=0)

# Evaluate / Predict ratings for all user-movie pairs
all_users = np.repeat(np.arange(n_users), n_items)
all_movies = np.tile(np.arange(n_items), n_users)

X_user_all = user_features[all_users]
X_movie_all = movie_features[all_movies]

dl_preds = dl_model.predict([X_user_all, X_movie_all], verbose=0).reshape(n_users, n_items)
print("Predicted ratings using Deep Learning:")
print(dl_preds.round(2))

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • Content-based: predict rating from item features + user      ║
║   preferences → generalises to NEW items                       ║
║ • Collaborative: find latent factors from user-item matrix     ║
║   → generalises to NEW users (but cold-start problem)         ║
║ • Matrix Factorisation: R ≈ U·Vᵀ (user × item factors)        ║
║ • Cosine Similarity: measures item-item closeness             ║
║ • Cold Start: new user/item has no history → use content-based ║
║ • Mean Normalisation: subtract user's mean rating first        ║
║   → helps predict for users who rate everything high/low      ║
║ • Deep Learning: dual networks output user & item embeddings   ║
║   → predicts ratings via the dot product of learned vectors    ║
╚════════════════════════════════════════════════════════════════╝
""")
