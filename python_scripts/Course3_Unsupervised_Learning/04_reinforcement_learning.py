"""
Course 3 — Reinforcement Learning
===================================
Concepts: MDP, state, action, reward, discount factor, value iteration,
Q-learning (tabular), exploration vs exploitation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Markov Decision Process (MDP) Components ───────────────────────
# Small gridworld: 3×3 grid, start=(0,0), goal=(2,2), pit=(1,1)

print("── Gridworld (3×3) ──")

GRID_SIZE = 3
N_STATES = GRID_SIZE * GRID_SIZE
ACTIONS = ["up", "down", "left", "right"]
N_ACTIONS = len(ACTIONS)

GOAL = (2, 2)
PIT = (1, 1)

def state_to_idx(state):
    r, c = state
    return r * GRID_SIZE + c

def idx_to_state(idx):
    return divmod(idx, GRID_SIZE)

# Reward function
def get_reward(state):
    if state == GOAL:
        return 10
    elif state == PIT:
        return -10
    else:
        return -0.1   # small step penalty


# ── 2. Value Iteration ────────────────────────────────────────────────
# Bellman optimality: V*(s) = max_a [ R(s) + γ·Σ P(s'|s,a)·V*(s') ]

print("\n── Value Iteration ──")

gamma = 0.9
V = np.zeros(N_STATES)

def next_state(state, action):
    """Deterministic transition."""
    r, c = state
    if action == "up":    r = max(0, r - 1)
    if action == "down":  r = min(GRID_SIZE - 1, r + 1)
    if action == "left":  c = max(0, c - 1)
    if action == "right": c = min(GRID_SIZE - 1, c + 1)
    return (r, c)


for iteration in range(100):
    delta = 0
    new_V = np.copy(V)
    for s_idx in range(N_STATES):
        state = idx_to_state(s_idx)
        if state == GOAL or state == PIT:
            new_V[s_idx] = get_reward(state)
            continue

        max_value = float("-inf")
        for a in ACTIONS:
            ns = next_state(state, a)
            ns_idx = state_to_idx(ns)
            value = get_reward(ns) + gamma * V[ns_idx]
            max_value = max(max_value, value)

        new_V[s_idx] = max_value
        delta = max(delta, abs(new_V[s_idx] - V[s_idx]))

    V = new_V
    if delta < 1e-6:
        print(f"  Converged at iteration {iteration + 1}")
        break

# Display value function
print("\nOptimal Value Function:")
print(V.reshape(GRID_SIZE, GRID_SIZE).round(1))

# Derived policy
def get_policy(state):
    """Greedy policy from optimal V."""
    if state == GOAL or state == PIT:
        return "stay"
    best_a, best_v = None, float("-inf")
    for a in ACTIONS:
        ns = next_state(state, a)
        ns_idx = state_to_idx(ns)
        v = get_reward(ns) + gamma * V[ns_idx]
        if v > best_v:
            best_v = v
            best_a = a
    return best_a

policy = np.array([get_policy(idx_to_state(s)) for s in range(N_STATES)])
print("\nOptimal Policy:")
print(policy.reshape(GRID_SIZE, GRID_SIZE))

# ── 3. Q-Learning (tabular) ───────────────────────────────────────────

print("\n── Q-Learning ──")

rng = np.random.default_rng(42)
Q = np.zeros((N_STATES, N_ACTIONS))
alpha = 0.5      # learning rate
gamma = 0.9
epsilon = 0.2    # exploration rate
n_episodes = 500

episode_rewards = []
state = (0, 0)

for ep in range(n_episodes):
    total_reward = 0
    state = (0, 0)
    done = False

    while not done:
        s_idx = state_to_idx(state)

        # ε-greedy action selection
        if rng.random() < epsilon:
            a_idx = rng.integers(N_ACTIONS)
        else:
            a_idx = np.argmax(Q[s_idx])

        action = ACTIONS[a_idx]
        next_s = next_state(state, action)
        ns_idx = state_to_idx(next_s)
        reward = get_reward(next_s)

        # Q-learning update
        Q[s_idx, a_idx] += alpha * (
            reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, a_idx]
        )

        total_reward += reward
        state = next_s

        if state == GOAL or state == PIT:
            done = True

    episode_rewards.append(total_reward)

# Plot learning progress
plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.5, label="Per episode")
# Smooth
window = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
plt.plot(range(window - 1, n_episodes), smoothed, "r-", linewidth=2, label=f"Moving avg ({window})")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning — Learning Progress")
plt.legend()
plt.grid(alpha=0.3)

# Learned policy from Q
q_policy = np.array([ACTIONS[np.argmax(Q[s])] for s in range(N_STATES)])

plt.subplot(1, 2, 2)
plt.imshow(Q, aspect="auto", cmap="viridis")
plt.xlabel("Actions")
plt.ylabel("States")
plt.title("Q-Table")
plt.yticks(range(N_STATES), [f"({r},{c})" for r in range(GRID_SIZE) for c in range(GRID_SIZE)])
plt.xticks(range(N_ACTIONS), ACTIONS)
plt.colorbar()
plt.tight_layout()
plt.show()

print("Learned Q-policy:")
print(q_policy.reshape(GRID_SIZE, GRID_SIZE))

# ── 4. Deep Q-Network (DQN) for Continuous State Space ────────────────
# Approximates continuous state Q-values with target network & experience replay

print("\n── Deep Q-Network (DQN) ──")
import tensorflow as tf
from collections import deque

# Set random seed for reproducibility
tf.random.set_seed(42)
rng = np.random.default_rng(42)

class ContinuousEnv:
    """1D tracking environment with continuous state [position, velocity]."""
    def __init__(self):
        self.state = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        self.state[0] = rng.uniform(-0.5, 0.5)
        self.state[1] = rng.uniform(-0.1, 0.1)
        return self.state.copy()
        
    def step(self, action):
        # Actions: 0 = accelerate left, 1 = stay, 2 = accelerate right
        a = -0.05 if action == 0 else (0.05 if action == 2 else 0.0)
        v = np.clip(self.state[1] + a, -0.2, 0.2)
        x = np.clip(self.state[0] + v, -1.0, 1.0)
        self.state = np.array([x, v], dtype=np.float32)
        
        # Reward
        reward = 1.0 - 5.0 * (x ** 2) - 2.0 * (v ** 2)
        done = False
        if abs(x) > 0.8:
            done = True
            reward = -10.0
        elif abs(x) < 0.05 and abs(v) < 0.05:
            done = True
            reward = 10.0
            
        return self.state.copy(), reward, done

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

def build_q_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    return model

env = ContinuousEnv()
buffer = ReplayBuffer(capacity=2000)

q_net = build_q_network()
target_net = build_q_network()
target_net.set_weights(q_net.get_weights())

# DQN Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.985
epsilon_min = 0.1
batch_size = 32
tau = 0.1  # Soft update parameter

n_episodes = 100
dqn_rewards = []

for ep in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action
        if rng.random() < epsilon:
            action = rng.integers(3)
        else:
            q_values = q_net(state[np.newaxis, :]).numpy()
            action = np.argmax(q_values[0])
            
        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Train step if enough samples in buffer
        if len(buffer.buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            # Predict Q(s_next, a) using target network
            target_q_next = target_net(next_states).numpy()
            max_target_q = np.max(target_q_next, axis=1)
            
            # Target Q-value
            targets = rewards + gamma * max_target_q * (1.0 - dones)
            
            # Update targets in predicted Q-values
            q_targets = q_net(states).numpy()
            q_targets[np.arange(batch_size), actions] = targets
            
            # Train Q-Network
            q_net.train_on_batch(states, q_targets)
            
            # Soft update of target network
            q_weights = q_net.get_weights()
            t_weights = target_net.get_weights()
            new_weights = []
            for qw, tw in zip(q_weights, t_weights):
                new_weights.append(tau * qw + (1.0 - tau) * tw)
            target_net.set_weights(new_weights)
            
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    dqn_rewards.append(total_reward)
    if (ep + 1) % 20 == 0:
        print(f"  Episode {ep+1:3d}  Total Reward: {total_reward:6.1f}  Epsilon: {epsilon:.3f}")

print(f"DQN training finished. Average reward of last 10 episodes: {np.mean(dqn_rewards[-10:]):.2f}")

# ── Key Intuitions ─────────────────────────────────────────────────────
print("""
╔══ Key Takeaways ───────────────────────────────────────────────╗
║ • MDP: (S, A, P, R, γ) — State, Action, Transition, Reward,  ║
║   Discount factor                                              ║
║ • Bellman equation: V*(s) = max_a [R(s) + γ·V*(s')]          ║
║ • Value Iteration: repeatedly update V until convergence       ║
║ → gives optimal value function, then derive policy greedily   ║
║ • Q-Learning: model-free — learns state-action values directly ║
║ • Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',a') - Q(s,a)]         ║
║ • ε-greedy: explore with prob ε, exploit with prob 1-ε        ║
║ • γ (discount): near 0 = myopic, near 1 = far-sighted         ║
║ • DQN: Q(s,a) approximated by NN for continuous state spaces ║
║   → stabilized using target networks & experience replay.   ║
╚════════════════════════════════════════════════════════════════╝
""")
