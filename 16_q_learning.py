from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

size = 5
goal = (4, 4)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
q = np.zeros((size, size, len(actions)))
episode_rewards = []

for episode in range(500):
    state = (0, 0)
    total = 0
    epsilon = max(0.05, 0.8 * (0.995 ** episode))
    for _ in range(60):
        if rng.random() < epsilon:
            action = rng.integers(len(actions))
        else:
            action = int(np.argmax(q[state]))
        move = actions[action]
        next_state = (
            int(np.clip(state[0] + move[0], 0, size - 1)),
            int(np.clip(state[1] + move[1], 0, size - 1)),
        )
        reward = 10 if next_state == goal else -0.1
        q[state][action] += 0.2 * (reward + 0.95 * np.max(q[next_state]) - q[state][action])
        state = next_state
        total += reward
        if state == goal:
            break
    episode_rewards.append(total)

policy = np.argmax(q, axis=2)
symbols = np.array(["^", "v", "<", ">"])
print("Q-Learning greedy policy:")
for row in range(size):
    print(" ".join("G" if (row, col) == goal else symbols[policy[row, col]] for col in range(size)))

plt.figure(figsize=(7, 4))
plt.plot(np.convolve(episode_rewards, np.ones(20) / 20, mode="valid"))
plt.title("Q-Learning Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(OUT / "16_q_learning.png", dpi=150)
plt.show()

