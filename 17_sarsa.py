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
rewards = []


def choose_action(state, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(len(actions)))
    return int(np.argmax(q[state]))


for episode in range(500):
    state = (0, 0)
    epsilon = max(0.05, 0.8 * (0.995 ** episode))
    action = choose_action(state, epsilon)
    total = 0
    for _ in range(60):
        move = actions[action]
        next_state = (
            int(np.clip(state[0] + move[0], 0, size - 1)),
            int(np.clip(state[1] + move[1], 0, size - 1)),
        )
        reward = 10 if next_state == goal else -0.1
        next_action = choose_action(next_state, epsilon)
        q[state][action] += 0.2 * (reward + 0.95 * q[next_state][next_action] - q[state][action])
        state, action = next_state, next_action
        total += reward
        if state == goal:
            break
    rewards.append(total)

print(f"SARSA final 50-episode average reward: {np.mean(rewards[-50:]):.2f}")
plt.figure(figsize=(7, 4))
plt.plot(np.convolve(rewards, np.ones(20) / 20, mode="valid"), color="#9b5de5")
plt.title("SARSA Moving Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(OUT / "17_sarsa.png", dpi=150)
plt.show()

