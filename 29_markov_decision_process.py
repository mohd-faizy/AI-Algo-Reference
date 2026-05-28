from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

size = 5
goal = (4, 4)
trap = (2, 2)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
values = np.zeros((size, size))

for _ in range(120):
    new_values = values.copy()
    for row in range(size):
        for col in range(size):
            if (row, col) == goal:
                new_values[row, col] = 10
                continue
            action_values = []
            for dr, dc in actions:
                nr, nc = int(np.clip(row + dr, 0, size - 1)), int(np.clip(col + dc, 0, size - 1))
                reward = -5 if (nr, nc) == trap else (-0.1 if (nr, nc) != goal else 10)
                action_values.append(reward + 0.95 * values[nr, nc])
            new_values[row, col] = max(action_values)
    values = new_values

policy = np.zeros((size, size), dtype=int)
for row in range(size):
    for col in range(size):
        scores = []
        for dr, dc in actions:
            nr, nc = int(np.clip(row + dr, 0, size - 1)), int(np.clip(col + dc, 0, size - 1))
            scores.append(values[nr, nc])
        policy[row, col] = int(np.argmax(scores))

print(f"MDP start-state value after value iteration: {values[0, 0]:.2f}")
plt.figure(figsize=(6, 5))
plt.imshow(values, cmap="viridis")
plt.colorbar(label="State value")
plt.scatter([goal[1]], [goal[0]], marker="*", s=250, color="gold", edgecolor="k", label="Goal")
plt.scatter([trap[1]], [trap[0]], marker="X", s=170, color="crimson", label="Trap")
plt.title("MDP Value Iteration")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "29_markov_decision_process.png", dpi=150)
plt.show()

