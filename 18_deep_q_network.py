from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
torch.manual_seed(42)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.net(x)


def step(state, action):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move = moves[action]
    next_state = np.clip(np.array(state) + np.array(move), 0, 4)
    done = tuple(next_state) == (4, 4)
    return tuple(next_state), 10.0 if done else -0.1, done


model = DQN()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
memory = deque(maxlen=2000)
rewards = []

for episode in range(220):
    state = (0, 0)
    total = 0.0
    epsilon = max(0.05, 0.9 * (0.985 ** episode))
    for _ in range(50):
        with torch.no_grad():
            s = torch.tensor([state], dtype=torch.float32) / 4.0
            action = int(rng.integers(4) if rng.random() < epsilon else torch.argmax(model(s)).item())
        next_state, reward, done = step(state, action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total += reward

        if len(memory) >= 32:
            batch_idx = rng.choice(len(memory), size=32, replace=False)
            batch = [memory[i] for i in batch_idx]
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32) / 4.0
            actions = torch.tensor([b[1] for b in batch])
            batch_rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32) / 4.0
            done_mask = torch.tensor([b[4] for b in batch], dtype=torch.float32)
            q_values = model(states).gather(1, actions[:, None]).squeeze(-1)
            targets = batch_rewards + 0.95 * model(next_states).max(dim=1).values.detach() * (1 - done_mask)
            loss = loss_fn(q_values, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if done:
            break
    rewards.append(total)

print(f"DQN final 30-episode average reward: {np.mean(rewards[-30:]):.2f}")
plt.figure(figsize=(7, 4))
plt.plot(np.convolve(rewards, np.ones(10) / 10, mode="valid"))
plt.title("Deep Q-Network Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(OUT / "18_deep_q_network.png", dpi=150)
plt.show()

