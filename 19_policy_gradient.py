from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))

    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)


def step(state, action):
    moves = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
    next_state = torch.clamp(state + moves[action], 0, 4)
    done = bool(torch.equal(next_state, torch.tensor([4, 4])))
    return next_state, 10.0 if done else -0.1, done


policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=0.005)
scores = []

for _ in range(260):
    state = torch.tensor([0, 0])
    log_probs, rewards = [], []
    for _ in range(60):
        probs = policy(state.float() / 4.0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        next_state, reward, done = step(state, int(action))
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        state = next_state
        if done:
            break
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + 0.95 * g
        returns.insert(0, g)
    returns = torch.tensor(returns)
    loss = -torch.stack(log_probs).mul(returns).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    scores.append(sum(rewards))

print(f"Policy Gradient final average reward: {np.mean(scores[-30:]):.2f}")
plt.figure(figsize=(7, 4))
plt.plot(np.convolve(scores, np.ones(10) / 10, mode="valid"), color="#ef476f")
plt.title("Policy Gradient Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(OUT / "19_policy_gradient.png", dpi=150)
plt.show()

