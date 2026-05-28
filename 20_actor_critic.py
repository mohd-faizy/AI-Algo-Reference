from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.actor = nn.Linear(32, 4)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        hidden = self.body(state)
        return torch.softmax(self.actor(hidden), dim=-1), self.critic(hidden).squeeze(-1)


def step(state, action):
    moves = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
    next_state = torch.clamp(state + moves[action], 0, 4)
    done = bool(torch.equal(next_state, torch.tensor([4, 4])))
    return next_state, 10.0 if done else -0.1, done


model = ActorCritic()
opt = torch.optim.Adam(model.parameters(), lr=0.003)
scores = []

for _ in range(260):
    state = torch.tensor([0, 0])
    total = 0.0
    for _ in range(60):
        probs, value = model(state.float() / 4.0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        next_state, reward, done = step(state, int(action))
        with torch.no_grad():
            _, next_value = model(next_state.float() / 4.0)
            target = torch.tensor(reward) + (0.95 * next_value * (not done))
        advantage = target - value
        entropy = dist.entropy()
        actor_loss = -dist.log_prob(action) * advantage.detach() - 0.1 * entropy
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        state = next_state
        total += reward
        if done:
            break
    scores.append(total)


print(f"Actor-Critic final average reward: {np.mean(scores[-30:]):.2f}")
plt.figure(figsize=(7, 4))
plt.plot(np.convolve(scores, np.ones(10) / 10, mode="valid"), color="#118ab2")
plt.title("Actor-Critic Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig(OUT / "20_actor_critic.png", dpi=150)
plt.show()
