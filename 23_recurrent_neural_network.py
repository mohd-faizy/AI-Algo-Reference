from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

t = np.linspace(0, 80, 900)
series = np.sin(t) + 0.25 * np.sin(3 * t)
window = 20
x = np.array([series[i : i + window] for i in range(len(series) - window)], dtype=np.float32)[:, :, None]
y = np.array([series[i + window] for i in range(len(series) - window)], dtype=np.float32)[:, None]
x_train = torch.tensor(x[:700])
y_train = torch.tensor(y[:700])
x_test = torch.tensor(x[700:])
y_test = y[700:]


class RNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 24, batch_first=True)
        self.out = nn.Linear(24, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.out(hidden[-1])


model = RNNRegressor()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for _ in range(80):
    loss = loss_fn(model(x_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

pred = model(x_test).detach().numpy().ravel()
print(f"RNN test MSE: {np.mean((pred - y_test.ravel()) ** 2):.4f}")
plt.figure(figsize=(8, 4))
plt.plot(y_test[:160], label="Actual")
plt.plot(pred[:160], label="Predicted")
plt.title("RNN One-Step Sine Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "23_recurrent_neural_network.png", dpi=150)
plt.show()

