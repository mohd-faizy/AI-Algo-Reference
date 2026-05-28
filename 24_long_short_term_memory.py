from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

t = np.linspace(0, 90, 1000)
series = np.sin(t) + 0.4 * np.sin(0.2 * t)
window = 30
x = np.array([series[i : i + window] for i in range(len(series) - window)], dtype=np.float32)[:, :, None]
y = np.array([series[i + window] for i in range(len(series) - window)], dtype=np.float32)[:, None]
x_train = torch.tensor(x[:780])
y_train = torch.tensor(y[:780])
x_test = torch.tensor(x[780:])
y_test = y[780:]


class LSTMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 28, batch_first=True)
        self.out = nn.Linear(28, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.out(hidden[-1])


model = LSTMRegressor()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for _ in range(90):
    loss = loss_fn(model(x_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

pred = model(x_test).detach().numpy().ravel()
print(f"LSTM test MSE: {np.mean((pred - y_test.ravel()) ** 2):.4f}")
plt.figure(figsize=(8, 4))
plt.plot(y_test[:160], label="Actual")
plt.plot(pred[:160], label="Predicted")
plt.title("LSTM Forecast with Long-Range Signal")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "24_long_short_term_memory.png", dpi=150)
plt.show()

