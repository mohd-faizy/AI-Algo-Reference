from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

t = np.linspace(0, 70, 720)
series = np.sin(t) + 0.3 * np.cos(2.5 * t)
window = 24
x = np.array([series[i : i + window] for i in range(len(series) - window)], dtype=np.float32)[:, :, None]
y = np.array([series[i + window] for i in range(len(series) - window)], dtype=np.float32)[:, None]
x_train = torch.tensor(x[:560])
y_train = torch.tensor(y[:560])
x_test = torch.tensor(x[560:])
y_test = y[560:]


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        self.pos = nn.Parameter(torch.randn(1, window, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dim_feedforward=64)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        encoded = self.encoder(self.embed(x) + self.pos)
        return self.out(encoded[:, -1])


model = TinyTransformer()
opt = torch.optim.Adam(model.parameters(), lr=0.008)
loss_fn = nn.MSELoss()
for _ in range(70):
    loss = loss_fn(model(x_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

pred = model(x_test).detach().numpy().ravel()
print(f"Transformer test MSE: {np.mean((pred - y_test.ravel()) ** 2):.4f}")
plt.figure(figsize=(8, 4))
plt.plot(y_test[:140], label="Actual")
plt.plot(pred[:140], label="Predicted")
plt.title("Transformer Encoder Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "25_transformer.png", dpi=150)
plt.show()

