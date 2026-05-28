from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

digits = load_digits()
x = digits.data / 16.0
x_train, x_test = train_test_split(x, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)

encoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
model = nn.Sequential(encoder, decoder)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for _ in range(80):
    recon = model(x_train)
    loss = loss_fn(recon, x_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

sample = x_test[:8]
recon = model(sample).detach().numpy()
print(f"Autoencoder reconstruction MSE: {loss_fn(model(x_test), x_test).item():.4f}")

fig, axes = plt.subplots(2, 8, figsize=(9, 3))
for i in range(8):
    axes[0, i].imshow(sample[i].reshape(8, 8), cmap="gray")
    axes[1, i].imshow(recon[i].reshape(8, 8), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Decoded")
plt.suptitle("Autoencoder Compression on Digits")
plt.tight_layout()
plt.savefig(OUT / "27_autoencoders.png", dpi=150)
plt.show()

