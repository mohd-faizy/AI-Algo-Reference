from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=42)
scaler = StandardScaler()
x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
losses = []

for _ in range(60):
    logits = model(x_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))

pred = model(x_test).argmax(1).numpy()
print(f"ANN digit accuracy: {accuracy_score(y_test, pred):.3f}")
plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.title("ANN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross entropy")
plt.tight_layout()
plt.savefig(OUT / "21_artificial_neural_network.png", dpi=150)
plt.show()
