from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
torch.manual_seed(42)

digits = load_digits()
x = digits.images[:, None, :, :] / 16.0
x_train, x_test, y_train, y_test = train_test_split(x, digits.target, stratify=digits.target, random_state=42)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

model = nn.Sequential(
    nn.Conv2d(1, 12, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(12 * 4 * 4, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
losses = []

for _ in range(35):
    loss = loss_fn(model(x_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))

pred = model(x_test).argmax(1).numpy()
print(f"CNN digit accuracy: {accuracy_score(y_test, pred):.3f}")
plt.figure(figsize=(7, 4))
plt.plot(losses, color="#06d6a0")
plt.title("CNN Training Loss on Digits")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(OUT / "22_convolutional_neural_network.png", dpi=150)
plt.show()
