from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

x, y = make_moons(n_samples=500, noise=0.28, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)

model = AdaBoostClassifier(n_estimators=120, learning_rate=0.6, random_state=42)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"AdaBoost accuracy: {accuracy_score(y_test, pred):.3f}")

DecisionBoundaryDisplay.from_estimator(model, x, response_method="predict", alpha=0.35, cmap="coolwarm")
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
plt.title("AdaBoost on Two Moons")
plt.tight_layout()
plt.savefig(OUT / "07_adaboost.png", dpi=150)
plt.show()

