from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

x, y = make_friedman1(n_samples=800, n_features=10, noise=1.0, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model = GradientBoostingRegressor(n_estimators=180, max_depth=3, random_state=42)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Gradient Boosting R2: {r2_score(y_test, pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Gradient Boosting Regression")
plt.tight_layout()
plt.savefig(OUT / "08_gradient_boosting.png", dpi=150)
plt.show()
