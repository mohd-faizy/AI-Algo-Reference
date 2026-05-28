from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

diabetes = load_diabetes()
x = diabetes.data[:, [2]]
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

print(f"Linear Regression R2: {r2_score(y_test, pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

order = np.argsort(x_test.ravel())
plt.figure(figsize=(7, 5))
plt.scatter(x_test, y_test, alpha=0.7, label="Actual")
plt.plot(x_test[order], pred[order], color="crimson", linewidth=2, label="Fit")
plt.xlabel("BMI feature")
plt.ylabel("Disease progression")
plt.title("Linear Regression on Diabetes Dataset")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "01_linear_regression.png", dpi=150)
plt.show()
