from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

iris = load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, random_state=42
)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"SVM accuracy: {accuracy_score(y_test, pred):.3f}")

DecisionBoundaryDisplay.from_estimator(
    model, x, response_method="predict", cmap="Set3", alpha=0.45
)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="Dark2", edgecolor="k")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("SVM Decision Regions")
plt.tight_layout()
plt.savefig(OUT / "05_support_vector_machine.png", dpi=150)
plt.show()

