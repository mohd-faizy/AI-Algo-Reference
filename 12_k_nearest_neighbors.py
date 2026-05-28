from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, stratify=digits.target, random_state=42
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"K-NN digit accuracy: {accuracy_score(y_test, pred):.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("K-NN Confusion Matrix on Digits")
plt.tight_layout()
plt.savefig(OUT / "12_k_nearest_neighbors.png", dpi=150)
plt.show()

