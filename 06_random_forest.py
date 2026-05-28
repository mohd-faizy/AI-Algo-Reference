from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    wine.data, wine.target, stratify=wine.target, random_state=42
)

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(x_train, y_train)
pred = forest.predict(x_test)
print(f"Random Forest accuracy: {accuracy_score(y_test, pred):.3f}")

pd.Series(forest.feature_importances_, index=wine.feature_names).nlargest(10).sort_values().plot(
    kind="barh", figsize=(8, 5), color="#2d6cdf"
)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(OUT / "06_random_forest.png", dpi=150)
plt.show()

