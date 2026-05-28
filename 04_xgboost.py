from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, stratify=data.target, random_state=42
)

try:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
        eval_metric="logloss",
        random_state=42,
    )
    label = "XGBoost"
except Exception:
    model = GradientBoostingClassifier(random_state=42)
    label = "GradientBoosting fallback"

model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"{label} accuracy: {accuracy_score(y_test, pred):.3f}")

importances = getattr(model, "feature_importances_", None)
top = pd.Series(importances, index=data.feature_names).nlargest(10)
top.sort_values().plot(kind="barh", figsize=(8, 5), color="#168f72")
plt.title(f"{label} Top Feature Importances")
plt.tight_layout()
plt.savefig(OUT / "04_xgboost.png", dpi=150)
plt.show()

