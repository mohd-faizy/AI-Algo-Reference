from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, stratify=data.target, random_state=42
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(x_train, y_train)
pred = model.predict(x_test)
proba = model.predict_proba(x_test)[:, 1]

print(f"Logistic Regression accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")

sns.set_theme(style="whitegrid")
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=data.target_names)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "02_logistic_regression.png", dpi=150)
plt.show()

