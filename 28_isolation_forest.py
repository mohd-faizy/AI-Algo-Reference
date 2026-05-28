from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

normal = rng.normal(0, 1, size=(260, 2))
anomalies = rng.uniform(low=-6, high=6, size=(28, 2))
x = np.vstack([normal, anomalies])
model = IsolationForest(contamination=0.1, random_state=42)
labels = model.fit_predict(x)
print(f"Detected anomalies: {(labels == -1).sum()}")

plt.figure(figsize=(7, 5))
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], label="Normal", alpha=0.7)
plt.scatter(x[labels == -1, 0], x[labels == -1, 1], label="Anomaly", color="crimson", edgecolor="k")
plt.title("Isolation Forest Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "28_isolation_forest.png", dpi=150)
plt.show()

