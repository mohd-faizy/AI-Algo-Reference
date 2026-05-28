from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

x, _ = make_moons(n_samples=450, noise=0.07, random_state=42)
x = StandardScaler().fit_transform(x)
labels = DBSCAN(eps=0.25, min_samples=6).fit_predict(x)
cluster_labels = labels[labels != -1]
score = silhouette_score(x[labels != -1], cluster_labels) if len(set(cluster_labels)) > 1 else float("nan")
print(f"DBSCAN clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Silhouette excluding noise: {score:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="tab10", edgecolor="k", s=25)
plt.title("DBSCAN Finds Nonlinear Clusters and Noise")
plt.tight_layout()
plt.savefig(OUT / "14_dbscan.png", dpi=150)
plt.show()

