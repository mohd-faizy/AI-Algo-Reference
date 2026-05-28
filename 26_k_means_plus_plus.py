from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

x, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.75, random_state=42)
model = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels = model.fit_predict(x)
print(f"K-Means++ inertia: {model.inertia_:.2f}")
print(f"K-Means++ silhouette score: {silhouette_score(x, labels):.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="viridis", s=25)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="red", marker="X", s=180, label="Centers")
plt.title("K-Means++ Initialization")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "26_k_means_plus_plus.png", dpi=150)
plt.show()

