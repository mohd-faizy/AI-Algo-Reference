from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

iris = load_iris()
x = StandardScaler().fit_transform(iris.data)
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(x)
print(f"K-Means silhouette score: {silhouette_score(x, labels):.3f}")

points = PCA(n_components=2, random_state=42).fit_transform(x)


plt.figure(figsize=(7, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis", edgecolor="k")
plt.title("K-Means Clusters on Iris")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig(OUT / "11_k_means_clustering.png", dpi=150)
plt.show()

