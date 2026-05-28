from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

wine = load_wine()
x = StandardScaler().fit_transform(wine.data)
pca = PCA(n_components=2, random_state=42)
points = pca.fit_transform(x)
print(f"Explained variance by first 2 PCs: {pca.explained_variance_ratio_.sum():.3f}")

plt.figure(figsize=(7, 5))
plt.scatter(points[:, 0], points[:, 1], c=wine.target, cmap="Set1", edgecolor="k")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA Projection of Wine Dataset")
plt.tight_layout()
plt.savefig(OUT / "13_principal_component_analysis.png", dpi=150)
plt.show()

