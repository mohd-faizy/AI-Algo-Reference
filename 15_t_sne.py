from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

digits = load_digits()
x = StandardScaler().fit_transform(digits.data)
embedding = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(x)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap="tab10", s=12)
plt.legend(*scatter.legend_elements(), title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("t-SNE Visualization of Handwritten Digits")
plt.tight_layout()
plt.savefig(OUT / "15_t_sne.png", dpi=150)
plt.show()

print("t-SNE embedding saved.")

