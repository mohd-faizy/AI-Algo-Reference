from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, stratify=iris.target, random_state=42
)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(x_train, y_train)
pred = tree.predict(x_test)
print(f"Decision Tree accuracy: {accuracy_score(y_test, pred):.3f}")

plt.figure(figsize=(12, 6))
plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
)
plt.title("Decision Tree on Iris Dataset")
plt.tight_layout()
plt.savefig(OUT / "03_decision_tree.png", dpi=150)
plt.show()

