from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def fitness(population):
    x = population[:, 0]
    y = population[:, 1]
    return np.sin(3 * x) * np.cos(2 * y) + 0.15 * x


population = rng.uniform(-3, 3, size=(80, 2))
best_scores = []

for _ in range(80):
    scores = fitness(population)
    best_scores.append(scores.max())
    parents = population[np.argsort(scores)[-20:]]
    children = []
    while len(children) < len(population):
        a, b = parents[rng.choice(len(parents), 2, replace=False)]
        alpha = rng.random()
        child = alpha * a + (1 - alpha) * b
        child += rng.normal(0, 0.18, size=2)
        children.append(np.clip(child, -3, 3))
    population = np.array(children)

best = population[np.argmax(fitness(population))]
print(f"Genetic Algorithm best x={best[0]:.3f}, y={best[1]:.3f}, fitness={fitness(best[None, :])[0]:.3f}")
plt.figure(figsize=(7, 4))
plt.plot(best_scores, color="#ff9f1c")
plt.title("Genetic Algorithm Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.tight_layout()
plt.savefig(OUT / "30_genetic_algorithms.png", dpi=150)
plt.show()

