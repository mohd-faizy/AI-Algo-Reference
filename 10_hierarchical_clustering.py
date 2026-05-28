from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. Define a rich, biologically-accurate taxonomic dataset of 14 animals with 10 distinct features.
# Features: [Warm-blooded, Gives birth, Has feathers, Has fur/hair, Breathes underwater, Lives in water, Can fly, Has legs, Has dry scales, Is predator]
# Note: Every single animal has a unique feature combination to ensure a beautifully resolved tree.
data = {
    "Dog":        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    "Cow":        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Bat":        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    "Dolphin":    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "Blue Whale": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "Eagle":      [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    "Ostrich":    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Penguin":    [1.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 1.0, 0.0, 1.0],
    "Shark":      [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "Salmon":     [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "Lizard":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "Snake":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    "Turtle":     [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0],
    "Frog":       [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0]
}

columns = [
    "Warm-blooded", "Gives birth", "Has feathers", "Has fur/hair", 
    "Breathes underwater", "Lives in water", "Can fly", "Has legs", 
    "Has dry scales", "Is predator"
]

df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
print(df)

# 2. Compute Ward Linkage
z = linkage(df.values, method="ward")

# Calculate optimal threshold to cut the dendrogram into exactly 4 natural classes:
# (Mammals, Birds, Fish, and Reptiles/Amphibians)
threshold = (z[-3, 2] + z[-4, 2]) / 2

# 3. Style and Plot
plt.figure(figsize=(11, 6.5), dpi=150)

# Set simple, named colors for the 4 main branch clusters
set_link_color_palette(["blue", "orange", "green", "purple"])

dendrogram(
    z,
    labels=df.index,
    color_threshold=threshold,
    leaf_rotation=45,
    leaf_font_size=10,
    above_threshold_color="grey"
)

# Draw optimal threshold cutting line
plt.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"4-Class Cut Threshold (Distance = {threshold:.2f})"
)

plt.title("Evolutionary Animal Taxonomy via Hierarchical Clustering", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Ward Linkage Distance", fontsize=11, labelpad=8)
plt.xlabel("Animal Species", fontsize=11, labelpad=8)
plt.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True, shadow=True)
plt.tight_layout()

# Save the polished output
plt.savefig(OUT / "10_hierarchical_clustering.png", dpi=200, bbox_inches="tight")
plt.close()

print("Hierarchical clustering dendrogram saved to outputs/10_hierarchical_clustering.png")
