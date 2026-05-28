import socket
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Set socket timeout to prevent fetch_20newsgroups from hanging indefinitely
socket.setdefaulttimeout(10.0)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

categories = ["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"]
try:
    data = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
    labels = data.target_names
except BaseException:
    print("\n[!] Could not download 20newsgroups dataset or download was interrupted. Using synthetic fallback data...")
    data = [
        "space shuttle orbit nasa moon", "graphics rendering image pixels",
        "engine wheels highway car", "election policy government vote",
        "planet telescope mars orbit", "gpu shader render texture",
        "sedan brake tire road", "senate campaign law debate",
    ] * 20
    y = [0, 1, 2, 3, 0, 1, 2, 3] * 20
    x_train, x_test, y_train, y_test = train_test_split(data, y, stratify=y, random_state=42)
    labels = categories

model = make_pipeline(TfidfVectorizer(stop_words="english"), MultinomialNB())
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(f"Naive Bayes text accuracy: {accuracy_score(y_test, pred):.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=labels, xticks_rotation=30)

plt.title("Naive Bayes Text Classification")
plt.tight_layout()
plt.savefig(OUT / "09_naive_bayes.png", dpi=150)
plt.show()

