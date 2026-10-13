"""Generate evaluation visualizations for NEURON."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from neuron.model import NeuralNetwork
from neuron.train import load_data, train
from neuron.evaluate import evaluate_model, misclassified_examples

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Reproduce the experiment using the same seed and hyperparameters.
X_train, X_val, X_test, y_train, y_val, y_test = load_data(seed=42)
model = NeuralNetwork(seed=42)

history = train(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    learning_rate=0.1,
    batch_size=32,
    seed=42,
)

evaluation = evaluate_model(model, X_test, y_test)
mistakes = misclassified_examples(model, X_test, y_test)

# 1. Confusion matrix
cm = evaluation["confusion_matrix"]

fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax, label="Number of samples")
ax.set(
    xlabel="Predicted digit",
    ylabel="Actual digit",
    title="NEURON — Test Set Confusion Matrix",
    xticks=np.arange(10),
    yticks=np.arange(10),
)

threshold = cm.max() / 2
for row in range(10):
    for col in range(10):
        ax.text(
            col, row, str(cm[row, col]),
            ha="center", va="center",
            color="white" if cm[row, col] > threshold else "black",
        )

fig.tight_layout()
fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
plt.close(fig)

# 2. Visualize up to 12 incorrectly classified examples.
digits = load_digits()
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
axes = axes.ravel()

for ax in axes:
    ax.axis("off")

for ax, mistake in zip(axes, mistakes[:12]):
    index = mistake["index"]
    ax.imshow(X_test[index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(
        f"Actual: {mistake['true_label']} | "
        f"Pred: {mistake['predicted_label']}\n"
        f"Confidence: {mistake['confidence']:.1%}",
        fontsize=9,
    )
    ax.axis("off")

fig.suptitle("NEURON — Misclassified Test Samples")
fig.tight_layout()
fig.savefig(RESULTS / "misclassified_examples.png", dpi=160)
plt.close(fig)

# 3. Save numerical metrics as machine-readable JSON.
report = {
    "dataset": "scikit-learn digits",
    "seed": 42,
    "architecture": "64-32-10",
    "epochs": 100,
    "learning_rate": 0.1,
    "batch_size": 32,
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "test_accuracy": evaluation["accuracy"],
    "precision_macro": evaluation["precision_macro"],
    "recall_macro": evaluation["recall_macro"],
    "f1_macro": evaluation["f1_macro"],
    "misclassified_count": len(mistakes),
    "confusion_matrix": cm.tolist(),
    "misclassified_examples": mistakes,
    "history": history,
}

(RESULTS / "test_metrics.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nEvaluation artifacts saved:")
print(f"- {RESULTS / 'confusion_matrix.png'}")
print(f"- {RESULTS / 'misclassified_examples.png'}")
print(f"- {RESULTS / 'test_metrics.json'}")
print(f"Test accuracy: {evaluation['accuracy']:.4%}")
print(f"Misclassified: {len(mistakes)}")
