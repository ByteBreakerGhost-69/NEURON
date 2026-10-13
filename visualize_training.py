from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuron.model import NeuralNetwork
from neuron.train import load_data, train, accuracy


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        seed=42
    )

    model = NeuralNetwork(seed=42)

    print("Running training to collect history...\n")

    history = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        learning_rate=0.1,
        batch_size=32,
        seed=42,
    )

    # Persist the numeric history for later analysis.
    history_path = output_dir / "training_history.npz"
    np.savez(
        history_path,
        **{
            name: np.asarray(values)
            for name, values in history.items()
        },
    )

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Plot 1: loss curves
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Training loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.set_title("NEURON — Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    loss_path = output_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # Plot 2: accuracy curves
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epochs,
        history["train_accuracy"],
        label="Training accuracy",
    )
    ax.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy",
    )
    ax.set_title("NEURON — Accuracy Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    accuracy_path = output_dir / "accuracy_curve.png"
    fig.savefig(accuracy_path, dpi=150)
    plt.close(fig)

    # Report the best validation-loss epoch for analysis.
    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1

    print("\nVisualization complete.")
    print(f"History: {history_path}")
    print(f"Loss curve: {loss_path}")
    print(f"Accuracy curve: {accuracy_path}")
    print(f"Best validation-loss epoch: {best_epoch}")
    print(f"Validation loss there: {history['val_loss'][best_index]:.6f}")
    print(f"Validation accuracy there: {history['val_accuracy'][best_index]:.2%}")

    # Test set is used only for final evaluation, not model selection.
    test_accuracy = accuracy(model, X_test, y_test)
    print(f"Final test accuracy (last epoch): {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
