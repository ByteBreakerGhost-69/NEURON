
"""Training utilities for NEURON's NumPy neural network."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neuron.model import NeuralNetwork


def load_data(seed=42):
    """Load, normalize, and split the handwritten digit dataset."""
    digits = load_digits()

    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)

    # Reserve 20% for the final test set.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # Reserve 20% of the remaining data for validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=seed,
        stratify=y_trainval,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def accuracy(model, X, y, batch_size=256):
    """Calculate classification accuracy without changing parameters."""
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("X and y must contain the same nonzero sample count.")

    predictions = []

    for start in range(0, len(X), batch_size):
        probabilities = model.forward(X[start:start + batch_size])
        predictions.append(np.argmax(probabilities, axis=1))

    predicted_labels = np.concatenate(predictions)
    return float(np.mean(predicted_labels == y))


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    learning_rate=0.1,
    batch_size=32,
    seed=42,
):
    """Train the model using mini-batch gradient descent."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("Training data must contain matching samples.")
    if len(X_val) == 0 or len(X_val) != len(y_val):
        raise ValueError("Validation data must contain matching samples.")

    rng = np.random.default_rng(seed)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        # Shuffle training samples each epoch.
        order = rng.permutation(len(X_train))
        X_shuffled = X_train[order]
        y_shuffled = y_train[order]

        for start in range(0, len(X_train), batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            probabilities = model.forward(X_batch)
            gradients = model.backward(y_batch)
            model.update_parameters(
                gradients,
                learning_rate=learning_rate,
            )

        # Measure the epoch's final model on the full training set.
        train_probabilities = model.forward(X_train)
        train_loss = model.compute_loss(train_probabilities, y_train)
        train_accuracy = float(
            np.mean(np.argmax(train_probabilities, axis=1) == y_train)
        )

        val_probabilities = model.forward(X_val)
        val_loss = model.compute_loss(val_probabilities, y_val)
        val_accuracy = float(
            np.mean(np.argmax(val_probabilities, axis=1) == y_val)
        )

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(val_accuracy)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"train acc: {train_accuracy:.2%} | "
                f"val loss: {val_loss:.4f} | "
                f"val acc: {val_accuracy:.2%}"
            )

    return history


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print("Dataset split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    model = NeuralNetwork(seed=42)

    print("\nStarting training...\n")
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

    print("\nTraining finished.")
    print(f"Final training accuracy: {history['train_accuracy'][-1]:.2%}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2%}")

    # The test set is evaluated only after training is complete.
    test_accuracy = accuracy(model, X_test, y_test)
    print(f"Final test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
