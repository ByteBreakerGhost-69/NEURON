"""Evaluation utilities for NEURON's NumPy neural network."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _validate_data(X, y, output_size):
    """Validate features and integer class labels."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(X) == 0:
        raise ValueError("X must contain at least one sample.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y must contain one label per sample.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be integers.")
    if np.any(y < 0) or np.any(y >= output_size):
        raise ValueError("Label is outside the valid class range.")

    return X, y


def predict(model, X, batch_size=256):
    """Predict digit labels in batches."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if len(X) == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] != model.input_size:
        raise ValueError(
            f"Expected {model.input_size} features, received {X.shape[1]}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    batches = []

    for start in range(0, len(X), batch_size):
        probabilities = model.forward(X[start:start + batch_size])
        batches.append(np.argmax(probabilities, axis=1))

    return np.concatenate(batches)


def evaluate_model(model, X, y, batch_size=256):
    """Calculate classification metrics and a confusion matrix."""
    X, y = _validate_data(X, y, model.output_size)

    if X.shape[1] != model.input_size:
        raise ValueError(
            f"Expected {model.input_size} features, received {X.shape[1]}."
        )

    y_pred = predict(model, X, batch_size=batch_size)
    labels = np.arange(model.output_size)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(
            precision_score(
                y, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "confusion_matrix": sklearn_confusion_matrix(
            y, y_pred, labels=labels
        ),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": y_pred,
    }


def misclassified_examples(model, X, y, batch_size=256):
    """Return indices and confidence for incorrectly classified samples."""
    X, y = _validate_data(X, y, model.output_size)

    if X.shape[1] != model.input_size:
        raise ValueError(
            f"Expected {model.input_size} features, received {X.shape[1]}."
        )

    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    probability_batches = []

    for start in range(0, len(X), batch_size):
        probability_batches.append(
            model.forward(X[start:start + batch_size])
        )

    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = np.argmax(probabilities, axis=1)
    wrong_indices = np.flatnonzero(predictions != y)

    return [
        {
            "index": int(i),
            "true_label": int(y[i]),
            "predicted_label": int(predictions[i]),
            "confidence": float(probabilities[i, predictions[i]]),
        }
        for i in wrong_indices
    ]
