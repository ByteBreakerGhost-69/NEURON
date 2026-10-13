"""Tests for NEURON model evaluation."""

import numpy as np
import pytest

from neuron.evaluate import (
    evaluate_model,
    misclassified_examples,
    predict,
)
from neuron.model import NeuralNetwork


def make_test_data():
    rng = np.random.default_rng(7)
    X = rng.random((30, 64))
    y = np.arange(30) % 10
    return X, y


def test_predict_returns_one_valid_label_per_sample():
    model = NeuralNetwork(seed=1)
    X, _ = make_test_data()

    predictions = predict(model, X, batch_size=7)

    assert predictions.shape == (len(X),)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert np.all((predictions >= 0) & (predictions < 10))


def test_evaluate_model_returns_expected_metrics_and_matrix():
    model = NeuralNetwork(seed=1)
    X, y = make_test_data()

    result = evaluate_model(model, X, y, batch_size=8)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["precision_macro"] <= 1.0
    assert 0.0 <= result["recall_macro"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0
    assert result["confusion_matrix"].shape == (10, 10)
    assert result["confusion_matrix"].sum() == len(X)
    assert result["predictions"].shape == y.shape
    assert "macro avg" in result["classification_report"]


def test_misclassified_examples_only_contains_wrong_predictions():
    model = NeuralNetwork(seed=1)
    X, y = make_test_data()

    mistakes = misclassified_examples(model, X, y, batch_size=6)

    assert all(item["true_label"] != item["predicted_label"] for item in mistakes)
    assert all(0.0 <= item["confidence"] <= 1.0 for item in mistakes)
    assert all(0 <= item["index"] < len(X) for item in mistakes)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_predict_rejects_invalid_batch_size(batch_size):
    model = NeuralNetwork(seed=1)
    X, _ = make_test_data()

    with pytest.raises(ValueError):
        predict(model, X, batch_size=batch_size)


def test_evaluation_rejects_mismatched_labels():
    model = NeuralNetwork(seed=1)
    X, y = make_test_data()

    with pytest.raises(ValueError):
        evaluate_model(model, X, y[:-1])
