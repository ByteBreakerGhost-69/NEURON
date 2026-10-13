import numpy as np

from neuron.model import NeuralNetwork


def main():
    model = NeuralNetwork()

    X = np.zeros((3, 64))
    y = np.array([0, 1, 2])

    probabilities = model.forward(X)

    print("Probability shape:", probabilities.shape)
    print("Row sums:", probabilities.sum(axis=1))

    loss = model.compute_loss(probabilities, y)
    print(f"Cross-entropy loss: {loss:.6f}")

    assert probabilities.shape == (3, 10)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(loss)
    assert loss > 0

    # Perfectly confident predictions should have very low loss.
    good_probabilities = np.full((3, 10), 0.001)
    good_probabilities[np.arange(3), y] = 0.991

    good_loss = model.compute_loss(good_probabilities, y)

    print(f"Good prediction loss: {good_loss:.6f}")

    assert good_loss < loss

    print("\nAll model loss checks passed!")


if __name__ == "__main__":
    main()
