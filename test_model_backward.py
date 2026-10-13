import numpy as np

from neuron.model import NeuralNetwork


def main():
    model = NeuralNetwork(seed=42)

    rng = np.random.default_rng(10)
    X = rng.random((5, 64))
    y = np.array([0, 3, 7, 2, 9])

    probabilities = model.forward(X)
    loss = model.compute_loss(probabilities, y)
    gradients = model.backward(y)

    print(f"Loss before update: {loss:.6f}")

    for name, gradient in gradients.items():
        print(
            f"{name}: shape={gradient.shape}, "
            f"finite={np.all(np.isfinite(gradient))}"
        )

        assert np.all(np.isfinite(gradient))

    assert gradients["dW1"].shape == model.W1.shape
    assert gradients["db1"].shape == model.b1.shape
    assert gradients["dW2"].shape == model.W2.shape
    assert gradients["db2"].shape == model.b2.shape

    print("\nAll backward checks passed!")


if __name__ == "__main__":
    main()
