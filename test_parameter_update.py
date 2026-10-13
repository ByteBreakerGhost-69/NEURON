import numpy as np

from neuron.model import NeuralNetwork


def main():
    model = NeuralNetwork(seed=42)
    rng = np.random.default_rng(10)

    X = rng.random((5, 64))
    y = np.array([0, 3, 7, 2, 9])

    probabilities = model.forward(X)
    loss_before = model.compute_loss(probabilities, y)

    gradients = model.backward(y)
    model.update_parameters(gradients, learning_rate=0.01)

    new_probabilities = model.forward(X)
    loss_after = model.compute_loss(new_probabilities, y)

    print(f"Loss before update: {loss_before:.8f}")
    print(f"Loss after update:  {loss_after:.8f}")
    print(f"Loss decreased:     {loss_after < loss_before}")

    assert loss_after < loss_before, (
        "Loss did not decrease; inspect gradients or learning rate."
    )

    print("\nParameter update test passed!")


if __name__ == "__main__":
    main()
