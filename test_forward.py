import numpy as np

from neuron.model import NeuralNetwork


def main():
    model = NeuralNetwork()

    # Test a single image
    single_input = np.zeros(64)
    single_output = model.forward(single_input)

    assert single_output.shape == (1, 10)
    assert np.all(single_output >= 0)
    assert np.all(single_output <= 1)
    assert np.allclose(single_output.sum(axis=1), 1.0)

    # Test a batch of images
    batch_input = np.zeros((5, 64))
    batch_output = model.forward(batch_input)

    assert batch_output.shape == (5, 10)
    assert np.allclose(batch_output.sum(axis=1), 1.0)

    # Test reproducible initialization
    model2 = NeuralNetwork(seed=42)
    assert np.allclose(model.W1, model2.W1)
    assert np.allclose(model.W2, model2.W2)

    # Test invalid feature count
    try:
        model.forward(np.zeros(63))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid feature count was accepted.")

    print("Single input shape:", single_output.shape)
    print("Batch output shape:", batch_output.shape)
    print("Probability sum:", single_output.sum())
    print("Predicted class:", np.argmax(single_output[0]))
    print("\nAll forward propagation checks passed.")


if __name__ == "__main__":
    main()
