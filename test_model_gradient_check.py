import numpy as np

from neuron.model import NeuralNetwork


def main():
    # Model kecil agar gradient checking cepat.
    model = NeuralNetwork(
        input_size=3,
        hidden_size=4,
        output_size=3,
        seed=7,
    )

    X = np.array([
        [0.2, 0.5, 0.8],
        [0.9, 0.1, 0.4],
    ])
    y = np.array([2, 0])

    # Analytic gradients from our own backpropagation.
    model.forward(X)
    analytical = model.backward(y)

    parameters = {
        "W1": "dW1",
        "b1": "db1",
        "W2": "dW2",
        "b2": "db2",
    }

    epsilon = 1e-5
    max_difference = 0.0

    for parameter_name, gradient_name in parameters.items():
        parameter = getattr(model, parameter_name)
        numerical = np.zeros_like(parameter)

        for index in np.ndindex(parameter.shape):
            original = parameter[index]

            parameter[index] = original + epsilon
            loss_plus = model.compute_loss(model.forward(X), y)

            parameter[index] = original - epsilon
            loss_minus = model.compute_loss(model.forward(X), y)

            parameter[index] = original

            numerical[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

        difference = np.max(
            np.abs(analytical[gradient_name] - numerical)
        )
        max_difference = max(max_difference, difference)

        print(
            f"{parameter_name}: "
            f"maximum difference = {difference:.10e}"
        )

        assert np.allclose(
            analytical[gradient_name],
            numerical,
            atol=1e-5,
            rtol=1e-4,
        ), f"Gradient check failed for {parameter_name}"

    print(f"\nOverall maximum difference: {max_difference:.10e}")
    print("All model gradient checks passed!")


if __name__ == "__main__":
    main()
