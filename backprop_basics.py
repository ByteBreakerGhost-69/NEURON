import numpy as np


def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def loss_function(X, y, params):
    W1, b1, W2, b2 = params

    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)

    correct = P[np.arange(len(y)), y]
    return -np.mean(np.log(np.clip(correct, 1e-12, 1.0)))


def analytical_gradients(X, y, params):
    W1, b1, W2, b2 = params
    m = len(X)

    # Forward propagation
    Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)

    # Convert labels to one-hot encoding
    Y = np.zeros_like(P)
    Y[np.arange(m), y] = 1.0

    # Backpropagation: output layer
    dZ2 = (P - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # Backpropagation: hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return [dW1, db1, dW2, db2]


def numerical_gradients(X, y, params, epsilon=1e-5):
    gradients = []

    for param in params:
        gradient = np.zeros_like(param)

        for index in np.ndindex(param.shape):
            plus_params = [p.copy() for p in params]
            minus_params = [p.copy() for p in params]

            plus_params[len(gradients)][index] += epsilon
            minus_params[len(gradients)][index] -= epsilon

            loss_plus = loss_function(X, y, plus_params)
            loss_minus = loss_function(X, y, minus_params)

            gradient[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

        gradients.append(gradient)

    return gradients


def main():
    rng = np.random.default_rng(7)

    # Small network: 3 inputs -> 4 hidden neurons -> 3 classes
    X = np.array([
        [0.2, 0.5, 0.8],
        [0.9, 0.1, 0.4],
    ])
    y = np.array([2, 0])

    W1 = rng.normal(0, 0.3, size=(3, 4))
    b1 = np.array([0.1, 0.2, 0.1, 0.2])
    W2 = rng.normal(0, 0.3, size=(4, 3))
    b2 = np.array([0.1, -0.1, 0.2])

    params = [W1, b1, W2, b2]

    analytical = analytical_gradients(X, y, params)
    numerical = numerical_gradients(X, y, params)

    names = ["W1", "b1", "W2", "b2"]

    print(f"Initial loss: {loss_function(X, y, params):.6f}\n")

    for name, exact, approx in zip(names, analytical, numerical):
        difference = np.max(np.abs(exact - approx))
        print(f"{name}: maximum gradient difference = {difference:.10f}")

    max_difference = max(
        np.max(np.abs(a - n))
        for a, n in zip(analytical, numerical)
    )

    assert max_difference < 1e-5, (
        f"Gradient check failed: {max_difference}"
    )

    print("\nSemua gradien W1, b1, W2, dan b2 berhasil diverifikasi!")


if __name__ == "__main__":
    main()
