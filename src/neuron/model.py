
"""NumPy implementation of NEURON's neural network."""

import numpy as np


class NeuralNetwork:
    """A two-layer neural network for handwritten digit classification."""

    def __init__(
        self,
        input_size=64,
        hidden_size=32,
        output_size=10,
        seed=42,
    ):
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("Layer sizes must be positive.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        rng = np.random.default_rng(seed)

        # He initialization for the ReLU hidden layer
        self.W1 = rng.standard_normal(
            (input_size, hidden_size)
        ) * np.sqrt(2.0 / input_size)

        self.b1 = np.zeros(hidden_size)

        # Xavier-style initialization for the output layer
        self.W2 = rng.standard_normal(
            (hidden_size, output_size)
        ) * np.sqrt(1.0 / hidden_size)

        self.b2 = np.zeros(output_size)

        # Cache for values needed during backpropagation
        self._cache = None

    @staticmethod
    def relu(z):
        """Apply ReLU element-wise."""
        return np.maximum(0.0, z)

    @staticmethod
    def softmax(z):
        """Convert output scores into stable probabilities."""
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(shifted)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )

    def forward(self, X):
        """Return class probabilities for one sample or a batch."""
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if X.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {X.shape[1]}."
            )

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        probabilities = self.softmax(Z2)

        # Keep intermediate values for backward()
        self._cache = {
            "X": X.copy(),
            "Z1": Z1,
            "A1": A1,
            "probabilities": probabilities,
        }

        return probabilities

    def compute_loss(self, probabilities, y):
        """Compute mean cross-entropy loss for a batch."""
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )
        y = np.asarray(y)

        if probabilities.ndim != 2:
            raise ValueError("Probabilities must be a 2D array.")

        if probabilities.shape[1] != self.output_size:
            raise ValueError("Probability columns must match output_size.")

        if probabilities.shape[0] == 0:
            raise ValueError("At least one sample is required.")

        if y.ndim != 1 or len(y) != len(probabilities):
            raise ValueError("Labels must match the batch size.")

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Labels must be integers.")

        if np.any(y < 0) or np.any(y >= self.output_size):
            raise ValueError("Label is outside the valid class range.")

        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Probabilities must be finite.")

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Probabilities must be between 0 and 1.")

        correct_probabilities = probabilities[
            np.arange(len(y)),
            y,
        ]

        correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0,
        )

        return -np.mean(np.log(correct_probabilities))
       
    
    def backward(self, y):
        """Compute gradients for all weights and biases."""
        if self._cache is None:
            raise RuntimeError(
                "Call forward(X) before backward(y)."
            )

        y = np.asarray(y)

        X = self._cache["X"]
        Z1 = self._cache["Z1"]
        A1 = self._cache["A1"]
        probabilities = self._cache["probabilities"]

        m = len(X)

        if y.ndim != 1 or len(y) != m:
            raise ValueError("Labels must match the batch size.")

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Labels must be integers.")

        if np.any(y < 0) or np.any(y >= self.output_size):
            raise ValueError("Label is outside the valid class range.")

        # One-hot labels
        Y = np.zeros_like(probabilities)
        Y[np.arange(m), y] = 1.0

        # Output layer: Softmax + cross-entropy
        dZ2 = (probabilities - Y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # Hidden layer: backpropagate through ReLU
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

        return gradients

     
    def update_parameters(self, gradients, learning_rate=0.01):
        """Update weights and biases using gradient descent."""
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")

        parameter_names = ("W1", "b1", "W2", "b2")
        gradient_names = ("dW1", "db1", "dW2", "db2")

        # Validate everything before changing any parameter.
        for parameter_name, gradient_name in zip(
            parameter_names, gradient_names
        ):
            parameter = getattr(self, parameter_name)

            if gradient_name not in gradients:
                raise ValueError(f"Missing gradient: {gradient_name}")

            gradient = np.asarray(gradients[gradient_name])

            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"{gradient_name} has shape {gradient.shape}; "
                    f"expected {parameter.shape}."
                )

            if not np.all(np.isfinite(gradient)):
                raise ValueError(
                    f"{gradient_name} must contain finite values."
                )

        # Gradient descent
        for parameter_name, gradient_name in zip(
            parameter_names, gradient_names
        ):
            parameter = getattr(self, parameter_name)
            gradient = gradients[gradient_name]

            parameter -= learning_rate * gradient

