import numpy as np


def softmax(z):
    # Subtract the maximum for numerical stability
    shifted = z - np.max(z, axis=1, keepdims=True)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True,
    )


def main():
    # Three example output scores
    scores = np.array([
        [1.2, 0.3, 2.1],
        [2.0, 1.0, 0.1],
    ])

    probabilities = softmax(scores)

    print("Input scores:")
    print(scores)

    print("\nSoftmax probabilities:")
    print(probabilities)

    print("\nProbability sums per sample:")
    print(probabilities.sum(axis=1))

    print("\nPredicted classes:")
    print(np.argmax(probabilities, axis=1))

    # Verify mathematical properties
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )

    print("\nAll Softmax checks passed.")


if __name__ == "__main__":
    main()
