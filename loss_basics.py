import numpy as np


def cross_entropy_loss(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")

    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Labels must match the batch size.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    if np.any(labels < 0) or np.any(labels >= probabilities.shape[1]):
        raise ValueError("Label is outside the valid class range.")

    # Select the probability assigned to each correct class
    correct_probabilities = probabilities[
        np.arange(len(labels)),
        labels,
    ]

    # Avoid log(0)
    correct_probabilities = np.clip(
        correct_probabilities,
        1e-12,
        1.0,
    )

    return -np.mean(np.log(correct_probabilities))


def main():
    # Each row represents one sample and ten digit classes.
    probabilities = np.array([
        [0.02, 0.03, 0.04, 0.05, 0.04,
         0.06, 0.07, 0.60, 0.05, 0.04],

        [0.05, 0.70, 0.04, 0.03, 0.04,
         0.03, 0.02, 0.03, 0.03, 0.03],
    ])

    labels = np.array([7, 1])

    loss = cross_entropy_loss(probabilities, labels)

    print("Correct labels:", labels)
    print("Probabilities of correct classes:",
          probabilities[np.arange(len(labels)), labels])
    print(f"Mean cross-entropy loss: {loss:.6f}")

    # A confident correct prediction should have lower loss.
    good = cross_entropy_loss(
        np.array([[0.01, 0.98, 0.01]]),
        np.array([1]),
    )

    poor = cross_entropy_loss(
        np.array([[0.01, 0.01, 0.98]]),
        np.array([1]),
    )

    print(f"Good prediction loss: {good:.6f}")
    print(f"Poor prediction loss: {poor:.6f}")

    assert good < poor
    print("\nAll cross-entropy checks passed.")


if __name__ == "__main__":
    main()
