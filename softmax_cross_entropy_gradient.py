import numpy as np


def softmax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(probabilities, labels):
    n = len(labels)
    correct_probabilities = probabilities[np.arange(n), labels]
    return -np.mean(np.log(np.clip(correct_probabilities, 1e-12, 1.0)))


def analytical_gradient(logits, labels):
    n = len(labels)
    probabilities = softmax(logits)

    # Turunan Softmax + cross-entropy
    dZ = probabilities.copy()
    dZ[np.arange(n), labels] -= 1
    dZ /= n

    return dZ


def numerical_gradient(logits, labels, epsilon=1e-5):
    gradient = np.zeros_like(logits, dtype=np.float64)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()

            plus[i, j] += epsilon
            minus[i, j] -= epsilon

            loss_plus = cross_entropy_loss(softmax(plus), labels)
            loss_minus = cross_entropy_loss(softmax(minus), labels)

            gradient[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    return gradient


def main():
    # Dua sampel, tiga kelas untuk memudahkan pemeriksaan.
    logits = np.array([
        [1.2, 0.3, -0.5],
        [0.1, 1.0, 0.2],
    ])

    labels = np.array([0, 2])

    probabilities = softmax(logits)
    analytical = analytical_gradient(logits, labels)
    numerical = numerical_gradient(logits, labels)

    np.set_printoptions(precision=6, suppress=True)

    print("Probabilitas Softmax:")
    print(probabilities)

    print("\nLabel sebenarnya:", labels)

    print("\nGradien analitis (P - Y) / m:")
    print(analytical)

    print("\nGradien numerik:")
    print(numerical)

    difference = np.max(np.abs(analytical - numerical))
    print(f"\nSelisih maksimum: {difference:.10f}")

    assert np.allclose(analytical, numerical, atol=1e-5)
    print("\nSemua pengujian gradien berhasil!")


if __name__ == "__main__":
    main()
