import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(np.float64)


def main():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 2.0])

    print("Nilai sebelum ReLU:")
    print(z)

    print("\nNilai setelah ReLU:")
    print(relu(z))

    print("\nTurunan ReLU:")
    print(relu_derivative(z))

    expected = np.array([0, 0, 0, 1, 1], dtype=np.float64)

    assert np.array_equal(relu_derivative(z), expected)
    print("\nSemua pengujian turunan ReLU berhasil!")


if __name__ == "__main__":
    main()
