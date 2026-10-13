import numpy as np


def loss_function(w):
    """Menghitung loss sederhana L(w) = w^2."""
    return w**2


def analytical_gradient(w):
    """Turunan matematis dari L(w) = w^2."""
    return 2 * w


def numerical_gradient(function, w, epsilon=1e-5):
    """Memperkirakan turunan menggunakan finite differences."""
    loss_plus = function(w + epsilon)
    loss_minus = function(w - epsilon)

    return (loss_plus - loss_minus) / (2 * epsilon)


def main():
    test_values = [-3.0, -1.0, 0.5, 2.0, 4.0]

    print("Menguji gradien untuk L(w) = w^2\n")
    print(f"{'w':>8} {'Loss':>10} {'Analitis':>12} {'Numerik':>12}")

    for w in test_values:
        loss = loss_function(w)
        analytical = analytical_gradient(w)
        numerical = numerical_gradient(loss_function, w)

        print(
            f"{w:8.2f} {loss:10.4f} "
            f"{analytical:12.6f} {numerical:12.6f}"
        )

        assert np.isclose(
            analytical,
            numerical,
            atol=1e-5,
        )

    print("\nSemua pengujian gradien berhasil!")


if __name__ == "__main__":
    main()
