import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # Load the original dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Normalize pixel values from [0, 16] to [0, 1]
    X_train = X_train / 16.0
    X_test = X_test / 16.0

    print("=== NEURON: Dataset Preparation ===")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Training shape   : {X_train.shape}")
    print(f"Testing shape    : {X_test.shape}")
    print(f"Training labels  : {y_train.shape}")
    print(f"Testing labels   : {y_test.shape}")

    print("\n=== Normalization Check ===")
    print(f"Training minimum: {X_train.min():.4f}")
    print(f"Training maximum: {X_train.max():.4f}")
    print(f"Testing minimum : {X_test.min():.4f}")
    print(f"Testing maximum : {X_test.max():.4f}")

    print("\n=== Class Distribution ===")
    for digit in range(10):
        train_count = np.sum(y_train == digit)
        test_count = np.sum(y_test == digit)
        print(
            f"Digit {digit}: "
            f"train={train_count}, test={test_count}"
        )

    # Verify important data properties
    assert X_train.shape[1] == 64
    assert X_test.shape[1] == 64
    assert len(X_train) + len(X_test) == len(X)
    assert X_train.min() >= 0 and X_train.max() <= 1
    assert X_test.min() >= 0 and X_test.max() <= 1

    print("\nAll dataset preparation checks passed.")


if __name__ == "__main__":
    main()
