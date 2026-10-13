import numpy as np
from sklearn.datasets import load_digits


def main():
    # Load the handwritten digit dataset
    digits = load_digits()

    X = digits.data
    y = digits.target
    images = digits.images

    print("=== NEURON: Dataset Inspection ===")
    print(f"Samples       : {X.shape[0]}")
    print(f"Features      : {X.shape[1]}")
    print(f"Image shape   : {images.shape[1:]}")
    print(f"Classes       : {np.unique(y)}")
    print(f"Feature shape : {X.shape}")
    print(f"Label shape   : {y.shape}")
    print(f"Pixel minimum : {X.min()}")
    print(f"Pixel maximum : {X.max()}")

    # Inspect one image
    sample_index = 0

    print(f"\n=== Sample {sample_index} ===")
    print(f"Correct label: {y[sample_index]}")
    print("Pixel matrix:")
    print(images[sample_index])

    # Count samples for each digit
    print("\n=== Class Distribution ===")
    labels, counts = np.unique(y, return_counts=True)

    for label, count in zip(labels, counts):
        print(f"Digit {label}: {count} samples")


if __name__ == "__main__":
    main()
