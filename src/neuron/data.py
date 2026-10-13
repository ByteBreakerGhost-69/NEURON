"""Dataset loading for NEURON.

Currently implemented: loading scikit-learn's Digits dataset.

TODO (Stage 2): dataset inspection helpers.
TODO (Stage 2): preprocessing, e.g. scaling pixels from 0-16 to 0-1.
TODO (Stage 2): one-hot encoding of labels.
TODO (Stage 2): reproducible train/test split with a fixed random seed.
"""

import numpy as np
from sklearn.datasets import load_digits


def load_digits_dataset():
    """Load the scikit-learn Digits dataset.

    Returns:
        X: float array of shape (n_samples, 64). Each row is a flattened
           8 x 8 image. Pixel values are the raw ones (0 to 16);
           no preprocessing is applied here.
        y: integer array of shape (n_samples,) with labels 0 to 9.
    """
    dataset = load_digits()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)
    return X, y
