"""Smoke tests for the NEURON project foundation.

These verify the setup only. No neural network functionality exists yet,
so nothing about the model is tested.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import neuron

    assert neuron.__version__


def test_package_structure():
    package_dir = ROOT / "src" / "neuron"
    for name in ["__init__.py", "data.py", "model.py", "train.py", "evaluate.py"]:
        assert (package_dir / name).is_file(), f"missing {name}"
    for name in ["README.md", "requirements.txt", ".gitignore", "pytest.ini"]:
        assert (ROOT / name).is_file(), f"missing {name}"
    assert (ROOT / "experiments" / "README.md").is_file()
    assert (ROOT / "results").is_dir()


def test_digits_dataset_shape():
    from neuron.data import load_digits_dataset

    X, y = load_digits_dataset()
    assert X.ndim == 2 and y.ndim == 1
    assert X.shape[1] == 64
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 1797  # known size of the scikit-learn Digits dataset


def test_digits_dataset_ten_classes():
    from neuron.data import load_digits_dataset

    _, y = load_digits_dataset()
    assert sorted(np.unique(y).tolist()) == list(range(10))


def test_digits_arrays_are_usable():
    from neuron.data import load_digits_dataset

    X, y = load_digits_dataset()
    assert np.issubdtype(X.dtype, np.floating)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.isfinite(X).all()
    assert X.min() >= 0 and X.max() <= 16  # raw pixel intensity range
