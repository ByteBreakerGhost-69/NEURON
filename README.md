# NEURON

**Handwritten Digit Recognition with Neural Networks from Scratch**

NEURON is an educational and experimental Machine Learning project. Its goal is to classify handwritten digits (0 to 9) using a neural network implemented manually with Python and NumPy.

> **Status: foundation only. The neural network is not implemented yet.**
> There is no model, no training loop, and no accuracy result in this repository. Nothing here is production-ready.

## Motivation

Frameworks such as PyTorch or TensorFlow make it easy to train a network without understanding what happens inside. NEURON takes the opposite route: every piece of the learning mechanics is written by hand, so the developer can understand, debug, and experiment with each step. An LLM is used as a mentor during development.

## Learning objectives

- Understand forward propagation as matrix operations.
- Implement ReLU and Softmax manually.
- Compute cross-entropy loss and understand what it measures.
- Derive and implement backpropagation.
- Implement gradient descent parameter updates.
- Evaluate a classifier honestly and analyze its errors.
- Run reproducible experiments and record results truthfully.

## Current project status

| Item | State |
|---|---|
| Project structure, README, requirements, `.gitignore` | Done |
| Dataset loader (`load_digits_dataset()` in `data.py`) | Done |
| Setup tests (`tests/test_project_setup.py`) | Written |
| Preprocessing, model, training, evaluation | **Not implemented** |

`model.py`, `train.py`, and `evaluate.py` currently contain only docstrings and TODOs.

## From-scratch policy

The neural network will be implemented **from scratch using NumPy**: matrix multiplication, activation functions, Softmax, loss, gradients, and parameter updates are all written by hand. No PyTorch, TensorFlow, or Keras is used for the core model. scikit-learn is used only to load the initial dataset and, later, for selected evaluation utilities.

## Planned neural network architecture

```
Input layer   : 64 features (flattened 8 x 8 image)
Hidden layer  : 32 neurons, ReLU activation
Output layer  : 10 neurons, Softmax activation
```

Training will use backpropagation and gradient descent. A first experiment may try 300 epochs with a learning rate of 0.5. These are starting points, not known-optimal values.

## Dataset

The initial dataset is scikit-learn's Digits dataset. Properties (checked against the installed dataset, not assumed):

- 1,797 samples
- 8 x 8 grayscale images, flattened to 64 features
- 10 classes (digits 0 to 9)
- Raw pixel intensities from 0 to 16

## Technology stack

- Python 3
- NumPy (numerical operations, the core of the model)
- scikit-learn (dataset loading, selected evaluation utilities)
- pytest (testing)
- Matplotlib (future experiment visualization)
- Git and GitHub (version control and hosting)

## Planned repository structure

```
neuron/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── src/
│   └── neuron/
│       ├── __init__.py
│       ├── data.py        # dataset loading (implemented), preprocessing (TODO)
│       ├── model.py       # TODO
│       ├── train.py       # TODO
│       └── evaluate.py    # TODO
├── tests/
│   └── test_project_setup.py
├── experiments/
│   └── README.md
└── results/
    └── .gitkeep
```

## Environment setup

Create and activate a virtual environment (do not install packages globally).

```bash
# from the repository root
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

## Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` uses lower-bound version pins so it works across a range of Python 3.9+ environments. Exact pins can be added later if cross-machine reproducibility needs them.

## Run the tests

```bash
pytest
```

`pytest.ini` adds `src/` to the import path, so the package does not need to be installed. The tests only verify the setup: package import, file structure, and loading of the real Digits dataset. They do not test any model, because none exists yet.

## Development roadmap

This roadmap is a **plan, not a list of completed features**. Only Stage 1 is in progress.

1. Project foundation and environment setup
2. Dataset inspection and preprocessing
3. Manual forward propagation using NumPy
4. ReLU and Softmax implementation
5. Loss calculation
6. Backpropagation and gradient computation
7. Gradient descent and model training
8. Model evaluation and error analysis
9. Reproducible experiments
10. Training visualization
11. Optional comparison against a suitable reference implementation
12. Portfolio documentation and project presentation

## Experiments and results

No experiments have been run and no results exist yet. This section will be filled in only with results actually produced by running the code. See `experiments/README.md` for the planned note-taking convention.

## Evaluation and Experiments

NEURON includes test-set evaluation, per-class classification metrics,
a confusion matrix, and visualization of incorrectly classified
samples.

To reproduce the evaluation artifacts:

```
PYTHONPATH=src python experiments/visualize_evaluation.py
```

Generated artifacts are stored in `results/`. The detailed baseline
configuration, metrics, limitations, and proposed next experiments
are documented in [`docs/experiment-log.md`](docs/experiment-log.md).

Run the automated tests with:

```
PYTHONPATH=src pytest -q
```
