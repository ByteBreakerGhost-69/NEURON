# NEURON — Experiment Log

## Experiment 001: NumPy Neural Network Baseline

### Objective
Build and evaluate a handwritten digit classifier using a neural
network implemented manually with NumPy.

### Dataset
- Dataset: scikit-learn Digits
- Total samples: 1,797
- Input image: 8 × 8 grayscale pixels
- Input features: 64
- Classes: digits 0 through 9
- Pixel values: normalized from [0, 16] to [0, 1]
- Train samples: 1,149
- Validation samples: 288
- Test samples: 360
- Split: stratified, with random seed 42

### Architecture
- Input layer: 64 features
- Hidden layer: 32 neurons
- Hidden activation: ReLU
- Output layer: 10 neurons
- Output activation: Softmax
- Loss: categorical cross-entropy
- Optimization: mini-batch gradient descent
- Core model implementation: NumPy

### Hyperparameters
- Epochs: 100
- Learning rate: 0.1
- Batch size: 32
- Initialization and data-order seeds: 42

### Test Results
- Accuracy: 96.1111%
- Macro precision: 96.2296%
- Macro recall: 96.0789%
- Macro F1-score: 96.1020%
- Correct predictions: 346 / 360
- Incorrect predictions: 14 / 360

### Error Analysis
The most difficult class on this test split was digit 8:
30 of 35 examples were classified correctly (recall 85.71%).

Observed confusions included:
- Digit 8 predicted as 1: 4 examples
- Digit 8 predicted as 5: 1 example
- Digit 1 predicted as 8: 3 examples
- Digit 9 predicted as 4: 2 examples

One incorrect prediction assigned 99.47% confidence to digit 1
when the actual label was digit 8. This illustrates that a high
Softmax probability is not a guarantee of correctness.

### Learning Curves
Training accuracy reached 99.74% at epoch 100, while validation
accuracy was 97.22%. Training loss continued to decrease, while
validation loss fluctuated and ended at 0.0983.

This indicates a generalization gap and possible overfitting.
It does not, by itself, establish severe overfitting.

### Artifacts
- results/accuracy_curve.png
- results/loss_curve.png
- results/confusion_matrix.png
- results/misclassified_examples.png
- results/training_history.npz
- results/test_metrics.json

### Interpretation and Limitations
The model is a baseline, not a production-ready handwriting
recognition system. The dataset contains small 8 × 8 images, which
differ substantially from free-form drawings on a web canvas.

The test set should remain reserved for final evaluation.
Hyperparameter selection and model improvements should use the
training and validation sets.

### Next Experiments
1. Inspect the incorrectly classified images.
2. Compare the baseline with early stopping based on validation loss.
3. Compare hyperparameters one at a time using the same data split.
4. Add tests and reproducible experiment records.
5. Build a web interface for drawing and predicting digits.
