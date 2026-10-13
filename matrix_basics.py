import numpy as np

# One sample with four features
x = np.array([[0.2, 0.5, 0.8, 1.0]])

# Four input features connected to three neurons
W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
])

b = np.array([0.1, 0.2, 0.3])

Z = x @ W + b

print("Input shape:", x.shape)
print("Weight shape:", W.shape)
print("Bias shape:", b.shape)
print("Output shape:", Z.shape)
print("Output values:", Z)
