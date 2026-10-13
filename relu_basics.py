import numpy as np


def relu(z):
    return np.maximum(0, z)


z = np.array([
    [-2.0, -0.5, 0.0, 1.88, 2.58]
])

a = relu(z)

print("Input Z:")
print(z)

print("\nAfter ReLU:")
print(a)

print("\nInput shape:", z.shape)
print("Output shape:", a.shape)

assert np.all(a >= 0)
assert np.array_equal(a, np.maximum(0, z))

print("\nAll ReLU checks passed.")
