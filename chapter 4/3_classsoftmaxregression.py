import numpy as np

# Example input (4 features)
x = np.array([[1],
              [0.5],
              [2],
              [1.5]])

# 3 classes → weight matrix shape = (3, 4)
W = np.array([
    [0.2, 0.1, -0.5, 1.2],
    [-1.5, 2.0, 0.3, -0.3],
    [1.0, -0.8, 1.5, 0.5]
])

# Bias vector for 3 classes
b = np.array([[0.1],
              [0.2],
              [-0.1]])

# Compute logits
z = W @ x + b   # shape: (3,1)

# Softmax
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

y_hat = softmax(z)

print("Logits:\n", z)
print("\nProbabilities:\n", y_hat)
