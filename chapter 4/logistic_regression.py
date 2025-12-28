import numpy as np

# Feature matrix (3 examples, 2 features)
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

# Weights (2 features → 2 weights)
w = np.array([[0.5], [-0.25]])

# Bias term
b = 1  

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 1: Linear part
z = X.dot(w) + b  # Automatically broadcasts bias to all rows

# Step 2: Apply sigmoid
y_hat = sigmoid(z)

print("Xw + b =\n", z)
print("\nPredicted probabilities (ŷ):\n", y_hat)
