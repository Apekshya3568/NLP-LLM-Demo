"""import numpy as np

# Sigmoid and derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

# Training example
x = np.array([[1], [2]])  # column vector
y = np.array([[1]])

# Initialize weights
W1 = np.array([[0.1, 0.2],
               [0.3, 0.4]])

b1 = np.array([[0.01, 0.02]])

W2 = np.array([[0.5],
               [0.6]])

b2 = np.array([[0.03]])

lr = 0.1  # learning rate

for epoch in range(20):

    ### ---- FORWARD ---- ###
    z1 = W1.T @ x + b1.T
    h = sigmoid(z1)

    z2 = W2.T @ h + b2
    y_hat = sigmoid(z2)

    # Binary cross entropy
    loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

    ### ---- BACKWARD ---- ###
    dL_dz2 = y_hat - y            # output layer gradient
    dL_dW2 = h @ dL_dz2.T         # (2x1)
    dL_db2 = dL_dz2

    dL_dh = W2 @ dL_dz2
    dL_dz1 = dL_dh * sigmoid_deriv(h)

    dL_dW1 = x @ dL_dz1.T
    dL_db1 = dL_dz1.T

    ### ---- UPDATE ---- ###
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2

    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss = {loss[0][0]:.4f}")

"""


import numpy as np

# -----------------------------
# 1. Activation functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is the OUTPUT of sigmoid
    return x * (1 - x)

# -----------------------------
# 2. Training data (XOR)
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# -----------------------------
# 3. Initialize weights
# -----------------------------
np.random.seed(0)
W1 = np.random.randn(2, 2)     # shape: (input, hidden)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)     # shape: (hidden, output)
b2 = np.zeros((1, 1))

# learning rate
lr = 0.1

# -----------------------------
# 4. Training loop
# -----------------------------
for epoch in range(20000):
    
    # -------------------------
    # FORWARD PASS
    # -------------------------
    z1 = X.dot(W1) + b1            # hidden layer pre-activation
    a1 = sigmoid(z1)               # hidden layer output

    z2 = a1.dot(W2) + b2           # output layer pre-activation
    a2 = sigmoid(z2)               # final prediction (y_hat)

    # -------------------------
    # LOSS (binary cross-entropy)
    # -------------------------
    m = y.shape[0]                 # number of samples
    loss = -np.mean(y*np.log(a2) + (1-y)*np.log(1-a2))

    # -------------------------
    # BACKPROP
    # -------------------------
    # dLoss/dz2
    dz2 = a2 - y                   # (y_hat - y)

    # dLoss/dW2
    dW2 = a1.T.dot(dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Backprop into hidden layer
    dz1 = dz2.dot(W2.T) * sigmoid_derivative(a1)

    dW1 = X.T.dot(dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # -------------------------
    # Update weights
    # -------------------------
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Print every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# 5. Final predictions
# -----------------------------
print("\nFinal predictions:")
print(a2.round())
