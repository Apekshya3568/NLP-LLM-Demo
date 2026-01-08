"""import numpy as np

# ---------------------------------------
# 1. Input: three tokens represented as vectors
# ---------------------------------------
# Each token is a 4-dim vector (random example)
tokens = np.array([
    [1, 0, 1, 0],   # token 1
    [0, 2, 0, 2],   # token 2
    [1, 1, 1, 1]    # token 3
])

# ---------------------------------------
# 2. Weight matrices for Q, K, V (4x4)
# (In real transformers these are learned)
# ---------------------------------------
W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)

# ---------------------------------------
# 3. Compute Q, K, V vectors
# ---------------------------------------
Q = tokens @ W_Q   # (3x4)
K = tokens @ W_K   # (3x4)
V = tokens @ W_V   # (3x4)

# ---------------------------------------
# 4. Compute attention scores = Q · K^T
# ---------------------------------------
scores = Q @ K.T   # shape (3x3) K.T → transpose of K

# If K is (3×4), then K.T becomes (4×3).
print("Attention scores:\n", scores)

# ---------------------------------------
# 5. Softmax for each row
# ---------------------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability trick
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)
print("\nAttention Weights (Softmax):\n", attention_weights)

# ---------------------------------------
# 6. Weighted sum of V vectors
# ---------------------------------------
output = attention_weights @ V
print("\nSelf-Attention Output:\n", output)
"""

"""import numpy as np

# Small example
X = np.array([
    [1, 0],  # "I"
    [0, 1]   # "eat"
])

# Weight matrices (identity for simplicity)
WQ = np.eye(2)
WK = np.eye(2)
WV = np.eye(2)

# Step 1: Compute Q, K, V
Q = X @ WQ
K = X @ WK
V = X @ WV

# Step 2: Compute attention scores
scores = Q @ K.T

# Step 3: Scale
dk = K.shape[-1]
scaled = scores / np.sqrt(dk)

# Step 4: Softmax
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=-1, keepdims=True)

weights = softmax(scaled)

# Step 5: Weighted sum of V
output = weights @ V

print("Attention weights:\n", weights)
print("Output contextual embeddings:\n", output)





# -------------------------------------------------------------
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Sample token embeddings
x = np.array([
    [1,0],   # x1
    [0,1],   # x2
    [1,1]    # x3
])

# Identity matrices to keep things simple
WQ = WK = WV = np.eye(2)
WO = np.eye(2)

# Compute Q, K, V
Q = x @ WQ
K = x @ WK
V = x @ WV

i = 2  # compute attention for token 3 (index 2)

# Step 1: scores
scores = (Q[i] @ K.T) / np.sqrt(2)

# Step 2: softmax
weights = softmax(scores)

# Step 3: weighted sum of values
head = weights @ V

# Step 4: output
a3 = head @ WO

print("Attention weights:", weights)
print("Output vector a3:", a3)"""




# ---------------------------------------------------------------------------------------------------------
"""import numpy as np

# ----- Step 1: Input (3 tokens, d=2) -----
X = np.array([
    [1, 2],
    [0, 1],
    [1, 0]
])

# ----- Step 2: Weights -----
WQ = np.array([[1, 0], [0, 1]])
WK = np.array([[1, 1], [1, 0]])
WV = np.array([[1, 2], [0, 1]])

# ----- Step 3: Compute Q,K,V -----
Q = X @ WQ
K = X @ WK
V = X @ WV

# ----- Step 4: Compute attention scores -----
scores = Q @ K.T

# ----- Step 5: Apply causal mask -----
mask = np.triu(np.ones_like(scores), k=1) * -1e9
masked_scores = scores + mask

# ----- Step 6: Softmax -----
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

A = softmax(masked_scores)

# ----- Step 7: Attention output -----
output = A @ V

print("Q=\n", Q)
print("K=\n", K)
print("V=\n", V)
print("Scores (QK^T)=\n", scores)
print("Masked=\n", masked_scores)
print("Attention Weights=\n", A)
print("Output=\n", output)"""
# --------------------------------------------------------------------------------------------------
"""import numpy as np

# embeddings (3 tokens, dim 2)
X = np.array([
    [1, 0],   # A
    [0, 1],   # B
    [1, 1]    # C
])

# weight matrices
WQ = np.array([[1, 0],
               [0, 1]])

WK = np.array([[1, 1],
               [1, 0]])

WV = np.array([[1, 2],
               [0, 1]])

# compute Q, K, V
Q = X @ WQ
K = X @ WK
V = X @ WV

print("Q =\n", Q)
print("K =\n", K)
print("V =\n", V)

# attention scores
scores = Q @ K.T
print("Scores =\n", scores)

# softmax function
def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

# compute attention weights
weights = np.apply_along_axis(softmax, 1, scores)
print("Attention weights =\n", weights)

# attention output
output = weights @ V
print("Final attention output =\n", output)
# --------------------------------------------"""
# --------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMHead(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.E = nn.Embedding(vocab_size, d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.unembed.weight = self.E.weight

    def forward(self, hidden):
        # hidden shape: (batch, d_model)
        logits = self.unembed(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs
