'''import numpy as np
from collections import defaultdict
import random
import math


# ---------------------------------------------------------
# 1. TEXT PREPROCESSING
# ---------------------------------------------------------

def tokenize_corpus(corpus):
    """
    Input: corpus as a list of sentences
    Output: tokenized corpus (list of list of words)
    """
    return [sentence.lower().split() for sentence in corpus]


# ---------------------------------------------------------
# 2. BUILD VOCABULARY
# ---------------------------------------------------------

def build_vocab(tokenized_corpus):
    """
    Build word → index and index → word dictionaries
    """
    word_freq = defaultdict(int)

    # Count frequency of each word
    for sentence in tokenized_corpus:
        for word in sentence:
            word_freq[word] += 1

    # Create vocabulary mappings
    word2idx = {word: idx for idx, (word, _) in enumerate(word_freq.items())}
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, word_freq


# ---------------------------------------------------------
# 3. GENERATE TRAINING PAIRS (SKIP-GRAM)
# ---------------------------------------------------------

def generate_training_pairs(tokenized_corpus, word2idx, window_size=2):
    """
    Generates (target, context) positive pairs
    """
    pairs = []

    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]

        for center_pos, word_idx in enumerate(indices):
            # context window
            start = max(0, center_pos - window_size)
            end = min(len(indices), center_pos + window_size + 1)

            for pos in range(start, end):
                if pos != center_pos:
                    context_idx = indices[pos]
                    pairs.append((word_idx, context_idx))

    return pairs


# ---------------------------------------------------------
# 4. NEGATIVE SAMPLING
# ---------------------------------------------------------

def get_negative_samples(true_context, vocab_size, num_samples=5):
    """
    Randomly selects words that are NOT the true context
    """
    negatives = []
    while len(negatives) < num_samples:
        neg = random.randint(0, vocab_size - 1)
        if neg != true_context:
            negatives.append(neg)
    return negatives


# ---------------------------------------------------------
# 5. WORD2VEC MODEL CLASS
# ---------------------------------------------------------

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=50, lr=0.025):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = lr

        # Initialize embeddings randomly
        self.W1 = np.random.uniform(-0.5/embedding_dim, 
                                     0.5/embedding_dim, 
                                     (vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-0.5/embedding_dim, 
                                     0.5/embedding_dim, 
                                     (embedding_dim, vocab_size))

    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Training step for ONE positive pair
    def train_pair(self, target_idx, context_idx, neg_samples):
        v_target = self.W1[target_idx]           # target embedding
        v_context = self.W2[:, context_idx]      # context embedding

        # Positive sample prediction (target, context)
        score_pos = self.sigmoid(np.dot(v_target, v_context))
        error_pos = 1 - score_pos  # positive label = 1

        # Gradient updates
        grad_target = error_pos * v_context
        grad_context = error_pos * v_target

        # Update positive weights
        self.W1[target_idx] += self.lr * grad_target
        self.W2[:, context_idx] += self.lr * grad_context

        # Negative samples training
        for neg_idx in neg_samples:
            v_neg = self.W2[:, neg_idx]

            score_neg = self.sigmoid(np.dot(v_target, v_neg))
            error_neg = 0 - score_neg  # negative label = 0

            grad_target_neg = error_neg * v_neg
            grad_neg = error_neg * v_target

            # Update
            self.W1[target_idx] += self.lr * grad_target_neg
            self.W2[:, neg_idx] += self.lr * grad_neg

    # Full training
    def train(self, training_pairs, epochs=5, neg_samples=5):
        for epoch in range(epochs):
            random.shuffle(training_pairs)
            for target_idx, context_idx in training_pairs:
                negatives = get_negative_samples(context_idx, 
                                                 self.vocab_size,
                                                 num_samples=neg_samples)
                self.train_pair(target_idx, context_idx, negatives)
            print(f"Epoch {epoch+1}/{epochs} completed")


# ---------------------------------------------------------
# 6. COSINE SIMILARITY
# ---------------------------------------------------------

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ---------------------------------------------------------
# 7. DEMO — TRAIN WORD2VEC
# ---------------------------------------------------------

corpus = [
    "he likes fruits",
    "he likes apples",
    "she hates apples",
    "apples are fruits"
]

# 1. Tokenize
tokenized = tokenize_corpus(corpus)

# 2. Build vocabulary
word2idx, idx2word, freq = build_vocab(tokenized)
vocab_size = len(word2idx)

# 3. Generate skip-gram training pairs
pairs = generate_training_pairs(tokenized, word2idx, window_size=2)

# 4. Train Word2Vec
w2v = Word2Vec(vocab_size, embedding_dim=20)
w2v.train(pairs, epochs=50, neg_samples=4)

# 5. Test Similarities
v_apples = w2v.W1[word2idx["apples"]]
v_fruits = w2v.W1[word2idx["fruits"]]
v_likes = w2v.W1[word2idx["likes"]]

print("\nCosine similarity results:")
print("apples vs fruits:", cosine_similarity(v_apples, v_fruits))
print("apples vs likes:", cosine_similarity(v_apples, v_likes))'''




# second methos
'''import numpy as np
import random

# -----------------------------
# 1. Simple corpus
# -----------------------------
corpus = [
    "he likes fruits",
    "he likes apples",
    "she hates apples",
    "apples are fruits"
]

# -----------------------------
# 2. Tokenize and build vocab
# -----------------------------
words = set(word for sentence in corpus for word in sentence.lower().split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

# -----------------------------
# 3. Generate skip-gram pairs
# -----------------------------
window_size = 1
pairs = []

for sentence in corpus:
    tokens = sentence.lower().split()
    for center_pos, word in enumerate(tokens):
        center_idx = word2idx[word]
        start = max(0, center_pos - window_size)
        end = min(len(tokens), center_pos + window_size + 1)
        for pos in range(start, end):
            if pos != center_pos:
                context_idx = word2idx[tokens[pos]]
                pairs.append((center_idx, context_idx))

# -----------------------------
# 4. Simple embeddings
# -----------------------------
embedding_dim = 5
embeddings = np.random.rand(vocab_size, embedding_dim)

# -----------------------------
# 5. Very simple training loop
# -----------------------------
learning_rate = 0.1

for epoch in range(100):
    for center_idx, context_idx in pairs:
        # Dot product (prediction)
        pred = np.dot(embeddings[center_idx], embeddings[context_idx])
        # Simple error: assume positive pair should be 1
        error = 1 - pred
        # Gradient update
        embeddings[center_idx] += learning_rate * error * embeddings[context_idx]
        embeddings[context_idx] += learning_rate * error * embeddings[center_idx]

# -----------------------------
# 6. Cosine similarity
# -----------------------------
def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

v_apples = embeddings[word2idx["apples"]]
v_fruits = embeddings[word2idx["fruits"]]
v_likes = embeddings[word2idx["likes"]]

print("apples vs fruits:", cosine_sim(v_apples, v_fruits))
print("apples vs likes:", cosine_sim(v_apples, v_likes))
'''

import numpy as np
import random

# -----------------------------------
# 1. Sigmoid function
# -----------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------------------
# 2. Training step for one (w, cpos)
# -----------------------------------
def train_step(w, c_pos, c_negs, lr=0.1):

    # ---- Positive pair ----
    dot_pos = np.dot(c_pos, w)
    p_pos = sigmoid(dot_pos)
    grad_pos = (p_pos - 1)

    # ---- Negative pairs ----
    grad_negs = []
    for c_neg in c_negs:
        dot_neg = np.dot(c_neg, w)
        p_neg = sigmoid(dot_neg)
        grad_negs.append(p_neg)

    # ---- Update context positive ----
    c_pos -= lr * grad_pos * w

    # ---- Update context negatives ----
    for i in range(len(c_negs)):
        c_negs[i] -= lr * grad_negs[i] * w

    # ---- Update target word ----
    w -= lr * (grad_pos * c_pos + sum(grad_negs[i] * c_negs[i] for i in range(len(c_negs))))

    return w, c_pos, c_negs

# -----------------------------------
# 3. Example usage
# -----------------------------------

# initial embeddings
w       = np.array([1.0, 2.0])
c_pos   = np.array([1.5, -0.5])
c_negs  = [np.array([-1.0, 1.0]), np.array([0.5, -1.5])]

print("Before training:")
print("w:", w)
print("c_pos:", c_pos)
print("c_negs:", c_negs)

w, c_pos, c_negs = train_step(w, c_pos, c_negs)

print("\nAfter 1 training step:")
print("w:", w)
print("c_pos:", c_pos)
for i, c in enumerate(c_negs):
    print(f"c_neg{i+1}:", c)
