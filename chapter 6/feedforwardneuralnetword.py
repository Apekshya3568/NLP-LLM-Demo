# Step 1: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Step 2: Example Data
sentences = ["I love Python", "I hate bugs", "Python is fun", "Bugs are annoying"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Step 3: Build Vocabulary
vocab = list(set(" ".join(sentences).lower().split()))
word2idx = {word: i for i, word in enumerate(vocab)}

# Step 4: One-Hot Encode Sentences
def one_hot_encode(sentence):
    vec = [0] * len(vocab)
    for word in sentence.lower().split():
        vec[word2idx[word]] = 1
    return vec

X = [one_hot_encode(s) for s in sentences]
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

# Step 5: Define Simple Feedforward Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # 2 classes: positive or negative

    def forward(self, x):
        return self.fc(x)

model = SimpleNN(len(vocab))

# Step 6: Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step 7: Training Loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Step 8: Test on New Sentences
test_sentences = ["I love bugs", "Python is annoying"]
X_test = torch.tensor([one_hot_encode(s) for s in test_sentences], dtype=torch.float)

with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)
    print(list(zip(test_sentences, predictions.tolist())))
