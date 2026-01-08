"""# -------------------------------
# 1. Import libraries
# -------------------------------
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# -------------------------------
# 2. Load pretrained BERT tokenizer and model
# -------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# -------------------------------
# 3. Prepare your dataset
# -------------------------------
texts = ["I love this movie", "I hate this movie"]  # Example sentences
labels = [1, 0]  # 1=positive, 0=negative

# Convert text to numbers (tokenization)
encodings = tokenizer(texts, truncation=True, padding=True)

# -------------------------------
# 4. Create PyTorch Dataset class
# -------------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Convert input_ids & attention_mask to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add labels
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset
dataset = Dataset(encodings, labels)

# -------------------------------
# 5. Set training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',     # Where model predictions and checkpoints are saved
    num_train_epochs=3,         # Train for 3 epochs
    per_device_train_batch_size=2,  # Batch size
    logging_dir='./logs',       # Directory for logs
    logging_steps=1,            # Log every step
    save_steps=10,              # Save checkpoint every 10 steps
    evaluation_strategy="no"    # No evaluation for now
)

# -------------------------------
# 6. Create Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -------------------------------
# 7. Train the model
# -------------------------------
trainer.train()

# -------------------------------
# 8. (Optional) Save the trained model
# -------------------------------
model.save_pretrained("./trained_bert_model")
tokenizer.save_pretrained("./trained_bert_model")

print("Training complete and model saved!")
"""



# ============================
# 1. Install dependencies
# ============================
# !pip install transformers torch sklearn

# ============================
# 2. Import libraries
# ============================
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# 3. Load BERT base (uncased)
# ============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased",
                                  output_hidden_states=True)
model.eval()


# ============================
# 4. Function: Extract full embeddings
# ============================
def get_embeddings(sentence):
    # Tokenize sentence
    tokens = tokenizer(sentence, return_tensors='pt')

    # Pass through BERT
    with torch.no_grad():
        outputs = model(**tokens)

    hidden_states = outputs.hidden_states  # tuple: 13 layers
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    return tokens_decoded, hidden_states


# ============================
# 5. Function: Extract a specific word's embedding
# ============================
def get_word_embedding(tokens, hidden_states, target_word, layer=-1):
    # Find all indices of the word
    indices = [i for i, tok in enumerate(tokens) if tok == target_word]

    if not indices:
        raise ValueError(f"'{target_word}' not found in token list: {tokens}")

    idx = indices[0]  # take first occurrence

    vector = hidden_states[layer][0][idx]  # shape: [768]
    return vector


# ============================
# 6. Example embeddings for one sentence
# ============================
sentence = "The bank of the river was flooded."
tokens, hidden_states = get_embeddings(sentence)

print("Tokens:", tokens)
vec_bank = get_word_embedding(tokens, hidden_states, "bank")
print("\nEmbedding of 'bank':")
print("Shape:", vec_bank.shape)
print(vec_bank[:10])  # first 10 values


# ============================
# 7. Compare 2 different meanings of “bank”
# ============================
sentence1 = "I deposited money in the bank."
sentence2 = "The fisherman sat on the bank of the river."

tokens1, hs1 = get_embeddings(sentence1)
tokens2, hs2 = get_embeddings(sentence2)

vec_bank1 = get_word_embedding(tokens1, hs1, "bank")
vec_bank2 = get_word_embedding(tokens2, hs2, "bank")

v1 = vec_bank1.numpy().reshape(1, -1)
v2 = vec_bank2.numpy().reshape(1, -1)

similarity = cosine_similarity(v1, v2)[0][0]
print("\nCosine similarity between two meanings of 'bank':", similarity)


# ============================
# 8. Function: Full similarity matrix (optional)
# ============================
def similarity_matrix(tokens, hidden_states, layer=-1):
    embeddings = hidden_states[layer][0].numpy()
    sim = cosine_similarity(embeddings)
    print("\nSimilarity matrix shape:", sim.shape)
    return sim


# Generate similarity matrix
sim_mat = similarity_matrix(tokens, hidden_states)
