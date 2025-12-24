# An N-gram model predicts the next word using the previous (N−1) words.
# If N = 1 → Unigram (uses no history)
# If N = 2 → Bigram ( uses 1 previous word)
# If N = 3 → Trigram ( uses 2 previous words)
# If N = 4 → 4-gram (uses 3 previous words)
#Bigram
"""from collections import defaultdict

text = "the water of walden pond is clear and the water of walden pond is deep"

# Tokenize
words = text.split()

# Create bigram counts
bigram_counts = defaultdict(int)
print(bigram_counts)
word_counts = defaultdict(int)

for w1, w2 in zip(words[:-1], words[1:]):
    bigram_counts[(w1, w2)] += 1
    word_counts[w1] += 1

# Function to compute bigram probability
def bigram_prob(w1, w2):
    return bigram_counts[(w1, w2)] / word_counts[w1]

# Example prediction
print("P(pond | walden) =", bigram_prob("walden", "pond"))
print("P(clear | is) =", bigram_prob("is", "clear"))"""


#trigram
"""from collections import defaultdict

text = "the water of walden pond is clear and the water of walden pond is deep"

# Tokenize
words = text.split()

# Create trigram + bigram counts
trigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for w1, w2, w3 in zip(words[:-2], words[1:-1], words[2:]):
    trigram_counts[(w1, w2, w3)] += 1
    print(f"Trigram: ({w1}, {w2}, {w3}) Count: {trigram_counts[(w1, w2, w3)]}")
    bigram_counts[(w1, w2)] += 1

# Probability function: P(w3 | w1, w2)
def trigram_prob(w1, w2, w3):
    print(f"trigram_counts {w1,w2,w3} : bigram_counts {w1,w2}")
    return trigram_counts[(w1, w2, w3)] / bigram_counts[(w1, w2)]


# Predict next word given history
def predict_next(w1, w2):
    candidates = {}
    for (a, b, c), count in trigram_counts.items():
        if a == w1 and b == w2:
            candidates[c] = trigram_prob(w1, w2, c)
    return sorted(candidates.items(), key=lambda x: -x[1])

# Try predicting
print("P(pond | of walden) =", trigram_prob("of", "walden", "pond"))
print("Predictions after 'walden pond':")
print(predict_next("walden", "pond"))"""


# -------------------------------------------------------------------------------------------------------------------------------

# word ocunt example 
'''import nltk


from nltk import word_tokenize
from collections import Counter

corpus = """
I want to eat Chinese food
I want English food
I want to find Chinese restaurants
Chinese restaurants serve good food
"""

# tokenize
tokens = word_tokenize(corpus.lower())

# generate bigrams
bigrams = list(nltk.bigrams(tokens))

# count bigrams
bigram_counts = Counter(bigrams)

print("Bigram Counts:")
for pair, count in bigram_counts.items():
    print(pair, ":", count)



from math import prod

def bigram_prob(tokens, bigram_counts):
    probs = []
    for w1, w2 in nltk.bigrams(tokens):
        num = bigram_counts[(w1, w2)]
        den = sum(count for (x, _), count in bigram_counts.items() if x == w1)
        prob = num / den if den > 0 else 0
        probs.append(prob)
    return prod(probs)

sentence = "i want chinese food"
tokens = word_tokenize(sentence.lower())
print("Probability:", bigram_prob(tokens, bigram_counts))


import matplotlib.pyplot as plt

pairs = [" ".join(b) for b,c in bigram_counts.items()]
values = [c for b,c in bigram_counts.items()]

plt.barh(pairs, values)
plt.title("Bigram Frequency Visualization")
plt.show()'''
# --------------------------------------------------------------------------------------




'''import nltk
from nltk import word_tokenize
from collections import Counter
import math

# Mini corpus
corpus = """
I want to eat Chinese food
I want English food
I want to find Chinese restaurants
Chinese restaurants serve good food
"""

# Tokenize and add start/end pseudo-words
sentences = [ ["<s>"] + word_tokenize(s.lower()) + ["</s>"] for s in corpus.strip().split("\n") ]
tokens = [token for sentence in sentences for token in sentence]

# Bigram counts
bigrams = list(nltk.bigrams(tokens))
bigram_counts = Counter(bigrams)

# Unigram counts
unigram_counts = Counter(tokens)

# Vocabulary size
V = len(set(tokens))

# Function to compute log probability of a sentence
def bigram_log_prob(sentence, bigram_counts, unigram_counts, V, smoothing=False):
    sentence_tokens = ["<s>"] + word_tokenize(sentence.lower()) + ["</s>"]
    log_prob = 0.0
    for w1, w2 in nltk.bigrams(sentence_tokens):
        count_bigram = bigram_counts.get((w1, w2), 0)
        count_unigram = unigram_counts.get(w1, 0)
        
        if smoothing:
            # Laplace smoothing
            prob = (count_bigram + 1) / (count_unigram + V)
        else:
            # Maximum Likelihood Estimate (no smoothing)
            if count_unigram == 0:
                prob = 0
            else:
                prob = count_bigram / count_unigram
        
        if prob == 0:
            # log(0) is undefined, use very small value
            log_prob += -float('inf')
        else:
            log_prob += math.log(prob)  # natural log
    
    return log_prob

# Example sentences
sent1 = "I want Chinese food"
sent2 = "I want Italian food"

# Compute log probabilities without and with smoothing
print("Sentence 1 log-prob (no smoothing):", bigram_log_prob(sent1, bigram_counts, unigram_counts, V, smoothing=False))
print("Sentence 1 log-prob (with smoothing):", bigram_log_prob(sent1, bigram_counts, unigram_counts, V, smoothing=True))

print("Sentence 2 log-prob (no smoothing):", bigram_log_prob(sent2, bigram_counts, unigram_counts, V, smoothing=False))
print("Sentence 2 log-prob (with smoothing):", bigram_log_prob(sent2, bigram_counts, unigram_counts, V, smoothing=True))'''
# ----------------------------------------------------------------------------------------------------------------------




# Sample mini corpus
# Setup a Mini Corpus
'''corpus = [
    "<s> I want Chinese food </s>",
    "<s> I want English food </s>",
    "<s> I like Chinese food </s>"
]

# Tokenize corpus into list of words
tokenized_corpus = [sentence.split() for sentence in corpus]

# Flatten for unigram counts
all_words = [word for sentence in tokenized_corpus for word in sentence]


# Count Unigrams, Bigrams, Trigrams
from collections import Counter
from itertools import tee

# Unigram counts
unigrams = Counter(all_words)

# Function to make n-grams
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Bigram counts
bigrams = Counter([bg for sentence in tokenized_corpus for bg in ngrams(sentence, 2)])

# Trigram counts
trigrams = Counter([tg for sentence in tokenized_corpus for tg in ngrams(sentence, 3)])

# Print counts
print("Unigrams:", unigrams)
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)


# Compute N-gram Probabilities
# Unigram probabilities
total_words = sum(unigrams.values())
unigram_probs = {word: count/total_words for word, count in unigrams.items()}

# Bigram probabilities P(w2|w1) = count(w1 w2) / count(w1)
bigram_probs = {bg: count/unigrams[bg[0]] for bg, count in bigrams.items()}

# Trigram probabilities P(w3|w1 w2) = count(w1 w2 w3) / count(w1 w2)
trigram_probs = {tg: count/bigrams[(tg[0], tg[1])] for tg, count in trigrams.items()}

print("\nUnigram Probabilities:", unigram_probs)
print("\nBigram Probabilities:", bigram_probs)
print("\nTrigram Probabilities:", trigram_probs)

# Compute Sentence Probability
import math

sentence = "<s> I want Chinese food </s>".split()

# Function to compute log probability for any n-gram
def sentence_log_prob(sentence, ngram_probs, n=2):
    probs = []
    if n == 1:
        for word in sentence:
            probs.append(math.log(ngram_probs.get(word, 1e-6)))
    else:
        for i in range(len(sentence)-n+1):
            ng = tuple(sentence[i:i+n])
            probs.append(math.log(ngram_probs.get(ng, 1e-6)))  # small value for unseen
    return sum(probs)

# Example: Bigram probability
log_prob_bigram = sentence_log_prob(sentence, bigram_probs, n=2)
print("Log Probability (Bigram):", log_prob_bigram)




# Compute Perplexity
def perplexity(sentence, ngram_probs, n=2):
    N = len(sentence) if n==1 else len(sentence)-n+1
    log_prob = sentence_log_prob(sentence, ngram_probs, n)
    return math.exp(-log_prob/N)

pp_unigram = perplexity(sentence, unigram_probs, n=1)
pp_bigram = perplexity(sentence, bigram_probs, n=2)
pp_trigram = perplexity(sentence, trigram_probs, n=3)

print(f"Perplexity (Unigram): {pp_unigram:.2f}")
print(f"Perplexity (Bigram): {pp_bigram:.2f}")
print(f"Perplexity (Trigram): {pp_trigram:.2f}")



# Visual Comparison
import matplotlib.pyplot as plt

models = ['Unigram', 'Bigram', 'Trigram']
pp_values = [pp_unigram, pp_bigram, pp_trigram]

plt.bar(models, pp_values, color=['skyblue','orange','green'])
plt.ylabel('Perplexity')
plt.title('Comparison of N-gram Models')
plt.show()'''



#here the , a , of  apppars more than the however and polyphonic has very less probability
"""import random

words = ["the", "of", "a", "to", "in", "however", "polyphonic"]
probs = [0.06, 0.03, 0.02, 0.02, 0.02, 0.0003, 0.0000018]

def sample_unigram(words, probs, n=10):
    return random.choices(words, weights=probs, k=n)

print(sample_unigram(words, probs, 20))"""


# full chapter 3 summary
# Step 1: Setup and tokenization
import random
import math
from collections import defaultdict, Counter

# Example training corpus
corpus = [
    "I want to eat lunch",
    "I want to eat Chinese food",
    "I want to drink water",
    "I like to eat cake",
    "She likes to eat lunch"
]

# Tokenize corpus
tokenized_corpus = [sentence.lower().split() for sentence in corpus]


# Step 2: Build n-gram counts
def build_ngrams(tokenized_corpus, n=2):
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    
    for sentence in tokenized_corpus:
        tokens = ["<s>"] * (n-1) + sentence + ["</s>"]
        for i in range(len(tokens)-n+1):
            ngram = tuple(tokens[i:i+n])
            context = tuple(tokens[i:i+n-1])
            ngram_counts[ngram] += 1
            context_counts[context] += 1
            
    return ngram_counts, context_counts

# Build bigram and trigram
bigram_counts, bigram_contexts = build_ngrams(tokenized_corpus, n=2)
trigram_counts, trigram_contexts = build_ngrams(tokenized_corpus, n=3)

# Step 3: Convert counts → probabilities (with add-1 smoothing)
def ngram_prob(ngram, ngram_counts, context_counts, vocab_size, k=1):
    context = ngram[:-1]
    count = ngram_counts.get(ngram, 0)
    context_count = context_counts.get(context, 0)
    return (count + k) / (context_count + k * vocab_size)

# Vocabulary
vocab = set(word for sentence in tokenized_corpus for word in sentence) | {"<s>", "</s>"}
V = len(vocab)


# Step 4: Sample a sentence from bigram model
def sample_sentence_bigram(ngram_counts, context_counts, vocab, max_len=10):
    sentence = ["<s>"]
    while len(sentence) < max_len:
        context = tuple([sentence[-1]])
        probs = []
        words = []
        for w in vocab:
            ngram = context + (w,)
            p = ngram_prob(ngram, ngram_counts, context_counts, len(vocab))
            probs.append(p)
            words.append(w)
        next_word = random.choices(words, weights=probs)[0]
        if next_word == "</s>":
            break
        sentence.append(next_word)
    return " ".join(sentence[1:])

# Example
for _ in range(5):
    print(sample_sentence_bigram(bigram_counts, bigram_contexts, vocab))


# Step 5: Compute perplexity
def perplexity(sentence, ngram_counts, context_counts, vocab, n=2, k=1):
    tokens = ["<s>"]*(n-1) + sentence.lower().split() + ["</s>"]
    N = len(tokens) - (n-1)
    log_prob = 0
    for i in range(len(tokens)-n+1):
        ngram = tuple(tokens[i:i+n])
        p = ngram_prob(ngram, ngram_counts, context_counts, len(vocab), k)
        log_prob += math.log2(p)
    return 2 ** (-log_prob / N)

# Test sentence
test_sentence = "I want to eat lunch"
pp = perplexity(test_sentence, bigram_counts, bigram_contexts, vocab)
print("Perplexity:", pp)


