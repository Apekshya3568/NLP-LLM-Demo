

"""from collections import Counter

# Step 1: Initial corpus split into characters
corpus = ["l o w", "l o w e r", "l o w e s t"]

# Step 2: Flatten corpus into list of tokens
tokens = [word.split() for word in corpus]  # [['l','o','w'], ['l','o','w','e','r'], ...]

# Function to count pairs
def get_pairs(tokens):
    pairs = Counter()
    for word in tokens:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

# Function to merge a pair
def merge_pair(tokens, pair_to_merge):
    new_tokens = []
    for word in tokens:
        new_word = []
        i = 0
        while i < len(word):
            # Check if the current and next token is the pair
            if i < len(word)-1 and (word[i], word[i+1]) == pair_to_merge:
                new_word.append(word[i]+word[i+1])  # merge
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_tokens.append(new_word)
    return new_tokens

# Step 3: Run a few merges
for _ in range(4):
    pairs = get_pairs(tokens)
    print("Current pairs and counts:", pairs)
    if not pairs:
        break
    most_frequent = pairs.most_common(1)[0][0]
    print("Merging pair:", most_frequent)
    tokens = merge_pair(tokens, most_frequent)
    print("Updated tokens:", tokens)
    print("-"*30)

                """

"""
from collections import Counter
list = [ "l o w" , "l o w e s t","L O W E R","L O W","l o w"]
my_list = ["low", "lower","loweres"]
Counter(list)
print(Counter(my_list))
print(Counter(list))
"""
# Counter is a special dictionary that automatically counts how many times each item appears.

"""my_list = ["low","lowest"]
tokens = [word.split() for word in my_list]
token = [list(word) for word in my_list]
print(tokens)
print(token)
"""
from collections import Counter
corpus = ["l o w", "l o w e r", "l o w e s t"]
tokens= [word.split() for word in corpus]
print(tokens)
def get_pairs(tokens):
    pairs = Counter()
    for word in tokens:
        print(word)
        for i in range(len(word)-1):
            print(len(word))
            pairs[(word[i], word[i+1])] += 1
            print(pairs)
    return pairs

get_pairs(tokens)