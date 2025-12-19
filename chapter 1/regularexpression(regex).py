#for word [a-zA-z]
"""import re
text = "I love programming in Python since 2025 and glad i star it #pythonprogrammer"
word = re.findall(r"[a-zA-Z]+", text)
print(word)
"""

# for numbers[0-9] as word + numbers [a-zA-Z0-9]
"""import re
text = " I am a happy person as i am 22 years old $"
word = re.findall(r"[a-zA-z0-9]+", text)
print(word)"""


#for special characters[#] for # [$] for $ and together [a-zA-z0-9#]
"""import re 
text = "i am great chef and i love to eat at 5 star hotel #foodie"
word = re.findall(r"[a-zA-Z0-9#]+", text)
print(word)"""


#  for doller $
"""import re 
text = "i am great chef and i love to eat at 5 star hotel $foodie"
word = re.findall(r"[a-zA-Z0-9$]+", text)
print(word)
"""

# here if we did not write + as word = re.findall(r"[a-zA-z0-9%]",text) this will give all the characters instead of words")
"""import re 
text = "i am great chef and i love to eat at 5 star hotel $foodie"
word = re.findall(r"[a-zA-Z0-9$]", text)
print(word)"""



"""import nltk
from nltk.tokenize import regexp_tokenize

text = "That U.S.A. poster-print costs $12.40..."
pattern = r'''(?x)               # verbose modenl
    (?:[A-Z]\.)+                # abbreviations, e.g. U.S.A.
  | \w+(?:-\w+)*                # words with optional hyphens
  | \$?\d+(?:\.\d+)?%?          # currency, percentages
  | \.\.\.                      # ellipsis
  | [][.,;"’?():_‘-]            # punctuation
'''

tokens = regexp_tokenize(text, pattern)
print(tokens)"""



'''
import re

# Sample text
text = """
Dr. Smith loves NLP! He bought 12 apples, 3 bananas, and $45.50 worth of oranges.
Visit https://www.stanford.edu or email him at dr.smith@stanford.edu.
What's your favorite fruit? I can't decide!
"""

# Step 1: Sentence Tokenization
sentence_pattern = r'(?<=[.!?])\s+'
sentences = re.split(sentence_pattern, text.strip())'''
# Sentence tokenization → Uses re.split() with the pattern (?<=[.!?])\s+ to split text into sentences.
# text.strip()
# Removes any leading and trailing whitespace from the string.
# Spaces, tabs \t, and newlines \n at the start and end are removed.
# It does not remove spaces inside the text.
# egtext = "   Hello world!  \n"
# print(text.strip()) output is Hello world!



# import re

# # Sample text
# text = """
# Dr. Smith loves NLP! He bought 12 apples, 3 bananas, and $45.50 worth of oranges.
# Visit https://www.stanford.edu or email him at dr.smith@stanford.edu.
# What's your favorite fruit? I can't decide!
# """

# # Step 1: Sentence Tokenization
# sentence_pattern = r'(?<=[.!?])\s+'
# sentences = re.split(sentence_pattern, text.strip())
# print(sentences)

# # Step 2: Word Tokenization
# word_pattern = r'''(?x)               # verbose mode
# (?:[A-Z]\.)+                        # abbreviations like U.S.A., Dr.
# | \w+(?:[-']\w+)*                    # words with optional internal hyphens or apostrophes
# | \$?\d+(?:\.\d+)?%?                  # numbers, currency, percentages
# | https?://[^\s]+                     # URLs
# | \.\.\.                               # ellipsis
# | [.,!?;]                             # punctuation
# '''

# print(word_pattern)
# # Apply word tokenization for each sentence
# for i, sentence in enumerate(sentences):
#     words = re.findall(word_pattern, sentence)
#     print(f"Sentence {i+1} tokens:", words)
#     print(i, sentence)
#     print(words)

# # Optional: flatten all tokens into a single list
# all_tokens = [token for sentence in sentences for token in re.findall(word_pattern, sentence)]
# print("\nAll tokens:", all_tokens)


# 1 Anchors (^ and $)
"""import re

text1 = "The dog is cute."
text2 = "A dog is cute."
text3 = "Hello you are cute.The cuties"

# Match "The" only at the start
pattern_start = r"^The"
print(re.findall(pattern_start, text1))  # ['The']
print(re.findall(pattern_start, text2))  # []
print(re.findall(pattern_start,text3))   # []


# Match "cute." only at the end \. needed to match literal period
pattern_end = r"cute\.$"
print(re.findall(pattern_end, text1))  # ['cute.']
print(re.findall(pattern_end, text2))  # ['cute.']
print(re.findall(pattern_end,text3))   #[]
"""


# 2️ Word boundaries (\b and \B)
"""import re
text = "The other 99 bottles cost $99 299 @99 2992 th99."

# Match the exact word "the" (case-insensitive)
pattern_word = r"\b[Tt]he\b"
print(re.findall(pattern_word, text))  # ['The']

# Match "99" as a separate word
pattern_number = r"\b99\b"
print(re.findall(pattern_number, text))  # ['99','99','99']

# Match "99" inside other numbers (non-word boundary)\B → non-word boundary: position inside a word, not at the start or end of a wor
pattern_non_word = r"\B99\B"
print(re.findall(pattern_non_word, text))  # ['99']"""


# Disjunction (OR) and Grouping
"""import re
text = "I have a cat, a dog, and some guppies."

# Match cat or dog
pattern_or = r"cat|dog"
print(re.findall(pattern_or, text))  # ['cat', 'dog']

# Match guppy or guppies using grouping
pattern_group = r"gupp(y|ies)"
print(re.findall(pattern_group, text))  # ['ies'] -> captures group content
# To match full word, use non-capturing group:
pattern_group_full = r"gupp(?:y|ies)"
print(re.findall(pattern_group_full, text))  # ['guppies']
"""
# | → OR operator

# (y|ies) → group for suffix alternatives

# (?:...) → non-capturing group (matches full word instead of just group content)




# Precedence with Quantifiers and Sequences
import re
text = "Column 1    Column 2    Column 3"

# Without parentheses
pattern_simple = r"Column [0-9]+ *"
print(re.findall(pattern_simple, text))
# Matches only individual columns with spaces

# With parentheses to repeat whole sequence
pattern_group = r"(Column [0-9]+ +)*"
print(re.findall(pattern_group, text))
