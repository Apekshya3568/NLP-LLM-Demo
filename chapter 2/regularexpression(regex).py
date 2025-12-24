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
"""import re
text = "Column 1    Column 2    Column 3"

# Without parentheses
pattern_simple = r"Column [0-9]+ *"
print(re.findall(pattern_simple, text))
# Matches only individual columns with spaces

# With parentheses to repeat whole sequence
pattern_group = r"(Column [0-9]+ +)*"
print(re.findall(pattern_group, text))
"""
# Quantifiers (*, +) apply to previous token only

# Parentheses allow repeating the whole sequence



# Greedy vs Non-Greedy
"""import re
text = "<div>Hello</div><div>World</div>"

# Greedy: matches everything between first < and last >
pattern_greedy = r"<.*>"
print(re.findall(pattern_greedy, text))  
# Output: ['<div>Hello</div><div>World</div>']

# Non-Greedy: matches smallest possible
pattern_non_greedy = r"<.*?>"
print(re.findall(pattern_non_greedy, text))  
# Output: ['<div>', '</div>', '<div>', '</div>']"""


# Combining Everything: Match "the" as a whole word anywhere
"""import re
text = "The cat and the dog are playing."

pattern = r"\b[Tt]he\b"
matches = re.findall(pattern, text)
print(matches)  # ['The', 'the']

"""


#  1 \d → match digits
"""import re

text = "Party of 5 people"

result = re.findall(r"\d", text) #\d searched text and found the digit 5.
print(result)  #['5']"""

#  \D → match NON-digits
"""import re

text = "Room 101A"

result = re.findall(r"\D", text)
print(result)  #['R', 'o', 'o', 'm', ' ', 'A']
# 101 removed — remaining characters are not digits."""

#  \w → match alphanumeric + underscore
"""import re

text = "Daiyu_12!!"

result = re.findall(r"\w", text)
print(result)  #['D', 'a', 'i', 'y', 'u', '_', '1', '2']
#  Symbols like ! are NOT included."""


# 4\W → match NON-alphanumeric
"""import re

text = "Hello!!! 123"

result = re.findall(r"\W", text)
print(result)   ['!', '!', '!', ' ']
# Punctuation + space matched."""


# 5️ \s → match whitespace
"""import re

text = "Hello world\tPython\nRocks"

result = re.findall(r"\s", text)
print(result)   [' ', '\t', '\n']
# space, tab, newline detected."""



# 6️ \S → match NON-whitespace
"""import re

text = "Hi!\n"

result = re.findall(r"\S", text)
print(result)  #['H', 'i', '!']
# newline removed.
"""
# SPECIAL CHARACTERS (ESCAPING)
# Example: match literal period .
"""import re

text = "Dr. Strange"

result = re.findall(r"\.", text)
print(result)   ['.']"""
#  Without \. the regex dot means “any character”.



# Example: match literal star *
"""import re

text = "A*B*C"

result = re.findall(r"\*", text)
print(result)  ['*', '*']
# Without escape, * means repetition.
"""
# Example: match literal question mark ?
"""import re

text = "Why? When? How?"

result = re.findall(r"\?", text)
print(result) """ #['?', '?', '?']

# NEWLINE + TAB



#  Example \n → newline
"""text = "Hello\nWorld"
print(text)
# Hello
# World"""

#  Example \t → tab
"""text = "Hello\tWorld"
print(text)
# Hello   World"""



# Substitutions: re.sub()
import re

"""text = "I love cherry pie"
result = re.sub(r"cherry", r"apricot", text)
print(result)  #I love apricot pie"""



# Change name formatting
"""import re

text = "janet is here"
result = re.sub(r"janet", r"Janet", text)
print(result) #Janet is here
"""

# Capture Groups: (\ )
# US format → mm/dd/yyyy
# EU format → dd-mm-yyyy

"""import re

text = "The date is 10/15/2011"

result = re.sub(
    r"(\d{2})/(\d{2})/(\d{4})",
    r"\2-\1-\3",
    text
)

print(result)""" #The date is 15-10-2011


# Capture repeated word
"""import re

text = "This is is a problem"

result = re.findall(r"\b([A-Za-z]+)\s+\1\b", text)
print(result)  #['is']"""


# Pattern meaning:

# ([A-Za-z]+) → capture a word

# \s+ → space

# \1 → same word again



# Positive lookahead: (?= )

# Match only if pattern appears after current position.

# 🔹 Negative lookahead: (?! )

# Match only if pattern does NOT appear after current position.


"""import re

text = "apple pie, apple tart, banana split"

# Match "apple" only if it's followed by " pie"
pattern = r"apple(?= pie)"

result = re.findall(pattern, text)
print(result)  # ['apple']"""


"""import re

text = "apple pie, apple tart, banana split"

# Match "apple" only if it's NOT followed by " pie"
pattern = r"apple(?! pie)"

result = re.findall(pattern, text)
print(result)  # ['apple']"""



# Capture first word on a line — but only if it does NOT start with T or t.

'''import re

text = """
Tree
apple
Table
banana
top
cat
"""

result = re.findall(
    r"^(?![tT])(\w+)\b",
    text,
    flags=re.MULTILINE
)

print(result)  #['apple', 'banana', 'cat']'''


"""import re


# Problem 1: Two consecutive repeated words
pattern1 = r'\b([a-zA-Z]+)\s+\1\b'
test_strings1 = ["the the", "Humbert Humbert", "the bug", "the big bug"]

print("Problem 1:")
for s in test_strings1:
    if re.search(pattern1, s):
        print(f"Matched: {s}")
    else:
        print(f"Not Matched: {s}")
print("\n")"""


# Problem 2: Start with integer, end with word
"""pattern2 = r'^\d+.*[a-zA-Z]+$'
test_strings2 = ["123 this", "42 is the answer", "hello 123 world"]

print("Problem 2:")
for s in test_strings2:
    if re.match(pattern2, s):
        print(f"Matched: {s}")
    else:
        print(f"Not Matched: {s}")
print("\n")"""

# ------------------------------
# Problem 3: Contains both 'grotto' and 'raven'
"""pattern3 = r'\b(?=.*\bgrotto\b)(?=.*\braven\b).*'
test_strings3 = ["The grotto and the raven",
                 "raven flies over the grotto",
                 "grottos and the raven",
                 "raven is flying"]

print("Problem 3:")
for s in test_strings3:
    if re.search(pattern3, s):
        print(f"Matched: {s}")
    else:
        print(f"Not Matched: {s}")
print("\n")"""

# ------------------------------
"""# Problem 4: Capture first word of a sentence
pattern4 = r'^\s*["“]?([a-zA-Z]+)'
test_strings4 = ['Hello world.', ' “Once upon a time', 'Python is fun.']

print("Problem 4:")
for s in test_strings4:
    match = re.match(pattern4, s)
    if match:
        print(f"First word: {match.group(1)} in \"{s}\"")
    else:
        print(f"No match in: {s}")
"""

# this gave a error not an error but its output is none as 
# re.match(): tries to match the pattern only at the very beginning of the string.

# re.search(): searches for the pattern anywhere in the string.
"""import re
text = "The cat is cater and dog is doggy"
result = re.match(r"\bcat\b",text)
print(result)"""

"""import re
text = "The cat is cater and dog is doggy"
result = re.search(r"\bcat\b",text)
result1 = re.search(r"\bdog\b",text)
print(result) #<re.Match object; span=(4, 7), match='cat'> this is hte output of this 
print(result.group()) #result.group() gives the actual matched string.
print(result1)
print(result1.group())
"""

import re
text = "The cat is Scater and dog is Bdoggy"
result = re.search(r"\Bcat\B",text)
result1 = re.search(r"\Bdog\B",text)
print(result) #<re.Match object; span=(4, 7), match='cat'> this is hte output of this 
print(result.group()) #result.group() gives the actual matched string.
print(result1)
print(result1.group())