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
import re 
text = "i am great chef and i love to eat at 5 star hotel $foodie"
word = re.findall(r"[a-zA-Z0-9$]", text)
print(word)