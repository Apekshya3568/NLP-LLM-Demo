"""import re
email = "content@gmail.com"
match = re.search(r"\S+@\S+\.\S+", email)
print(match.group())"""  #returns the exact part of the text that matched the pattern. Since the whole email matches, the output is: 

"""import re
email = "Hello my email is content@gmail.com"
match = re.search(r"\S+@\S+\.\S+", email)
print(match.group())""" #as this returrns only content@gmail.com not the hwllo and other so this is why gropup

"""import re
email = "Hello my email is aap.lamichhane123@gmail.com"
match = re.search(r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}",email)
print(match.group())"""

import re
email = "aap.lamichhane123@gmail.com"
match = re.search(r"^[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}$",email)
print(match)
print(match.group())