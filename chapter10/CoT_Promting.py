from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a pretrained tokenizer and model
model_name = "gpt2"  # you can change this to any causal LM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Your prompt
prompt = """
Q: A train travels 60 km/hr for 30 minutes. How far does it go?
Let's think step by step.
"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate output
output = model.generate(**inputs, max_new_tokens=100)

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=True))
