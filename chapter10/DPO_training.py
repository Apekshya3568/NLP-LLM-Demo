!pip install transformers datasets trl

from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

dataset = load_dataset("json", data_files="dpo_data.json")

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    beta=0.1,
)

trainer.train()
