from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=1
)

ds = load_dataset("json", data_files="pair_data.json")

def preprocess(batch):
    return tokenizer(batch["prompt"] + batch["answer"], truncation=True)

ds = ds.map(preprocess)

args = TrainingArguments(
    output_dir="reward_model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
)

trainer = Trainer(
    model=reward_model,
    args=args,
    train_dataset=ds["train"],
)

trainer.train()
