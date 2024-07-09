import sys
sys.path.append(".")

from transformers import (
    DataCollatorWithPadding,
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer
)

from src.lora import LoRAConfig
from src.utils import Utils


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

utility_func = Utils(tokenizer=tokenizer)
lora_config = LoRAConfig(model=model)

# Load dataset
tokenized_hf_ds = utility_func.get_tokenized_data()

# Load trainer
training_args = TrainingArguments(
    output_dir="./test/test_output/lora_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy="no"
)

trainer = Trainer(
    model=lora_config.model,
    args=training_args,
    train_dataset=tokenized_hf_ds["train"],
    eval_dataset=tokenized_hf_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=utility_func.compute_metrics,
)

# Train model
trainer.train()
