import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from evaluate import load as load_metric

# 1. זיהוי ה-GPU של Apple Silicon (M1/M2)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. טעינת טוקנייזר ומודל של Dicta
MODEL_NAME = "dicta-il/dictabert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 3. טעינת הדאטה
dataset = load_dataset('json', data_files='data.json')
dataset = dataset['train'].train_test_split(test_size=0.2) # פיצול ל-Train ו-Validation

# 4. עיבוד הטקסט (Tokenization)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5. הגדרות אימון
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    use_mps_device=True, # הגדרה יעודית ל-Mac M1/M2!
    logging_steps=10
)

# 6. יצירת ה-Trainer והרצת האימון
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

print("Starting training...")
trainer.train()

# 7. שמירת המודל המאומן שלך!
model.save_pretrained("./my_hebrew_toxic_model")
tokenizer.save_pretrained("./my_hebrew_toxic_model")
print("Model saved to ./my_hebrew_toxic_model")