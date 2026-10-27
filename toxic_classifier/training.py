import datetime
from dataclasses import dataclass, field
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .config import BASE_MODEL, MAX_LENGTH, MIN_TRAIN_EXAMPLES, MODELS_DIR, NUM_LABELS, RESULTS_DIR
from .data import load_data


def new_version_name() -> str:
    return f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"


@dataclass
class TrainConfig:
    version: str = field(default_factory=new_version_name)
    epochs: int = 5
    learning_rate: float = 2e-5
    batch_size: int = 8
    test_size: float = 0.2


@dataclass
class TrainResult:
    version: str
    model_path: Path
    accuracy: float
    f1: float


def _compute_metrics_fn():
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        return {"accuracy": acc, "f1": f1}

    return compute_metrics


def train_model(cfg: TrainConfig) -> TrainResult:
    """מאמן מודל חדש על הדאטה-סט ושומר אותו ברג'יסטרי תחת cfg.version."""
    df = load_data()
    if len(df) < MIN_TRAIN_EXAMPLES:
        raise ValueError(f"need at least {MIN_TRAIN_EXAMPLES} examples, got {len(df)}")

    output_model_path = MODELS_DIR / cfg.version
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ds_split = Dataset.from_pandas(df).train_test_split(test_size=cfg.test_size)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    tokenized_ds = ds_split.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR / cfg.version),
        eval_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=0.01,
        use_mps_device=torch.backends.mps.is_available(),
        logging_steps=5,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=_compute_metrics_fn(),
    )

    trainer.train()

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    eval_results = trainer.evaluate()
    return TrainResult(
        version=cfg.version,
        model_path=output_model_path,
        accuracy=eval_results.get("eval_accuracy", 0.0),
        f1=eval_results.get("eval_f1", 0.0),
    )
