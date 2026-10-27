import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import BASE_MODEL, ID2LABEL, MAX_LENGTH, MIN_TRAIN_EXAMPLES, MODELS_DIR, NUM_LABELS, RESULTS_DIR


def new_version_name() -> str:
    return f"v{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


class TrainingCanceled(Exception):
    pass


@dataclass
class TrainingHooks:
    """חיבור של מי שמריץ את האימון (worker, CLI) למעקב התקדמות וביטול. כולם אופציונליים."""

    # (progress בין 0 ל-1, epoch נוכחי)
    on_progress: Optional[Callable[[float, float], None]] = None
    # מטריקות eval בסוף כל אפוק: {"epoch", "accuracy", "f1", "loss"}
    on_epoch_end: Optional[Callable[[dict], None]] = None
    should_stop: Optional[Callable[[], bool]] = None


class _HooksCallback(TrainerCallback):
    def __init__(self, hooks: TrainingHooks):
        self.hooks = hooks
        self.canceled = False
        self.training_done = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.hooks.on_progress and state.max_steps:
            self.hooks.on_progress(state.global_step / state.max_steps, state.epoch or 0.0)
        if self.hooks.should_stop and self.hooks.should_stop():
            self.canceled = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.training_done = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # ה-evaluate הסופי אחרי האימון חוזר על האפוק האחרון - לא לדווח פעמיים
        if self.hooks.on_epoch_end and metrics and not self.training_done:
            self.hooks.on_epoch_end(
                {
                    "epoch": round(state.epoch or 0.0, 2),
                    "accuracy": metrics.get("eval_accuracy"),
                    "f1": metrics.get("eval_f1"),
                    "loss": metrics.get("eval_loss"),
                }
            )


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


def train_model(cfg: TrainConfig, df: pd.DataFrame, hooks: Optional[TrainingHooks] = None) -> TrainResult:
    """מאמן מודל חדש על df (עמודות text, label) ושומר אותו ברג'יסטרי תחת cfg.version.

    זורק TrainingCanceled אם hooks.should_stop החזיר True - ואז שום מודל לא נשמר.
    """
    if len(df) < MIN_TRAIN_EXAMPLES:
        raise ValueError(f"need at least {MIN_TRAIN_EXAMPLES} examples, got {len(df)}")

    output_model_path = MODELS_DIR / cfg.version
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ds_split = Dataset.from_pandas(df).train_test_split(test_size=cfg.test_size)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
    )

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
        # המודל הסופי נשמר ברג'יסטרי; צ'קפוינטים לכל אפוק רק תופסים ~700MB כל אחד
        save_strategy="no",
        report_to="none",
    )

    callback = _HooksCallback(hooks or TrainingHooks())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=_compute_metrics_fn(),
        callbacks=[callback],
    )

    trainer.train()
    if callback.canceled:
        raise TrainingCanceled(cfg.version)

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    eval_results = trainer.evaluate()
    return TrainResult(
        version=cfg.version,
        model_path=output_model_path,
        accuracy=eval_results.get("eval_accuracy", 0.0),
        f1=eval_results.get("eval_f1", 0.0),
    )
