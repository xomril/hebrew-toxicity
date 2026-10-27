from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_LENGTH, TOXIC_LABEL
from .registry import get_model_path


@dataclass
class Prediction:
    label: int
    toxic_prob: float
    safe_prob: float

    @property
    def is_toxic(self) -> bool:
        return self.label == TOXIC_LABEL

    @property
    def confidence(self) -> float:
        return self.toxic_prob if self.is_toxic else self.safe_prob


@lru_cache(maxsize=4)
def _load(version: str):
    # טעינה פעם אחת לכל גרסה במקום בכל קריאה
    path = get_model_path(version)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict(version: str, text: str) -> Prediction:
    tokenizer, model = _load(version)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    toxic_prob = probs[TOXIC_LABEL].item()
    safe_prob = 1 - toxic_prob
    label = TOXIC_LABEL if toxic_prob > 0.5 else 1 - TOXIC_LABEL
    return Prediction(label=label, toxic_prob=toxic_prob, safe_prob=safe_prob)
