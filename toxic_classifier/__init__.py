"""ליבת ה-ML: אימון, רישום מודלים וחיזוי. ללא תלות ב-UI או במסד הנתונים."""

from .inference import Prediction, predict
from .registry import get_model_path, latest_model, list_models
from .training import TrainConfig, TrainResult, train_model

__all__ = [
    "Prediction",
    "predict",
    "get_model_path",
    "latest_model",
    "list_models",
    "TrainConfig",
    "TrainResult",
    "train_model",
]
