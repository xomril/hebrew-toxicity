"""ליבת ה-ML: דאטה, אימון, רישום מודלים וחיזוי. ללא תלות ב-UI."""

from .data import add_example, load_data, save_data
from .inference import Prediction, predict
from .registry import get_model_path, latest_model, list_models
from .training import TrainConfig, TrainResult, train_model

__all__ = [
    "add_example",
    "load_data",
    "save_data",
    "Prediction",
    "predict",
    "get_model_path",
    "latest_model",
    "list_models",
    "TrainConfig",
    "TrainResult",
    "train_model",
]
