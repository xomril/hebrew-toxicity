from pathlib import Path
from typing import List, Optional

from .config import MODELS_DIR


def list_models() -> List[str]:
    """גרסאות המודלים ברג'יסטרי, מהחדשה לישנה."""
    if not MODELS_DIR.exists():
        return []
    return sorted((d.name for d in MODELS_DIR.iterdir() if d.is_dir()), reverse=True)


def latest_model() -> Optional[str]:
    models = list_models()
    return models[0] if models else None


def get_model_path(version: str) -> Path:
    path = MODELS_DIR / version
    if not path.is_dir():
        raise FileNotFoundError(f"model version not found: {version}")
    return path
