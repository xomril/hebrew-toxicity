from pathlib import Path
from typing import List, Optional

from .config import MODELS_DIR


def list_models() -> List[str]:
    """גרסאות המודלים ברג'יסטרי, מהחדשה לישנה (לפי זמן שמירה, כי שמות גרסה יכולים להיות חופשיים)."""
    if not MODELS_DIR.exists():
        return []
    dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def latest_model() -> Optional[str]:
    models = list_models()
    return models[0] if models else None


def get_model_path(version: str) -> Path:
    path = MODELS_DIR / version
    if not path.is_dir():
        raise FileNotFoundError(f"model version not found: {version}")
    return path
