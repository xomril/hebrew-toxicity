import os
from pathlib import Path

# שורש הפרויקט (תיקייה אחת מעל החבילה)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# נתיבים - ניתנים לדריסה דרך משתני סביבה
DATA_FILE = Path(os.environ.get("TOXIC_DATA_FILE", PROJECT_ROOT / "data.json"))
MODELS_DIR = Path(os.environ.get("TOXIC_MODELS_DIR", PROJECT_ROOT / "model_registry"))
RESULTS_DIR = Path(os.environ.get("TOXIC_RESULTS_DIR", PROJECT_ROOT / "results"))

# הגדרות מודל
BASE_MODEL = "dicta-il/dictabert"
MAX_LENGTH = 64
NUM_LABELS = 2
TOXIC_LABEL = 1
MIN_TRAIN_EXAMPLES = 10
