import os
from typing import Optional

from fastapi import HTTPException

from toxic_classifier import Prediction, latest_model, list_models, predict


def default_version() -> Optional[str]:
    # ניתן לקבע גרסה דרך משתנה סביבה, אחרת - הגרסה החדשה ביותר
    return os.environ.get("TOXIC_MODEL_VERSION") or latest_model()


def resolve_version(requested: Optional[str]) -> str:
    version = requested or default_version()
    if not version:
        raise HTTPException(status_code=503, detail="no trained model available")
    if version not in list_models():
        raise HTTPException(status_code=404, detail=f"model version not found: {version}")
    return version


def run_prediction(version: str, text: str) -> Prediction:
    return predict(version, text)
