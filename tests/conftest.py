import os
import tempfile

# חייב לקרות לפני ייבוא db, כי ה-engine נוצר בזמן הייבוא
_db_file = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_db_file}"
os.environ.pop("TOXIC_MODEL_VERSION", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import inference  # noqa: E402
from api.main import app  # noqa: E402
from db import Base, engine  # noqa: E402
from toxic_classifier import Prediction  # noqa: E402

FAKE_VERSION = "v_test"
TOXIC_WORDS = ("אידיוט", "מטומטם")


def fake_predict(version, text):
    toxic_prob = 0.95 if any(w in text for w in TOXIC_WORDS) else 0.05
    return Prediction(label=int(toxic_prob > 0.5), toxic_prob=toxic_prob, safe_prob=1 - toxic_prob)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    from api.routers import models as models_router
    from api.routers import training as training_router

    monkeypatch.setattr(inference, "predict", fake_predict)
    for module in (inference, models_router, training_router):
        monkeypatch.setattr(module, "list_models", lambda: [FAKE_VERSION])
    monkeypatch.setattr(inference, "latest_model", lambda: FAKE_VERSION)


@pytest.fixture(autouse=True)
def fake_queue(monkeypatch):
    """משימות שנשלחו ל-worker נאספות כאן במקום להישלח ל-Redis."""
    from api import jobs

    sent = []

    def enqueue(task_name, *args):
        sent.append((task_name, args))
        return f"task-{len(sent)}"

    monkeypatch.setattr(jobs, "enqueue", enqueue)
    return sent


@pytest.fixture
def client():
    # ללא with - כדי לא להריץ את ה-lifespan שטוען מודל אמיתי
    return TestClient(app)
