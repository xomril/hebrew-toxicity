import datetime
import logging
import time

from celery.signals import worker_ready
from sqlalchemy import select, update

from db import ModelPublication, SessionLocal, TrainingRun
from db.repository import labeled_dataframe
from toxic_classifier.publish import publish_model
from toxic_classifier.training import TrainConfig, TrainingCanceled, TrainingHooks, train_model

from .celery_app import app

log = logging.getLogger(__name__)

# לא לכתוב למסד בכל צעד אימון
PROGRESS_WRITE_INTERVAL = 1.0
CANCEL_CHECK_INTERVAL = 2.0


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _update_run(run_id: int, **values):
    with SessionLocal() as session:
        session.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(**values))
        session.commit()


@worker_ready.connect
def _fail_orphaned_jobs(**kwargs):
    """ריצות שנשארו 'רצות' מ-worker קודם שנפל לא ימשיכו לעולם - מסמנים אותן כנכשלות."""
    with SessionLocal() as session:
        session.execute(
            update(TrainingRun)
            .where(TrainingRun.status.in_(["running", "canceling"]))
            .values(status="failed", error="worker restarted during training", finished_at=_now())
        )
        session.execute(
            update(ModelPublication)
            .where(ModelPublication.status == "uploading")
            .values(status="failed", error="worker restarted during upload", finished_at=_now())
        )
        session.commit()


class _RunTracker:
    """מתרגם את ה-hooks של האימון לעדכונים בטבלת training_runs."""

    def __init__(self, run_id: int):
        self.run_id = run_id
        self.history = []
        self._last_write = 0.0
        self._last_cancel_check = 0.0
        self._canceled = False

    def on_progress(self, progress: float, epoch: float):
        now = time.monotonic()
        if now - self._last_write >= PROGRESS_WRITE_INTERVAL or progress >= 1.0:
            self._last_write = now
            _update_run(self.run_id, progress=round(progress, 4), current_epoch=round(epoch, 2))

    def on_epoch_end(self, metrics: dict):
        self.history.append(metrics)
        _update_run(self.run_id, history=list(self.history))

    def should_stop(self) -> bool:
        now = time.monotonic()
        if not self._canceled and now - self._last_cancel_check >= CANCEL_CHECK_INTERVAL:
            self._last_cancel_check = now
            with SessionLocal() as session:
                status = session.scalar(select(TrainingRun.status).where(TrainingRun.id == self.run_id))
            self._canceled = status == "canceling"
        return self._canceled

    def hooks(self) -> TrainingHooks:
        return TrainingHooks(on_progress=self.on_progress, on_epoch_end=self.on_epoch_end, should_stop=self.should_stop)


@app.task(name="train_run")
def train_run(run_id: int):
    with SessionLocal() as session:
        run = session.get(TrainingRun, run_id)
        # בוטל לפני שהתחיל, או כבר טופל
        if run is None or run.status != "queued":
            return
        run.status = "running"
        run.started_at = _now()
        df = labeled_dataframe(session)
        run.dataset_size = len(df)
        params = dict(run.params)
        version = run.version
        session.commit()

    cfg = TrainConfig(
        version=version,
        epochs=params.get("epochs", TrainConfig.epochs),
        learning_rate=params.get("learning_rate", TrainConfig.learning_rate),
        batch_size=params.get("batch_size", TrainConfig.batch_size),
    )
    tracker = _RunTracker(run_id)
    try:
        result = train_model(cfg, df, tracker.hooks())
    except TrainingCanceled:
        _update_run(run_id, status="canceled", finished_at=_now())
        return
    except Exception as e:
        log.exception("training run %s failed", run_id)
        _update_run(run_id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        raise

    _update_run(
        run_id,
        status="succeeded",
        progress=1.0,
        accuracy=result.accuracy,
        f1=result.f1,
        finished_at=_now(),
    )


@app.task(name="publish_model")
def publish_model_task(publication_id: int):
    with SessionLocal() as session:
        pub = session.get(ModelPublication, publication_id)
        if pub is None or pub.status != "queued":
            return
        pub.status = "uploading"
        run = session.scalar(select(TrainingRun).where(TrainingRun.version == pub.version))
        metrics = {"accuracy": run.accuracy, "f1": run.f1, "dataset_size": run.dataset_size} if run else None
        version, repo_id, private = pub.version, pub.repo_id, pub.private
        session.commit()

    try:
        result = publish_model(version, repo_id, private=private, metrics=metrics)
    except Exception as e:
        log.exception("publishing %s failed", version)
        with SessionLocal() as session:
            pub = session.get(ModelPublication, publication_id)
            pub.status, pub.error, pub.finished_at = "failed", f"{type(e).__name__}: {e}", _now()
            session.commit()
        raise

    with SessionLocal() as session:
        pub = session.get(ModelPublication, publication_id)
        pub.status, pub.url, pub.finished_at = "published", result.url, _now()
        session.commit()
