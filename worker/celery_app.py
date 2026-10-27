"""Celery worker לאימון והעלאת מודלים.

הרצה (pool=solo - אימון אחד בכל פעם, ונדרש ל-MPS על Mac):
    celery -A worker.celery_app worker --pool=solo --loglevel=info
"""

import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery("toxic_worker", broker=REDIS_URL, include=["worker.tasks"])
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)
