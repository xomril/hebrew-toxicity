"""שליחת משימות ל-worker. ה-API לא מייבא את קוד המשימות - רק שולח לפי שם."""

from fastapi import HTTPException

from worker.celery_app import app as celery_app


def enqueue(task_name: str, *args) -> str:
    try:
        return celery_app.send_task(task_name, args=list(args)).id
    except Exception as e:  # Redis לא זמין
        raise HTTPException(status_code=503, detail=f"job queue unavailable: {e}")
