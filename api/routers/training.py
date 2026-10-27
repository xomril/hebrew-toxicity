from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import jobs
from api.schemas import TrainingRunOut
from db import TrainingRun, get_session
from db.repository import dataset_stats
from toxic_classifier import list_models
from toxic_classifier.config import MIN_TRAIN_EXAMPLES
from toxic_classifier.training import new_version_name

router = APIRouter(prefix="/training/runs", tags=["training"])


class TrainingRunCreate(BaseModel):
    version: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9._-]{1,100}$")
    epochs: int = Field(5, ge=1, le=20)
    learning_rate: float = Field(2e-5, gt=0, le=1e-3)
    batch_size: Literal[4, 8, 16, 32] = 8


@router.post("", response_model=TrainingRunOut, status_code=201)
def create_run(req: TrainingRunCreate, session: Session = Depends(get_session)):
    total = dataset_stats(session)["total"]
    if total < MIN_TRAIN_EXAMPLES:
        raise HTTPException(status_code=422, detail=f"need at least {MIN_TRAIN_EXAMPLES} labeled examples, have {total}")

    version = req.version or new_version_name()
    if version in list_models() or session.scalar(select(TrainingRun.id).where(TrainingRun.version == version)):
        raise HTTPException(status_code=409, detail=f"version already exists: {version}")

    run = TrainingRun(
        version=version,
        status="queued",
        params={"epochs": req.epochs, "learning_rate": req.learning_rate, "batch_size": req.batch_size},
        progress=0.0,
        history=[],
    )
    session.add(run)
    session.commit()

    try:
        run.task_id = jobs.enqueue("train_run", run.id)
    except HTTPException:
        run.status, run.error = "failed", "job queue unavailable"
        session.commit()
        raise
    session.commit()
    return run


@router.get("", response_model=List[TrainingRunOut])
def list_runs(limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_session)):
    return session.scalars(select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)).all()


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@router.post("/{run_id}/cancel", response_model=TrainingRunOut)
def cancel_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status == "queued":
        # ה-worker מדלג על ריצות שאינן queued
        run.status = "canceled"
    elif run.status == "running":
        # ה-worker בודק את הסטטוס תוך כדי אימון ועוצר
        run.status = "canceling"
    else:
        raise HTTPException(status_code=409, detail=f"cannot cancel a run that is {run.status}")
    session.commit()
    return run
