import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import inference, jobs
from api.schemas import HubStatus, ModelOut, PublicationOut
from db import ModelPublication, TrainingRun, get_session
from toxic_classifier import list_models
from toxic_classifier.publish import default_repo_id

router = APIRouter(prefix="/models", tags=["models"])


def _hub_status() -> HubStatus:
    configured = os.environ.get("HF_REPO_ID")
    try:
        repo_id = default_repo_id()
    except Exception:
        return HubStatus(logged_in=False, user=None, default_repo_id=configured)
    return HubStatus(logged_in=True, user=repo_id.split("/")[0], default_repo_id=configured or repo_id)


@router.get("", response_model=List[ModelOut])
def models(session: Session = Depends(get_session)):
    versions = list_models()
    default = inference.default_version()
    runs = {r.version: r for r in session.scalars(select(TrainingRun).where(TrainingRun.version.in_(versions)))}
    pubs = {}
    for p in session.scalars(
        select(ModelPublication).where(ModelPublication.version.in_(versions)).order_by(ModelPublication.id.desc())
    ):
        pubs.setdefault(p.version, []).append(p)

    return [
        ModelOut(
            version=v,
            is_default=v == default,
            run=runs.get(v),
            publications=pubs.get(v, []),
        )
        for v in versions
    ]


@router.get("/hub", response_model=HubStatus)
def hub_status():
    return _hub_status()


class PublishRequest(BaseModel):
    repo_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    private: bool = True


@router.post("/{version}/publish", response_model=PublicationOut, status_code=201)
def publish(version: str, req: PublishRequest, session: Session = Depends(get_session)):
    if version not in list_models():
        raise HTTPException(status_code=404, detail=f"model version not found: {version}")

    repo_id = req.repo_id
    if not repo_id:
        hub = _hub_status()
        if not hub.default_repo_id:
            raise HTTPException(status_code=400, detail="not logged in to Hugging Face - run `hf auth login`")
        repo_id = hub.default_repo_id

    pub = ModelPublication(version=version, repo_id=repo_id, private=req.private, status="queued")
    session.add(pub)
    session.commit()
    try:
        pub.task_id = jobs.enqueue("publish_model", pub.id)
    except HTTPException:
        pub.status, pub.error = "failed", "job queue unavailable"
        session.commit()
        raise
    session.commit()
    return pub


@router.get("/publications", response_model=List[PublicationOut])
def publications(version: Optional[str] = None, session: Session = Depends(get_session)):
    query = select(ModelPublication).order_by(ModelPublication.id.desc()).limit(50)
    if version:
        query = query.where(ModelPublication.version == version)
    return session.scalars(query).all()
