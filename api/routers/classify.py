from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api import inference
from api.schemas import MAX_TEXT_CHARS, BatchClassification, BatchClassifyRequest, Classification, ClassifyRequest
from db import RawComment, get_session

router = APIRouter(tags=["classify"])


def _classify_and_store(session: Session, version: str, text: str, source: Optional[str]) -> Classification:
    pred = inference.run_prediction(version, text)
    comment = RawComment(
        text=text,
        source=source,
        predicted_label=pred.label,
        toxic_prob=pred.toxic_prob,
        model_version=version,
    )
    session.add(comment)
    session.flush()
    return Classification(
        comment_id=comment.id,
        text=text,
        is_toxic=pred.is_toxic,
        label=pred.label,
        toxic_prob=round(pred.toxic_prob, 4),
        model_version=version,
    )


@router.post("/classify", response_model=Classification)
def classify(req: ClassifyRequest, session: Session = Depends(get_session)):
    version = inference.resolve_version(req.model_version)
    result = _classify_and_store(session, version, req.text, req.source)
    session.commit()
    return result


@router.post("/classify/batch", response_model=BatchClassification)
def classify_batch(req: BatchClassifyRequest, session: Session = Depends(get_session)):
    if any(not t.strip() or len(t) > MAX_TEXT_CHARS for t in req.texts):
        raise HTTPException(status_code=422, detail=f"each text must be 1-{MAX_TEXT_CHARS} characters")
    version = inference.resolve_version(req.model_version)
    results = [_classify_and_store(session, version, t, req.source) for t in req.texts]
    session.commit()
    return BatchClassification(results=results)
