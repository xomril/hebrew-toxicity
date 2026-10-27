from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import FeedbackOut, FeedbackRequest
from db import UserFeedback, get_session
from db.repository import record_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, status_code=201)
def create_feedback(req: FeedbackRequest, session: Session = Depends(get_session)):
    try:
        return record_feedback(session, req.correct_label, raw_comment_id=req.comment_id, text=req.text)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("", response_model=List[FeedbackOut])
def list_feedback(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    return session.scalars(select(UserFeedback).order_by(UserFeedback.id.desc()).limit(limit).offset(offset)).all()
