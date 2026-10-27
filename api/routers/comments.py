from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import CommentOut, CommentPage
from db import LabeledExample, RawComment, get_session

router = APIRouter(prefix="/comments", tags=["comments"])


@router.get("", response_model=CommentPage)
def list_comments(
    predicted_label: Optional[int] = Query(None, ge=0, le=1),
    unlabeled: bool = Query(False, description="only comments not yet in the labeled dataset"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    query = select(RawComment)
    if predicted_label is not None:
        query = query.where(RawComment.predicted_label == predicted_label)
    if unlabeled:
        labeled_ids = select(LabeledExample.raw_comment_id).where(LabeledExample.raw_comment_id.is_not(None))
        query = query.where(RawComment.id.not_in(labeled_ids))

    total = session.scalar(select(func.count()).select_from(query.subquery()))
    items = session.scalars(query.order_by(RawComment.id.desc()).limit(limit).offset(offset)).all()
    return CommentPage(total=total, items=items)


@router.get("/{comment_id}", response_model=CommentOut)
def get_comment(comment_id: int, session: Session = Depends(get_session)):
    comment = session.get(RawComment, comment_id)
    if comment is None:
        raise HTTPException(status_code=404, detail="comment not found")
    return comment
