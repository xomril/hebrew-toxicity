"""פעולות על הדאטה, משותפות ל-API ולדשבורד."""

from typing import Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import LabeledExample, RawComment, UserFeedback


def add_labeled_example(
    session: Session, text: str, label: int, source: str = "manual", raw_comment_id: Optional[int] = None
) -> LabeledExample:
    if not text or not text.strip():
        raise ValueError("empty text")
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    example = LabeledExample(text=text, label=label, source=source, raw_comment_id=raw_comment_id)
    session.add(example)
    session.commit()
    return example


def labeled_dataframe(session: Session) -> pd.DataFrame:
    """הדאטה-סט המתויג בפורמט שהאימון מצפה לו (text, label)."""
    rows = session.execute(select(LabeledExample.text, LabeledExample.label).order_by(LabeledExample.id)).all()
    return pd.DataFrame(rows, columns=["text", "label"])


def dataset_stats(session: Session) -> dict:
    counts = dict(session.execute(select(LabeledExample.label, func.count()).group_by(LabeledExample.label)).all())
    return {"total": sum(counts.values()), "safe": counts.get(0, 0), "toxic": counts.get(1, 0)}


def record_feedback(
    session: Session,
    correct_label: int,
    raw_comment_id: Optional[int] = None,
    text: Optional[str] = None,
    predicted_label: Optional[int] = None,
    model_version: Optional[str] = None,
) -> UserFeedback:
    """שומר משוב ומוסיף את הדוגמה המתוקנת לדאטה-סט לאימון הבא."""
    if raw_comment_id is not None:
        comment = session.get(RawComment, raw_comment_id)
        if comment is None:
            raise LookupError(f"comment not found: {raw_comment_id}")
        text = comment.text
        predicted_label = comment.predicted_label
        model_version = comment.model_version
    if not text or not text.strip():
        raise ValueError("feedback needs a comment_id or text")
    if correct_label not in (0, 1):
        raise ValueError("label must be 0 or 1")

    feedback = UserFeedback(
        raw_comment_id=raw_comment_id,
        text=text,
        predicted_label=predicted_label,
        correct_label=correct_label,
        model_version=model_version,
    )
    session.add(feedback)
    session.add(LabeledExample(text=text, label=correct_label, source="feedback", raw_comment_id=raw_comment_id))
    session.commit()
    return feedback
