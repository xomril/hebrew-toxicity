import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawComment(Base):
    """כל תגובה שסווגה דרך ה-API, יחד עם תוצאת המודל."""

    __tablename__ = "raw_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    predicted_label: Mapped[Optional[int]] = mapped_column(Integer)
    toxic_prob: Mapped[Optional[float]] = mapped_column(Float)
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class LabeledExample(Base):
    """הדאטה-סט המתויג שעליו מאמנים."""

    __tablename__ = "labeled_dataset"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    label: Mapped[int] = mapped_column(Integer, index=True)
    # seed / manual / feedback
    source: Mapped[str] = mapped_column(String(50), default="manual")
    raw_comment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("raw_comments.id", ondelete="SET NULL"))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserFeedback(Base):
    """משוב על תחזית: האם המודל צדק, ומה התגית הנכונה."""

    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_comment_id: Mapped[Optional[int]] = mapped_column(ForeignKey("raw_comments.id", ondelete="SET NULL"), index=True)
    text: Mapped[str] = mapped_column(Text)
    predicted_label: Mapped[Optional[int]] = mapped_column(Integer)
    correct_label: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrainingRun(Base):
    """ריצת אימון: נוצרת מה-API, רצה ב-worker, ומייצרת גרסת מודל ברג'יסטרי."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(100), unique=True)
    # queued / running / canceling / succeeded / failed / canceled
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_epoch: Mapped[Optional[float]] = mapped_column(Float)
    # תוצאות eval לכל אפוק: [{"epoch": 1, "accuracy": .., "f1": .., "loss": ..}]
    history: Mapped[list] = mapped_column(JSON, default=list)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    f1: Mapped[Optional[float]] = mapped_column(Float)
    dataset_size: Mapped[Optional[int]] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text)
    task_id: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))


class ModelPublication(Base):
    """העלאה של גרסת מודל ל-Hugging Face Hub."""

    __tablename__ = "model_publications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(100), index=True)
    repo_id: Mapped[str] = mapped_column(String(200))
    private: Mapped[bool] = mapped_column(default=True)
    # queued / uploading / published / failed
    status: Mapped[str] = mapped_column(String(20), default="queued")
    url: Mapped[Optional[str]] = mapped_column(String(500))
    error: Mapped[Optional[str]] = mapped_column(Text)
    task_id: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
