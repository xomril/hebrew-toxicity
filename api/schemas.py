import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TEXT_CHARS = 2000
MAX_BATCH_SIZE = 64

Label = Literal[0, 1]


# ---------- classify ----------
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    model_version: Optional[str] = None
    source: Optional[str] = Field(None, max_length=100)


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    model_version: Optional[str] = None
    source: Optional[str] = Field(None, max_length=100)


class Classification(BaseModel):
    comment_id: int
    text: str
    is_toxic: bool
    label: int
    toxic_prob: float
    model_version: str


class BatchClassification(BaseModel):
    results: List[Classification]


# ---------- feedback ----------
class FeedbackRequest(BaseModel):
    """משוב על תגובה שסווגה (comment_id) או על טקסט חופשי."""

    correct_label: Label
    comment_id: Optional[int] = None
    text: Optional[str] = Field(None, min_length=1, max_length=MAX_TEXT_CHARS)

    @model_validator(mode="after")
    def _needs_target(self):
        if self.comment_id is None and self.text is None:
            raise ValueError("provide comment_id or text")
        return self


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    raw_comment_id: Optional[int]
    text: str
    predicted_label: Optional[int]
    correct_label: int
    model_version: Optional[str]
    created_at: datetime.datetime


# ---------- dataset ----------
class ExampleCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    label: Label


class ExampleUpdate(BaseModel):
    label: Label


class ExampleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    label: int
    source: str
    raw_comment_id: Optional[int]
    created_at: datetime.datetime


class ExamplePage(BaseModel):
    total: int
    items: List[ExampleOut]


class DatasetStats(BaseModel):
    total: int
    safe: int
    toxic: int


# ---------- comments ----------
class CommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    source: Optional[str]
    predicted_label: Optional[int]
    toxic_prob: Optional[float]
    model_version: Optional[str]
    created_at: datetime.datetime


class CommentPage(BaseModel):
    total: int
    items: List[CommentOut]


# ---------- training ----------
class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: str
    status: str
    params: dict
    progress: float
    current_epoch: Optional[float]
    history: List[dict]
    accuracy: Optional[float]
    f1: Optional[float]
    dataset_size: Optional[int]
    error: Optional[str]
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    finished_at: Optional[datetime.datetime]


# ---------- models ----------
class PublicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: str
    repo_id: str
    private: bool
    status: str
    url: Optional[str]
    error: Optional[str]
    created_at: datetime.datetime
    finished_at: Optional[datetime.datetime]


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    version: str
    is_default: bool
    run: Optional[TrainingRunOut]
    publications: List[PublicationOut]


class HubStatus(BaseModel):
    logged_in: bool
    user: Optional[str]
    default_repo_id: Optional[str]
