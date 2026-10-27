from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import DatasetStats, ExampleCreate, ExampleOut, ExamplePage, ExampleUpdate
from db import LabeledExample, get_session
from db.repository import add_labeled_example, dataset_stats

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("", response_model=ExamplePage)
def list_examples(
    label: Optional[int] = Query(None, ge=0, le=1),
    source: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    query = select(LabeledExample)
    if label is not None:
        query = query.where(LabeledExample.label == label)
    if source:
        query = query.where(LabeledExample.source == source)
    if search:
        query = query.where(LabeledExample.text.contains(search))

    total = session.scalar(select(func.count()).select_from(query.subquery()))
    items = session.scalars(query.order_by(LabeledExample.id.desc()).limit(limit).offset(offset)).all()
    return ExamplePage(total=total, items=items)


@router.get("/stats", response_model=DatasetStats)
def stats(session: Session = Depends(get_session)):
    return dataset_stats(session)


@router.post("", response_model=ExampleOut, status_code=201)
def create_example(req: ExampleCreate, session: Session = Depends(get_session)):
    try:
        return add_labeled_example(session, req.text, req.label, source="manual")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.patch("/{example_id}", response_model=ExampleOut)
def update_example(example_id: int, req: ExampleUpdate, session: Session = Depends(get_session)):
    example = session.get(LabeledExample, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="example not found")
    example.label = req.label
    session.commit()
    return example


@router.delete("/{example_id}", status_code=204)
def delete_example(example_id: int, session: Session = Depends(get_session)):
    example = session.get(LabeledExample, example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="example not found")
    session.delete(example)
    session.commit()
    return Response(status_code=204)
