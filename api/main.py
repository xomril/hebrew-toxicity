"""שירות API לסיווג תגובות רעילות. הרצה: uvicorn api.main:app"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import inference
from api.routers import classify, comments, dataset, feedback, models, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    # טעינת המודל מראש כדי שהבקשה הראשונה לא תהיה איטית
    version = inference.default_version()
    if version:
        inference.run_prediction(version, "warmup")
    yield


app = FastAPI(title="Hebrew Toxicity API", version="0.3.0", lifespan=lifespan)

# ממשק הניהול (Next.js) רץ על origin אחר
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ADMIN_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router)
app.include_router(feedback.router)
app.include_router(dataset.router)
app.include_router(comments.router)
app.include_router(training.router)
app.include_router(models.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_version": inference.default_version()}
