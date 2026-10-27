# The API and the Celery worker share this image: same code, different command (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf-cache

WORKDIR /app

# CPU-only torch. The default Linux x86 wheel bundles ~2.5GB of CUDA libraries.
# Docker on a Mac can't use the GPU (MPS) either, so training in a container runs on CPU.
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /hf-cache /app/model_registry /app/results \
 && chown -R app:app /hf-cache /app/model_registry /app/results

COPY --chown=app:app alembic.ini data.json ./
COPY --chown=app:app migrations migrations
COPY --chown=app:app toxic_classifier toxic_classifier
COPY --chown=app:app db db
COPY --chown=app:app api api
COPY --chown=app:app worker worker
COPY --chown=app:app scripts scripts

USER app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
