# Hebrew Toxicity

Classifies Hebrew comments as toxic or safe, using a fine-tuned [DictaBERT](https://huggingface.co/dicta-il/dictabert). The project also includes an admin for the dataset, training runs and model publishing.

```
admin/ (Next.js)  ──HTTP──▶  api/ (FastAPI)  ──▶  PostgreSQL (db/)
                                  │
                                  └── Redis ──▶ worker/ (Celery): training + Hugging Face uploads
toxic_classifier/: the ML core (training, inference, registry, publishing), with no UI or database code
```

## Setup (once)

```bash
docker compose up -d                              # Postgres + Redis
venv/bin/pip install -r requirements.txt
venv/bin/alembic upgrade head                     # create tables
venv/bin/python -m scripts.import_json data.json  # import the seed dataset
npm --prefix admin install
venv/bin/hf auth login                            # Hugging Face token with Write access (for publishing)
```

## Run (3 terminals)

```bash
venv/bin/uvicorn api.main:app --port 8000                                  # API, docs at /docs
venv/bin/celery -A worker.celery_app worker --pool=solo --loglevel=info    # training/upload worker
npm --prefix admin run dev                                                 # admin at http://localhost:3000
```

Restart the worker after changing Python code. Celery doesn't reload automatically.

## Tests

```bash
venv/bin/python -m pytest
```

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://toxic:toxic@localhost:5432/toxic` |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `TOXIC_MODEL_VERSION` | the most recently trained model in `model_registry/` |
| `HF_REPO_ID` | `<your HF user>/hebrew-toxicity` |
| `ADMIN_ORIGINS` | `http://localhost:3000` (CORS) |
| `NEXT_PUBLIC_API_URL` (admin) | `http://localhost:8000` |
