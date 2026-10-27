# Hebrew Toxicity

Classifies Hebrew comments as toxic or safe, using a fine-tuned [DictaBERT](https://huggingface.co/dicta-il/dictabert). The project also includes an admin for the dataset, training runs and model publishing.

```
admin/ (Next.js)  ──HTTP──▶  api/ (FastAPI)  ──▶  PostgreSQL (db/)
                                  │
                                  └── Redis ──▶ worker/ (Celery): training + Hugging Face uploads
toxic_classifier/: the ML core (training, inference, registry, publishing), with no UI or database code
```

## Run everything with one command

Needs Docker with about 6GB of memory (the API and the worker each load a 740MB model).

```bash
docker compose up --build
```

| | |
|---|---|
| Admin | http://localhost:3000 |
| API docs | http://localhost:8000/docs |

The first start takes several minutes (it builds the images), and the API needs another minute to load the model. Wait until `docker compose ps` shows the API as `healthy`.

First run only, to load the seed dataset into the database:

```bash
docker compose run --rm api python -m scripts.import_json data.json
```

How it fits together:

- `postgres`, `redis`: data and job queue. Postgres data lives in the `pgdata` volume and survives `docker compose down` (but not `down -v`).
- `migrate`: runs the Alembic migrations on every start and exits.
- `api`, `worker`: the same image (`Dockerfile`), different command.
- `admin`: the Next.js app, built from `admin/Dockerfile`.
- `./model_registry/` is mounted into the API and the worker, so trained models are kept on your machine and shared by both.
- Secrets come from `.env` at runtime and are never copied into an image. `HUGGING_FACE_TOKEN` is passed to the API and the worker only, for the private model repo and for publishing.

**Training in a container runs on the CPU** (Docker on a Mac can't use the Apple GPU). A 1-epoch run on the 301-example dataset took about 2 minutes this way, including the one-time download of the base model. Larger datasets will feel it more. To train on the Apple GPU, run everything else in Docker and the worker natively: `docker compose up --build --scale worker=0`, then start the worker as described below.

If you change `PUBLIC_API_URL`, rebuild the admin (`docker compose up --build admin`): the URL is built into the browser bundle.

## Run without Docker for the app (development)

```bash
docker compose up -d postgres redis              # only the databases
venv/bin/pip install -r requirements.txt
venv/bin/alembic upgrade head
venv/bin/python -m scripts.import_json data.json
npm --prefix admin install
venv/bin/hf auth login                           # Hugging Face token with Write access (for publishing)
```

Then, in 3 terminals:

```bash
venv/bin/uvicorn api.main:app --port 8000                                  # API, docs at /docs
venv/bin/celery -A worker.celery_app worker --pool=solo --loglevel=info    # training/upload worker
npm --prefix admin run dev                                                 # admin at http://localhost:3000
```

Restart the worker after changing Python code. Celery doesn't reload automatically. Stop these before `docker compose up`, since they use the same ports.

## Tests

```bash
venv/bin/python -m pytest
```

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://toxic:toxic@localhost:5432/toxic` (set automatically in Docker) |
| `REDIS_URL` | `redis://localhost:6379/0` (set automatically in Docker) |
| `HUGGING_FACE_TOKEN` | in `.env`; passed to the containers as `HF_TOKEN` |
| `TOXIC_MODEL_VERSION` | the most recently trained model in `model_registry/` |
| `HF_REPO_ID` | `<your HF user>/hebrew-toxicity` |
| `ADMIN_ORIGINS` | `http://localhost:3000` (CORS) |
| `NEXT_PUBLIC_API_URL` (admin, dev) | `http://localhost:8000` |

Docker only: `POSTGRES_PASSWORD`, `POSTGRES_PORT`, `REDIS_PORT`, `API_PORT`, `ADMIN_PORT`, `PUBLIC_API_URL` (the URL your browser uses for the API), `MODEL_REGISTRY_DIR`.
