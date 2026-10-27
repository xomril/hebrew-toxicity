from pathlib import Path

import pytest

from api.routers import models as models_router
from db import ModelPublication, SessionLocal, TrainingRun
from tests.conftest import FAKE_VERSION
from toxic_classifier.publish import PublishResult
from toxic_classifier.training import TrainingCanceled, TrainResult
from worker import tasks


def seed_dataset(client, n=10):
    for i in range(n):
        client.post("/dataset", json={"text": f"דוגמה {i}", "label": i % 2})


# ---------- training API ----------
def test_create_run_queues_job(client, fake_queue):
    seed_dataset(client)
    resp = client.post("/training/runs", json={"epochs": 3, "learning_rate": 5e-5, "batch_size": 16})
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["status"] == "queued"
    assert run["params"] == {"epochs": 3, "learning_rate": 5e-5, "batch_size": 16}
    assert fake_queue == [("train_run", (run["id"],))]
    assert client.get(f"/training/runs/{run['id']}").json()["version"] == run["version"]


def test_create_run_validation(client):
    assert client.post("/training/runs", json={}).status_code == 422  # dataset too small
    seed_dataset(client)
    assert client.post("/training/runs", json={"batch_size": 7}).status_code == 422
    assert client.post("/training/runs", json={"version": "bad name!"}).status_code == 422
    assert client.post("/training/runs", json={"version": FAKE_VERSION}).status_code == 409
    assert client.post("/training/runs", json={"version": "v1"}).status_code == 201
    assert client.post("/training/runs", json={"version": "v1"}).status_code == 409


def test_cancel_run(client):
    seed_dataset(client)
    queued = client.post("/training/runs", json={}).json()
    assert client.post(f"/training/runs/{queued['id']}/cancel").json()["status"] == "canceled"
    assert client.post(f"/training/runs/{queued['id']}/cancel").status_code == 409

    running = client.post("/training/runs", json={"version": "v2"}).json()
    with SessionLocal() as s:
        s.get(TrainingRun, running["id"]).status = "running"
        s.commit()
    assert client.post(f"/training/runs/{running['id']}/cancel").json()["status"] == "canceling"


# ---------- worker: train_run ----------
def make_run(**kw):
    with SessionLocal() as s:
        run = TrainingRun(version=kw.pop("version", "v_new"), status="queued", params={"epochs": 2}, progress=0.0, history=[], **kw)
        s.add(run)
        s.commit()
        return run.id


def get_run(run_id):
    with SessionLocal() as s:
        return s.get(TrainingRun, run_id)


def test_train_run_success_records_progress_and_metrics(client, monkeypatch):
    seed_dataset(client)
    seen = {}

    def fake_train(cfg, df, hooks):
        seen["epochs"], seen["rows"] = cfg.epochs, len(df)
        hooks.on_progress(0.5, 1.0)
        hooks.on_epoch_end({"epoch": 1.0, "accuracy": 0.8, "f1": 0.79, "loss": 0.4})
        return TrainResult(version=cfg.version, model_path=Path("/tmp/x"), accuracy=0.9, f1=0.88)

    monkeypatch.setattr(tasks, "train_model", fake_train)
    run_id = make_run()
    tasks.train_run(run_id)

    run = get_run(run_id)
    assert seen == {"epochs": 2, "rows": 10}
    assert run.status == "succeeded"
    assert (run.progress, run.accuracy, run.f1, run.dataset_size) == (1.0, 0.9, 0.88, 10)
    assert run.history == [{"epoch": 1.0, "accuracy": 0.8, "f1": 0.79, "loss": 0.4}]
    assert run.started_at and run.finished_at


def test_train_run_cancel_and_failure(client, monkeypatch):
    seed_dataset(client)

    def canceled_train(cfg, df, hooks):
        raise TrainingCanceled(cfg.version)

    monkeypatch.setattr(tasks, "train_model", canceled_train)
    run_id = make_run(version="v_c")
    tasks.train_run(run_id)
    assert get_run(run_id).status == "canceled"

    def broken_train(cfg, df, hooks):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(tasks, "train_model", broken_train)
    run_id = make_run(version="v_f")
    with pytest.raises(RuntimeError):
        tasks.train_run(run_id)
    run = get_run(run_id)
    assert run.status == "failed" and "out of memory" in run.error


def test_train_run_skips_canceled_before_start(monkeypatch):
    monkeypatch.setattr(tasks, "train_model", lambda *a: pytest.fail("should not train"))
    run_id = make_run()
    with SessionLocal() as s:
        s.get(TrainingRun, run_id).status = "canceled"
        s.commit()
    tasks.train_run(run_id)
    assert get_run(run_id).status == "canceled"


def test_tracker_detects_cancel_request():
    run_id = make_run()
    tracker = tasks._RunTracker(run_id)
    assert tracker.should_stop() is False
    with SessionLocal() as s:
        s.get(TrainingRun, run_id).status = "canceling"
        s.commit()
    tracker._last_cancel_check = 0.0
    assert tracker.should_stop() is True


# ---------- models + publish ----------
def test_models_list_includes_run_and_publications(client):
    run_id = make_run(version=FAKE_VERSION)
    with SessionLocal() as s:
        s.get(TrainingRun, run_id).accuracy = 0.91
        s.add(ModelPublication(version=FAKE_VERSION, repo_id="me/repo", status="published"))
        s.commit()

    [model] = client.get("/models").json()
    assert model["version"] == FAKE_VERSION
    assert model["is_default"] is True
    assert model["run"]["accuracy"] == 0.91
    assert model["publications"][0]["repo_id"] == "me/repo"


def test_publish_queues_job(client, fake_queue, monkeypatch):
    monkeypatch.setattr(models_router, "default_repo_id", lambda: "omri/hebrew-toxicity")
    assert client.get("/models/hub").json() == {
        "logged_in": True,
        "user": "omri",
        "default_repo_id": "omri/hebrew-toxicity",
    }

    resp = client.post(f"/models/{FAKE_VERSION}/publish", json={})
    assert resp.status_code == 201, resp.text
    pub = resp.json()
    assert (pub["repo_id"], pub["private"], pub["status"]) == ("omri/hebrew-toxicity", True, "queued")
    assert fake_queue == [("publish_model", (pub["id"],))]

    assert client.post("/models/nope/publish", json={}).status_code == 404
    assert client.post(f"/models/{FAKE_VERSION}/publish", json={"repo_id": "no-slash"}).status_code == 422


def test_publish_requires_hf_login(client, monkeypatch):
    def not_logged_in():
        raise RuntimeError("no token")

    monkeypatch.setattr(models_router, "default_repo_id", not_logged_in)
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    assert client.get("/models/hub").json()["logged_in"] is False
    assert client.post(f"/models/{FAKE_VERSION}/publish", json={}).status_code == 400


def test_publish_task(monkeypatch):
    run_id = make_run(version=FAKE_VERSION, dataset_size=300)
    with SessionLocal() as s:
        s.get(TrainingRun, run_id).accuracy = 0.9
        pub = ModelPublication(version=FAKE_VERSION, repo_id="me/repo", status="queued")
        s.add(pub)
        s.commit()
        pub_id = pub.id

    calls = []

    def fake_publish(version, repo_id, private, metrics):
        calls.append((version, repo_id, private, metrics["accuracy"], metrics["dataset_size"]))
        return PublishResult(repo_id=repo_id, revision=version, url="https://huggingface.co/me/repo/tree/v_test")

    monkeypatch.setattr(tasks, "publish_model", fake_publish)
    tasks.publish_model_task(pub_id)

    assert calls == [(FAKE_VERSION, "me/repo", True, 0.9, 300)]
    with SessionLocal() as s:
        pub = s.get(ModelPublication, pub_id)
        assert (pub.status, pub.url) == ("published", "https://huggingface.co/me/repo/tree/v_test")
