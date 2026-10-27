from tests.conftest import FAKE_VERSION


def classify(client, text, **extra):
    resp = client.post("/classify", json={"text": text, **extra})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------- classify ----------
def test_classify_returns_prediction_and_stores_comment(client):
    body = classify(client, "אתה אידיוט", source="web")
    assert body["is_toxic"] is True
    assert body["label"] == 1
    assert body["model_version"] == FAKE_VERSION

    comment = client.get(f"/comments/{body['comment_id']}").json()
    assert comment["text"] == "אתה אידיוט"
    assert comment["source"] == "web"
    assert comment["predicted_label"] == 1


def test_classify_batch(client):
    resp = client.post("/classify/batch", json={"texts": ["יום יפה", "מטומטם"]})
    assert resp.status_code == 200
    labels = [r["label"] for r in resp.json()["results"]]
    assert labels == [0, 1]
    assert client.get("/comments").json()["total"] == 2


def test_classify_rejects_bad_input(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify/batch", json={"texts": ["  "]}).status_code == 422
    assert client.post("/classify", json={"text": "שלום", "model_version": "nope"}).status_code == 404


# ---------- feedback ----------
def test_feedback_on_comment_adds_corrected_example(client):
    comment_id = classify(client, "יום יפה")["comment_id"]

    resp = client.post("/feedback", json={"comment_id": comment_id, "correct_label": 1})
    assert resp.status_code == 201
    fb = resp.json()
    assert fb["predicted_label"] == 0
    assert fb["correct_label"] == 1
    assert fb["model_version"] == FAKE_VERSION

    items = client.get("/dataset", params={"source": "feedback"}).json()["items"]
    assert [(i["text"], i["label"], i["raw_comment_id"]) for i in items] == [("יום יפה", 1, comment_id)]
    assert len(client.get("/feedback").json()) == 1


def test_feedback_validation(client):
    assert client.post("/feedback", json={"correct_label": 1}).status_code == 422
    assert client.post("/feedback", json={"comment_id": 1, "correct_label": 2}).status_code == 422
    assert client.post("/feedback", json={"comment_id": 999, "correct_label": 1}).status_code == 404


def test_unlabeled_comments_filter(client):
    first = classify(client, "יום יפה")["comment_id"]
    second = classify(client, "מטומטם")["comment_id"]
    client.post("/feedback", json={"comment_id": first, "correct_label": 0})

    ids = [c["id"] for c in client.get("/comments", params={"unlabeled": True}).json()["items"]]
    assert ids == [second]


# ---------- dataset ----------
def test_dataset_crud_and_stats(client):
    a = client.post("/dataset", json={"text": "תודה רבה", "label": 0}).json()
    client.post("/dataset", json={"text": "אידיוט", "label": 1})
    assert client.get("/dataset/stats").json() == {"total": 2, "safe": 1, "toxic": 1}

    assert client.patch(f"/dataset/{a['id']}", json={"label": 1}).json()["label"] == 1
    assert client.get("/dataset", params={"label": 1}).json()["total"] == 2
    assert client.get("/dataset", params={"search": "תודה"}).json()["total"] == 1

    assert client.delete(f"/dataset/{a['id']}").status_code == 204
    assert client.delete(f"/dataset/{a['id']}").status_code == 404
    assert client.get("/dataset/stats").json()["total"] == 1


def test_dataset_rejects_bad_label(client):
    assert client.post("/dataset", json={"text": "x", "label": 3}).status_code == 422


def test_dataset_pagination(client):
    for i in range(5):
        client.post("/dataset", json={"text": f"דוגמה {i}", "label": 0})
    page = client.get("/dataset", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 5
    assert [i["text"] for i in page["items"]] == ["דוגמה 2", "דוגמה 1"]


# ---------- meta ----------
def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": FAKE_VERSION}
