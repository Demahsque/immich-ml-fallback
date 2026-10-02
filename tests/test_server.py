import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import MODEL, word_vector
from immich_ml_fallback.server import create_app


def entries(model=MODEL, **tasks):
    return json.dumps(tasks or {"clip": {"textual": {"modelName": model, "options": {}}}})


def predict(client, text="plage", entries_json=None, **kwargs):
    # files={name: (None, value)} sends plain multipart fields, exactly like Node's FormData.
    files = {"entries": (None, entries_json or entries())}
    if text is not None:
        files["text"] = (None, text)
    files.update(kwargs.get("extra", {}))
    return client.post("/predict", files=files)


@pytest.fixture
def client(data_dir):
    with TestClient(create_app(data_dir)) as c:
        yield c


def test_root_and_ping(client):
    assert client.get("/").json() == {"message": "Immich ML fallback proxy (text search only)"}
    response = client.get("/ping")
    assert response.status_code == 200 and response.text == "pong"


def test_predict_matches_immich_response_format(client):
    response = predict(client, "plage")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert list(body) == ["clip"]
    assert isinstance(body["clip"], str)  # a JSON array serialised as a string, like Immich ML
    vector = np.array(json.loads(body["clip"]), dtype=np.float32)
    assert vector.shape == (64,)
    assert float(np.dot(vector, word_vector("plage"))) == pytest.approx(1.0, abs=1e-3)
    assert body["clip"].startswith("[") and body["clip"].endswith("]") and " " not in body["clip"]


def test_predict_accepts_urlencoded_form_too(client):
    response = client.post("/predict", data={"entries": entries(), "text": "chat"})
    assert response.status_code == 200


def test_model_name_with_prefix_is_accepted(client):
    assert predict(client, entries_json=entries("immich-app/" + MODEL)).status_code == 200


def test_model_mismatch_is_refused(client):
    response = predict(client, entries_json=entries("ViT-B-32__openai"))
    assert response.status_code == 409 and "ViT-B-32__openai" in response.json()["detail"]


def test_other_tasks_are_refused_with_501(client):
    for tasks in (
        {"clip": {"visual": {"modelName": MODEL}}},
        {"facial-recognition": {"detection": {"modelName": "buffalo_l", "options": {"minScore": 0.7}}}},
        {"ocr": {"detection": {"modelName": "PP-OCRv5_mobile"}}},
        {"clip": {"textual": {"modelName": MODEL}}, "ocr": {"detection": {"modelName": "x"}}},
    ):
        response = predict(client, entries_json=json.dumps(tasks))
        assert response.status_code == 501, tasks
        assert "only supports" in response.json()["detail"]


def test_image_payload_is_refused(client):
    response = client.post("/predict", files={"entries": (None, entries()), "image": ("x.jpg", b"\xff\xd8\xff")})
    assert response.status_code == 400


def test_malformed_requests(client):
    assert client.post("/predict", files={"text": (None, "plage")}).status_code == 422
    assert predict(client, entries_json="not json").status_code == 422
    assert predict(client, entries_json="[1]").status_code == 422
    assert predict(client, entries_json=json.dumps({"clip": {"textual": {}}})).status_code == 422
    assert predict(client, text=None).status_code == 400


def test_query_without_known_word(client):
    response = predict(client, "xyzzy le la")
    assert response.status_code == 422 and "vocabulary" in response.json()["detail"]


def test_info_and_explain(client):
    info = client.get("/info").json()
    assert info["loaded"] and info["model"] == MODEL and info["dim"] == 64 and info["phrases"] == 2
    explained = client.get("/explain", params={"q": "un chat sur la plage"}).json()
    assert explained == {"used": ["chat", "plage"], "stop_words": ["un", "sur", "la"], "unknown": []}


def test_starts_without_vectors_then_picks_them_up(tmp_path, store):
    with TestClient(create_app(tmp_path)) as client:
        app_holder = client.app.state.holder
        app_holder.recheck_seconds = 0
        assert client.get("/ping").status_code == 503
        assert predict(client).status_code == 503
        store.save(tmp_path)
        assert client.get("/ping").status_code == 200
        assert predict(client).status_code == 200


def test_bad_replacement_keeps_serving_old_vectors(data_dir, store):
    with TestClient(create_app(data_dir)) as client:
        client.app.state.holder.recheck_seconds = 0
        assert predict(client).status_code == 200
        (data_dir / "index.json").write_text("{}")
        assert predict(client).status_code == 200
        assert client.app.state.holder.error
