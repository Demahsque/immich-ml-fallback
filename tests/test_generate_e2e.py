"""End to end: fake Immich ML server -> generate -> store -> proxy -> compare with the 'real' answer."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

import fake_ml
from conftest import MODEL, bag_of_words_embedding
from immich_ml_fallback.__main__ import main
from immich_ml_fallback.encoder import ApproxEncoder
from immich_ml_fallback.generate import ImmichMLClient, MLClientError, build_vocabulary, generate_store
from immich_ml_fallback.server import create_app
from immich_ml_fallback.stopwords import load_stopwords
from immich_ml_fallback.store import VectorStore

WORDS = ["plage", "chat", "chien", "forêt", "coucher de soleil", "mer", "bateau", "montagne", "neige"]


@pytest.fixture(scope="module")
def ml_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(fake_ml.app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_generate_then_serve_matches_the_real_embedding(ml_url, tmp_path):
    client = ImmichMLClient(ml_url, MODEL)
    client.ping()
    store = generate_store(client, WORDS, tmp_path, workers=3, checkpoint=4, log=lambda m: None)
    assert len(store) == len(WORDS) and store.dim == 64

    loaded = VectorStore.load(tmp_path)
    assert loaded.model == MODEL and loaded.vectors.dtype == np.float16
    encoder = ApproxEncoder(loaded, load_stopwords())
    for query in ("plage", "un chat sur la plage", "chien dans la neige"):
        approx = encoder.encode(query).vector
        real = bag_of_words_embedding(query)
        # The fake model IS a bag of content words, so the proxy must reproduce it (up to float16 rounding).
        assert float(np.dot(approx, real)) > 0.995, query
    # A phrase vector is normalised before being summed with the other words: close, not identical.
    phrase_query = "coucher de soleil sur la mer"
    assert float(np.dot(encoder.encode(phrase_query).vector, bag_of_words_embedding(phrase_query))) > 0.9

    # Same thing through the HTTP proxy, as Immich would call it.
    with TestClient(create_app(tmp_path)) as proxy:
        response = proxy.post(
            "/predict",
            files={"entries": (None, '{"clip":{"textual":{"modelName":"%s","options":{}}}}' % MODEL), "text": (None, "chat plage")},
        )
        assert response.status_code == 200
        served = np.array(__import__("json").loads(response.json()["clip"]), dtype=np.float32)
        assert float(np.dot(served, bag_of_words_embedding("chat plage"))) > 0.995
    client.close()


def test_generate_is_incremental_and_refuses_another_model(ml_url, tmp_path):
    client = ImmichMLClient(ml_url, MODEL)
    generate_store(client, WORDS[:4], tmp_path, log=lambda m: None)
    fake_ml.calls.clear()
    store = generate_store(client, WORDS, tmp_path, log=lambda m: None)
    assert sorted(fake_ml.calls) == sorted(WORDS[4:])  # only the missing words were encoded
    assert store.entries[:4] == WORDS[:4] and len(store) == len(WORDS)

    other = ImmichMLClient(ml_url, "ViT-B-32__openai")
    with pytest.raises(SystemExit):
        generate_store(other, WORDS, tmp_path, log=lambda m: None)
    store = generate_store(other, WORDS[:2], tmp_path, rebuild=True, log=lambda m: None)
    assert len(store) == 2 and store.model == "ViT-B-32__openai"
    client.close()
    other.close()


def test_client_errors(ml_url):
    with pytest.raises(MLClientError):
        ImmichMLClient("http://127.0.0.1:9", MODEL, retries=1).ping()
    bad = ImmichMLClient(ml_url, MODEL)
    bad._entries = '{"facial-recognition":{}}'
    with pytest.raises(MLClientError):
        bad.encode("plage")  # 400 from the server is not retried
    bad.close()


def test_build_vocabulary_orders_and_filters():
    pytest.importorskip("wordfreq")
    stop = load_stopwords()
    vocab = build_vocabulary(["fr", "en"], 500, stop)
    assert vocab[0] == "plage" and "coucher de soleil" in vocab and "sunset" in vocab
    assert all(w == w.lower() for w in vocab)
    assert not (set(vocab) & {"le", "the", "de", "of"})
    assert len(vocab) == len(set(vocab))
    assert "chien" in vocab and len(build_vocabulary(["fr"], 0, stop, include_builtin=False)) == 0


def test_cli_dry_run_and_explain(ml_url, tmp_path, capsys):
    pytest.importorskip("wordfreq")
    assert main(["generate", "--model", MODEL, "--data", str(tmp_path), "--top-n", "50", "--dry-run"]) == 0
    assert "Vocabulary:" in capsys.readouterr().err
    assert main(["generate", "--model", MODEL, "--ml-url", ml_url, "--data", str(tmp_path), "--top-n", "0", "--limit", "20", "--workers", "2"]) == 0
    assert main(["explain", "--data", str(tmp_path), "plage", "au", "coucher", "de", "soleil"]) == 0
    assert "plage" in capsys.readouterr().out
    assert main(["evaluate", "--model", MODEL, "--ml-url", ml_url, "--data", str(tmp_path)]) == 0
    assert "queries:" in capsys.readouterr().out
