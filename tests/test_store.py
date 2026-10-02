import json

import numpy as np
import pytest

from immich_ml_fallback.store import INDEX_FILE, VECTORS_FILE, StoreError, VectorStore, clean_name


def test_roundtrip(tmp_path, store):
    store.meta["note"] = "hello"
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert loaded.model == store.model and loaded.entries == store.entries
    assert loaded.dim == store.dim and loaded.vectors.dtype == np.float16
    assert loaded.meta["note"] == "hello" and "created" in loaded.meta
    assert isinstance(loaded.vectors, np.memmap)


def test_missing_directory(tmp_path):
    with pytest.raises(StoreError):
        VectorStore.load(tmp_path / "nope")


def test_mismatch_between_index_and_vectors(tmp_path, store):
    store.save(tmp_path)
    index = json.loads((tmp_path / INDEX_FILE).read_text())
    index["entries"] = index["entries"][:-1]
    (tmp_path / INDEX_FILE).write_text(json.dumps(index))
    with pytest.raises(StoreError):
        VectorStore.load(tmp_path)


def test_wrong_dim_and_format(tmp_path, store):
    store.save(tmp_path)
    index = json.loads((tmp_path / INDEX_FILE).read_text())
    (tmp_path / INDEX_FILE).write_text(json.dumps({**index, "dim": 3}))
    with pytest.raises(StoreError):
        VectorStore.load(tmp_path)
    (tmp_path / INDEX_FILE).write_text(json.dumps({**index, "format": 99}))
    with pytest.raises(StoreError):
        VectorStore.load(tmp_path)


def test_corrupt_vectors_file(tmp_path, store):
    store.save(tmp_path)
    (tmp_path / VECTORS_FILE).write_bytes(b"not a npy file")
    with pytest.raises(StoreError):
        VectorStore.load(tmp_path)


def test_duplicate_entries_rejected(store):
    with pytest.raises(StoreError):
        VectorStore(model="m", entries=["a", "a"], vectors=np.zeros((2, 4), dtype=np.float16))


def test_clean_name_matches_immich():
    assert clean_name("ViT-SO400M-16-SigLIP2-384__webli") == "ViT-SO400M-16-SigLIP2-384__webli"
    assert clean_name("immich-app/ViT-B-32__openai") == "ViT-B-32__openai"
    assert clean_name("xlm-roberta-large-ViT-H-14__frozen_laion5b_s13b_b90k") == "xlm-roberta-large-ViT-H-14__frozen_laion5b_s13b_b90k"
