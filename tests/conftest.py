from __future__ import annotations

import hashlib

import numpy as np
import pytest

from immich_ml_fallback.store import VectorStore
from immich_ml_fallback.stopwords import BUILTIN_STOPWORDS
from immich_ml_fallback.text import tokenize

MODEL = "ViT-SO400M-16-SigLIP2-384__webli"
DIM = 64


def word_vector(word: str, dim: int = DIM) -> np.ndarray:
    """Deterministic pseudo-random unit vector for a word (stand-in for a CLIP word embedding)."""
    seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
    vec = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def bag_of_words_embedding(text: str, dim: int = DIM) -> np.ndarray:
    """What an idealised 'bag of content words' model would answer for a sentence."""
    vec = sum(word_vector(t, dim) for t in tokenize(text) if t not in BUILTIN_STOPWORDS)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


@pytest.fixture
def entries() -> list[str]:
    return ["plage", "chat", "chien", "forêt", "cheval", "ville", "coucher de soleil", "arc en ciel", "bateau", "sunset"]


@pytest.fixture
def store(entries) -> VectorStore:
    vectors = np.stack([word_vector(e) for e in entries]).astype(np.float16)
    return VectorStore(model=MODEL, entries=entries, vectors=vectors)


@pytest.fixture
def data_dir(tmp_path, store):
    store.save(tmp_path)
    return tmp_path
