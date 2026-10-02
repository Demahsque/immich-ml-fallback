import numpy as np
import pytest

from conftest import word_vector
from immich_ml_fallback.encoder import ApproxEncoder, NoKnownWordsError
from immich_ml_fallback.stopwords import load_stopwords


@pytest.fixture
def encoder(store):
    return ApproxEncoder(store, load_stopwords())


def cos(a, b):
    return float(np.dot(a, b))


def test_single_word_returns_its_vector(encoder):
    result = encoder.encode("plage")
    assert cos(result.vector, word_vector("plage")) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(result.vector) == pytest.approx(1.0, abs=1e-5)
    assert result.vector.dtype == np.float32


def test_sum_and_normalise(encoder):
    result = encoder.encode("chat et chien")
    expected = word_vector("chat") + word_vector("chien")
    expected /= np.linalg.norm(expected)
    assert cos(result.vector, expected) == pytest.approx(1.0, abs=1e-3)
    assert result.analysis.stopped == ("et",)


def test_stopwords_and_unknown_words_are_ignored(encoder):
    a = encoder.analyse("Un chat qui regarde la mer")
    assert a.used == ("chat",)
    assert a.stopped == ("un", "qui", "la")
    assert a.unknown == ("regarde", "mer")


def test_phrase_longest_match_wins(encoder):
    a = encoder.analyse("un beau coucher de soleil")
    assert a.used == ("coucher de soleil",)
    assert a.unknown == ("beau",)
    assert encoder.analyse("arc-en-ciel").used == ("arc en ciel",)


def test_accent_insensitive_and_plural(encoder):
    assert encoder.analyse("foret").used == ("forêt",)
    assert encoder.analyse("FORÊTS").used == ("forêt",)
    assert encoder.analyse("chevaux").used == ("cheval",)
    assert encoder.analyse("chiens").used == ("chien",)


def test_repeated_word_counts_twice(encoder):
    assert encoder.analyse("chat chat").rows == (encoder.analyse("chat").rows[0],) * 2


def test_no_known_word_raises(encoder):
    for query in ("", "   ", "le la les", "xyzzy", "2019"):
        with pytest.raises(NoKnownWordsError):
            encoder.encode(query)


def test_opposite_vectors_cancel_out(store):
    store.vectors[1] = -store.vectors[0]
    encoder = ApproxEncoder(store, frozenset())
    with pytest.raises(NoKnownWordsError):
        encoder.encode("plage chat")
