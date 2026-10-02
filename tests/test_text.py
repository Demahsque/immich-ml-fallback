from immich_ml_fallback.stopwords import load_stopwords
from immich_ml_fallback.text import fold, normalize_entry, singular_variants, tokenize


def test_tokenize_splits_apostrophes_hyphens_digits():
    assert tokenize("L'enfant à Zürich, 2019!") == ["l", "enfant", "à", "zürich"]
    assert tokenize("Arc-en-ciel") == ["arc", "en", "ciel"]
    assert tokenize("") == []


def test_normalize_entry_matches_query_tokenisation():
    assert normalize_entry("feu d'artifice") == "feu d artifice"
    assert normalize_entry("  Coucher   de SOLEIL ") == "coucher de soleil"


def test_fold_strips_accents_and_ligatures():
    assert fold("forêt") == "foret"
    assert fold("œuf") == "oeuf"
    assert fold("Zürich") == "Zurich"


def test_singular_variants():
    assert "chien" in list(singular_variants("chiens"))
    assert "cheval" in list(singular_variants("chevaux"))
    assert "city" in list(singular_variants("cities"))
    assert "box" in list(singular_variants("boxes"))
    assert "cake" in list(singular_variants("cakes"))
    assert list(singular_variants("bus")) == ["bus"]  # too short to strip


def test_stopwords_builtin_and_extra_file(tmp_path):
    words = load_stopwords()
    assert {"le", "de", "à", "the", "of"} <= words
    assert "plage" not in words and "car" not in words
    extra = tmp_path / "stop.txt"
    extra.write_text("photo  # noise\nimage\n", encoding="utf-8")
    assert {"photo", "image"} <= load_stopwords(extra)
