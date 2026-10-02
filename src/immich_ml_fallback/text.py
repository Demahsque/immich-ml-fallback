"""Text helpers shared by the generator, the evaluator and the proxy.

Everything that decides "what is a word" lives here, so the vocabulary built on the
Mac and the queries received by the proxy are always tokenised the same way.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# A token is a run of Unicode letters. Digits, underscores, punctuation, apostrophes
# and hyphens are separators: "l'enfant" -> "l", "enfant"; "arc-en-ciel" -> "arc", "en", "ciel".
_TOKEN_RE = re.compile(r"[^\W\d_]+")
_SPECIAL_LETTERS = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "đ": "d", "ł": "l"})


def tokenize(text: str) -> list[str]:
    """Lower-case `text` and split it into letter-only tokens."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).casefold())


def normalize_entry(entry: str) -> str:
    """Canonical form of a vocabulary entry: its tokens joined by single spaces."""
    return " ".join(tokenize(entry))


def fold(text: str) -> str:
    """Strip accents so that 'forêt' and 'foret' compare equal."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_SPECIAL_LETTERS))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def singular_variants(token: str) -> Iterator[str]:
    """Yield `token`, then crude singular candidates (French and English plurals).

    Only a fallback for words missing from the vocabulary: the caller tries each
    candidate in turn and keeps the first one it knows.
    """
    yield token
    if len(token) <= 3 or not token.endswith(("s", "x")):
        return
    if token.endswith("aux"):  # chevaux -> cheval
        yield token[:-3] + "al"
    if token.endswith("ies"):  # cities -> city
        yield token[:-3] + "y"
    if token.endswith("ves"):  # wolves -> wolf, knives -> knife
        yield token[:-3] + "f"
        yield token[:-3] + "fe"
    yield token[:-1]  # chiens -> chien, cakes -> cake, bijoux -> bijou
    if token.endswith("es"):  # boxes -> box
        yield token[:-2]
