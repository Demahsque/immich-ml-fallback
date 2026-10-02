"""Approximate a CLIP text embedding from precomputed word vectors ("bag of words").

The query is split into words; known phrases ("coucher de soleil") are matched first, then
stop words are dropped, unknown words are ignored, and the remaining vectors are summed and
re-normalised. No model is loaded: this is a lookup plus a vector addition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .store import VectorStore
from .text import fold, singular_variants, tokenize


class NoKnownWordsError(ValueError):
    """None of the words of the query is in the vocabulary."""


@dataclass(frozen=True)
class Analysis:
    """How a query was understood."""

    rows: tuple[int, ...]  # vocabulary rows that will be summed
    used: tuple[str, ...]  # matching entries (words or phrases)
    stopped: tuple[str, ...]  # stop words that were dropped
    unknown: tuple[str, ...]  # words that are not in the vocabulary


@dataclass(frozen=True)
class EncodeResult:
    vector: np.ndarray  # float32, unit length
    analysis: Analysis


class Lexicon:
    """Exact, accent-insensitive and plural-tolerant lookup over the vocabulary entries."""

    def __init__(self, entries: Sequence[str]) -> None:
        self.entries = entries
        self._exact: dict[str, int] = {}
        self._folded: dict[str, int] = {}
        self.max_phrase_len = 1
        for row, entry in enumerate(entries):
            self._exact.setdefault(entry, row)
            self._folded.setdefault(fold(entry), row)
            self.max_phrase_len = max(self.max_phrase_len, entry.count(" ") + 1)

    def lookup_phrase(self, tokens: Sequence[str]) -> int | None:
        key = " ".join(tokens)
        row = self._exact.get(key)
        return row if row is not None else self._folded.get(fold(key))

    def lookup_word(self, token: str) -> int | None:
        for candidate in singular_variants(token):
            row = self._exact.get(candidate)
            if row is not None:
                return row
        for candidate in singular_variants(fold(token)):
            row = self._folded.get(candidate)
            if row is not None:
                return row
        return None


class ApproxEncoder:
    def __init__(self, store: VectorStore, stopwords: frozenset[str]) -> None:
        self.store = store
        self.stopwords = stopwords
        self.lexicon = Lexicon(store.entries)

    def analyse(self, query: str) -> Analysis:
        tokens = tokenize(query)
        rows: list[int] = []
        stopped: list[str] = []
        unknown: list[str] = []
        i = 0
        while i < len(tokens):
            span = self._match_phrase(tokens, i)
            if span is not None:
                row, length = span
                rows.append(row)
                i += length
                continue
            token = tokens[i]
            i += 1
            if token in self.stopwords:
                stopped.append(token)
                continue
            row = self.lexicon.lookup_word(token)
            if row is None:
                unknown.append(token)
            else:
                rows.append(row)
        used = tuple(self.store.entries[r] for r in rows)
        return Analysis(rows=tuple(rows), used=used, stopped=tuple(stopped), unknown=tuple(unknown))

    def encode(self, query: str) -> EncodeResult:
        analysis = self.analyse(query)
        if not analysis.rows:
            raise NoKnownWordsError(
                "no word of the query is in the fallback vocabulary "
                f"(ignored stop words: {list(analysis.stopped)}, unknown words: {list(analysis.unknown)})"
            )
        summed = self.store.vectors[list(analysis.rows)].astype(np.float32).sum(axis=0)
        norm = float(np.linalg.norm(summed))
        if not np.isfinite(norm) or norm == 0.0:
            raise NoKnownWordsError("the word vectors of this query cancel each other out")
        return EncodeResult(vector=np.ascontiguousarray(summed / norm), analysis=analysis)

    def _match_phrase(self, tokens: Sequence[str], start: int) -> tuple[int, int] | None:
        longest = min(self.lexicon.max_phrase_len, len(tokens) - start)
        for length in range(longest, 1, -1):
            row = self.lexicon.lookup_phrase(tokens[start : start + length])
            if row is not None:
                return row, length
        return None
