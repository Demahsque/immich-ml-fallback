"""Compare the fallback embedding with the real one for a list of queries."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .encoder import ApproxEncoder, NoKnownWordsError
from .generate import ImmichMLClient


@dataclass(frozen=True)
class Row:
    query: str
    cosine: float | None  # None: nothing usable in the query
    used: tuple[str, ...]
    unknown: tuple[str, ...]


def default_queries() -> list[str]:
    text = (resources.files("immich_ml_fallback") / "resources" / "test_queries.txt").read_text(encoding="utf-8")
    return read_queries(text)


def read_queries(text: str) -> list[str]:
    return [q for q in (line.split("#", 1)[0].strip() for line in text.splitlines()) if q]


def queries_from_file(path: Path) -> list[str]:
    return read_queries(path.read_text(encoding="utf-8"))


def evaluate(client: ImmichMLClient, encoder: ApproxEncoder, queries: list[str]) -> list[Row]:
    rows: list[Row] = []
    for query in queries:
        true = client.encode(query)
        true = true / np.linalg.norm(true)
        analysis = encoder.analyse(query)
        try:
            approx = encoder.encode(query).vector
            cosine: float | None = float(np.dot(true, approx))
        except NoKnownWordsError:
            cosine = None
        rows.append(Row(query, cosine, analysis.used, analysis.unknown))
    return rows


def format_report(rows: list[Row]) -> str:
    scored = sorted((r for r in rows if r.cosine is not None), key=lambda r: r.cosine)
    lines = [f"{'cosine':>7}  query  [used | unknown]"]
    for r in scored:
        lines.append(f"{r.cosine:7.3f}  {r.query}  [{', '.join(r.used)} | {', '.join(r.unknown)}]")
    for r in rows:
        if r.cosine is None:
            lines.append(f"{'n/a':>7}  {r.query}  [no usable word | {', '.join(r.unknown)}]")
    if scored:
        values = np.array([r.cosine for r in scored])
        lines.append(
            f"\n{len(scored)} queries: mean {values.mean():.3f}, median {np.median(values):.3f}, "
            f"min {values.min():.3f}, max {values.max():.3f}; {len(rows) - len(scored)} without usable word"
        )
    return "\n".join(lines)
