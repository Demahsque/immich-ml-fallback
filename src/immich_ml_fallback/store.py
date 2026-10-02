"""On-disk format of the precomputed vectors, shared by the generator and the proxy.

Two files live in the data directory:

* ``vectors.npy`` - ``(n, dim)`` float16 matrix, one L2-normalised row per entry;
* ``index.json``  - model name, dimension and the ``n`` entries (words or phrases) in row order.

``index.json`` is always written last, so a reader that sees a new index also sees its vectors.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__

FORMAT_VERSION = 1
INDEX_FILE = "index.json"
VECTORS_FILE = "vectors.npy"
STOPWORDS_FILE = "stopwords.txt"

_RESERVED_KEYS = {"format", "model", "dim", "dtype", "count", "generator", "entries"}
_CLEAN_NAME = str.maketrans(":\\/", "___", ".")


def clean_name(model_name: str) -> str:
    """Same normalisation as ``clean_name`` in Immich ML (machine-learning/immich_ml/config.py)."""
    return model_name.split("/")[-1].translate(_CLEAN_NAME)


class StoreError(RuntimeError):
    """The data directory is missing, incomplete or inconsistent."""


@dataclass
class VectorStore:
    model: str
    entries: list[str]
    vectors: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise StoreError(f"vectors must be a 2-D matrix, got shape {self.vectors.shape}")
        if self.vectors.dtype not in (np.float16, np.float32):
            raise StoreError(f"vectors must be float16 or float32, got {self.vectors.dtype}")
        if self.vectors.shape[0] != len(self.entries):
            raise StoreError(
                f"{self.vectors.shape[0]} vectors but {len(self.entries)} entries: "
                f"{VECTORS_FILE} and {INDEX_FILE} do not belong together"
            )
        if len(set(self.entries)) != len(self.entries):
            raise StoreError("duplicate entries in the index")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, directory: Path | str, *, mmap: bool = True) -> VectorStore:
        directory = Path(directory)
        index_path = directory / INDEX_FILE
        vectors_path = directory / VECTORS_FILE
        for path in (index_path, vectors_path):
            if not path.is_file():
                raise StoreError(f"{path} not found")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise StoreError(f"cannot read the vectors in {directory}: {exc}") from exc
        if not isinstance(index, dict) or index.get("format") != FORMAT_VERSION:
            raise StoreError(f"unsupported {INDEX_FILE} format (expected {FORMAT_VERSION})")
        entries, model = index.get("entries"), index.get("model")
        if not isinstance(entries, list) or not isinstance(model, str) or not model:
            raise StoreError(f"{INDEX_FILE} must contain 'model' and 'entries'")
        if index.get("dim") != (vectors.shape[1] if vectors.ndim == 2 else None):
            raise StoreError(f"{INDEX_FILE} dim={index.get('dim')} does not match {VECTORS_FILE} {vectors.shape}")
        meta = {k: v for k, v in index.items() if k not in _RESERVED_KEYS}
        return cls(model=model, entries=entries, vectors=vectors, meta=meta)

    def save(self, directory: Path | str) -> None:
        """Atomically (re)write both files; the index goes last."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        meta = {"created": now, **self.meta, "updated": now}
        index = {
            "format": FORMAT_VERSION,
            "model": self.model,
            "dim": self.dim,
            "dtype": str(self.vectors.dtype),
            "count": len(self.entries),
            "generator": f"immich-ml-fallback {__version__}",
            **meta,
            "entries": self.entries,
        }
        tmp_vectors = directory / (VECTORS_FILE + ".tmp")
        with open(tmp_vectors, "wb") as handle:
            np.save(handle, np.ascontiguousarray(self.vectors), allow_pickle=False)
        os.replace(tmp_vectors, directory / VECTORS_FILE)
        tmp_index = directory / (INDEX_FILE + ".tmp")
        tmp_index.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_index, directory / INDEX_FILE)
