"""Build the word-vector store on the Mac by asking the *real* Immich ML server.

Instead of re-implementing the CLIP text pipeline, each vocabulary entry is sent to the
Immich ML server exactly like Immich itself does for a search query. The stored vectors are
therefore, by construction, the ones Immich would compute for that word (same model, same
tokenizer, same cleaning, same numerical backend).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

import numpy as np

from .store import VectorStore, clean_name
from .text import normalize_entry, tokenize

BUILTIN_RESOURCES = ("extra_words.txt", "phrases_fr.txt", "phrases_en.txt")


class MLClientError(RuntimeError):
    pass


class ImmichMLClient:
    """Minimal client for ``POST /predict`` with a ``clip/textual`` entry."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 180.0, retries: int = 4, client=None) -> None:
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        self._http = client if client is not None else httpx.Client(timeout=timeout)
        self._entries = json.dumps({"clip": {"textual": {"modelName": model, "options": {}}}}, separators=(",", ":"))

    def close(self) -> None:
        self._http.close()

    def ping(self) -> None:
        try:
            response = self._http.get(f"{self.base_url}/ping")
            response.raise_for_status()
        except self._httpx.HTTPError as exc:
            raise MLClientError(f"Immich ML server not reachable at {self.base_url}: {exc}") from exc

    def encode(self, text: str) -> np.ndarray:
        """Embedding of `text`, as a float32 vector (not normalised)."""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                # (None, value) tuples make httpx send plain multipart fields, like Node's FormData.
                response = self._http.post(
                    f"{self.base_url}/predict",
                    files={"entries": (None, self._entries), "text": (None, text)},
                )
                if 400 <= response.status_code < 500:
                    raise MLClientError(f"{response.status_code} from the ML server: {response.text[:200]}")
                response.raise_for_status()
                payload = response.json()["clip"]
                vector = np.asarray(json.loads(payload) if isinstance(payload, str) else payload, dtype=np.float32)
                if vector.ndim != 1 or vector.size == 0:
                    raise MLClientError(f"unexpected embedding shape {vector.shape}")
                return vector
            except MLClientError:
                raise
            except (self._httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 10))
        raise MLClientError(f"request failed after {self.retries} attempts: {last_error}")


def _read_word_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [e for e in (normalize_entry(line.split("#", 1)[0]) for line in lines) if e]


def _builtin_entries() -> list[str]:
    entries: list[str] = []
    for name in BUILTIN_RESOURCES:
        text = (resources.files("immich_ml_fallback") / "resources" / name).read_text(encoding="utf-8")
        entries += [e for e in (normalize_entry(line.split("#", 1)[0]) for line in text.splitlines()) if e]
    return entries


def build_vocabulary(
    languages: Sequence[str],
    top_n: int,
    stopwords: frozenset[str],
    extra_files: Iterable[Path] = (),
    include_builtin: bool = True,
) -> list[str]:
    """Vocabulary to encode: curated words/phrases first, then the most frequent words."""
    ordered: dict[str, None] = {}
    if include_builtin:
        ordered.update(dict.fromkeys(_builtin_entries()))
    for path in extra_files:
        ordered.update(dict.fromkeys(_read_word_file(Path(path))))
    if top_n > 0 and languages:
        try:
            from wordfreq import top_n_list, zipf_frequency
        except ImportError as exc:
            raise SystemExit("wordfreq is missing: pip install 'immich-ml-fallback[generate]'") from exc
        scored: dict[str, float] = {}
        for lang in languages:
            for word in top_n_list(lang, top_n):
                tokens = tokenize(word)
                if len(tokens) != 1 or len(tokens[0]) < 2 or tokens[0] in stopwords:
                    continue
                scored[tokens[0]] = max(zipf_frequency(tokens[0], code) for code in languages)
        ordered.update(dict.fromkeys(sorted(scored, key=lambda w: (-scored[w], w))))
    return list(ordered)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm == 0.0:
        return None
    return (vector / norm).astype(np.float32)


def generate_store(
    client: ImmichMLClient,
    words: Sequence[str],
    out_dir: Path,
    *,
    workers: int = 4,
    checkpoint: int = 1000,
    rebuild: bool = False,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
) -> VectorStore:
    """Encode the missing `words` and save the store (after every `checkpoint` entries)."""
    entries: list[str] = []
    matrix = np.zeros((0, 0), dtype=np.float16)
    meta: dict = {}
    if not rebuild and (out_dir / "index.json").is_file():
        existing = VectorStore.load(out_dir, mmap=False)
        if clean_name(existing.model) != clean_name(client.model):
            raise SystemExit(
                f"{out_dir} already holds vectors for model '{existing.model}', not '{client.model}'. "
                "Use --rebuild to start over, or another --out directory."
            )
        entries, matrix, meta = list(existing.entries), np.array(existing.vectors), dict(existing.meta)
        log(f"Existing store: {len(entries)} entries, will only add missing ones.")
    known = set(entries)
    todo = [w for w in dict.fromkeys(words) if w not in known]
    if not todo:
        log("Nothing to do: every entry is already in the store.")
        return VectorStore(model=client.model, entries=entries, vectors=matrix, meta=meta)

    meta.update({"source_ml_url": client.base_url})
    started, done = time.monotonic(), 0

    def encode_one(word: str) -> tuple[str, np.ndarray | None]:
        try:
            return word, _unit(client.encode(word))
        except MLClientError as exc:
            log(f"  ! {word!r}: {exc}")
            return word, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(todo), checkpoint):
            chunk = todo[start : start + checkpoint]
            results = list(pool.map(encode_one, chunk))
            good = [(w, v) for w, v in results if v is not None]
            if not good:
                raise SystemExit("Every request of the last batch failed: is the Immich ML server up?")
            dim = good[0][1].shape[0]
            if matrix.size and matrix.shape[1] != dim:
                raise SystemExit(f"Dimension changed ({matrix.shape[1]} -> {dim}): wrong model on the ML server?")
            if any(v.shape[0] != dim for _, v in good):
                raise SystemExit("The ML server returned vectors of different sizes.")
            if not matrix.size:
                matrix = np.zeros((0, dim), dtype=np.float16)
            matrix = np.vstack([matrix, np.stack([v for _, v in good]).astype(np.float16)])
            entries += [w for w, _ in good]
            VectorStore(model=client.model, entries=entries, vectors=matrix, meta=meta).save(out_dir)
            done += len(chunk)
            rate = done / max(time.monotonic() - started, 1e-9)
            log(f"{done}/{len(todo)} encoded ({rate:.1f}/s, ~{(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min left)")
    return VectorStore(model=client.model, entries=entries, vectors=matrix, meta=meta)
