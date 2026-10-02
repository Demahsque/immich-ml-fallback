"""FastAPI proxy that imitates the part of the Immich ML API used by Smart Search.

Implemented (same wire format as Immich ML, checked against Immich v3.2.4):

* ``GET /ping``     -> ``pong`` (503 while no vectors are loaded);
* ``POST /predict`` -> multipart form with ``entries`` = ``{"clip":{"textual":{"modelName":...}}}``
  and ``text``; answers ``{"clip": "<JSON array of floats, as a string>"}``.

Everything else (image encoding, faces, OCR) is refused with an explicit error so that
Immich moves on to its next ML server instead of getting a wrong answer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.datastructures import UploadFile

from . import __version__
from .encoder import ApproxEncoder, NoKnownWordsError
from .stopwords import load_stopwords
from .store import INDEX_FILE, STOPWORDS_FILE, VECTORS_FILE, StoreError, VectorStore, clean_name

log = logging.getLogger("immich_ml_fallback")


class ApiError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class EncoderHolder:
    """Loads the vectors lazily and reloads them when the files change on disk.

    The proxy therefore starts even if the vectors are not there yet, and picks up a new
    ``vectors.npy`` / ``index.json`` pair copied from the Mac without a restart. A pair that
    does not load (e.g. mid-copy) is ignored and the previous vectors keep serving.
    """

    def __init__(self, data_dir: Path, recheck_seconds: float = 2.0) -> None:
        self.data_dir = data_dir
        self.recheck_seconds = recheck_seconds
        self.error: str | None = None
        self._encoder: ApproxEncoder | None = None
        self._stamp: tuple[Any, ...] | None = None
        self._checked = 0.0
        self._lock = threading.Lock()

    def _current_stamp(self) -> tuple[Any, ...]:
        stamp = []
        for name in (INDEX_FILE, VECTORS_FILE, STOPWORDS_FILE):
            try:
                st = (self.data_dir / name).stat()
                stamp.append((st.st_mtime_ns, st.st_size))
            except OSError:
                stamp.append(None)
        return tuple(stamp)

    def get(self) -> ApproxEncoder | None:
        now = time.monotonic()
        if self._encoder is not None and now - self._checked < self.recheck_seconds:
            return self._encoder
        with self._lock:
            self._checked = now
            stamp = self._current_stamp()
            if stamp != self._stamp:
                self._stamp = stamp
                self._reload()
            return self._encoder

    def _reload(self) -> None:
        try:
            store = VectorStore.load(self.data_dir)
            encoder = ApproxEncoder(store, load_stopwords(self.data_dir / STOPWORDS_FILE))
        except StoreError as exc:
            self.error = str(exc)
            log.error("Cannot load vectors from %s: %s", self.data_dir, exc)
            return
        self._encoder, self.error = encoder, None
        log.info(
            "Loaded %d entries (dim %d) for model '%s' from %s",
            len(store), store.dim, store.model, self.data_dir,
        )


def _parse_entries(raw: str) -> str:
    """Return the CLIP model name if `raw` is a ``clip/textual`` request, else raise ApiError."""
    try:
        entries = orjson.loads(raw)
    except orjson.JSONDecodeError:
        raise ApiError(422, "Invalid request format.") from None
    if not isinstance(entries, dict):
        raise ApiError(422, "Invalid request format.")
    clip = entries.get("clip")
    if set(entries) != {"clip"} or not isinstance(clip, dict) or set(clip) != {"textual"}:
        raise ApiError(
            501,
            "This fallback proxy only supports CLIP text encoding (clip/textual) for Smart Search; "
            "the real Immich ML server must handle everything else.",
        )
    textual = clip["textual"]
    name = textual.get("modelName") if isinstance(textual, dict) else None
    if not isinstance(name, str) or not name:
        raise ApiError(422, "Invalid request format.")
    return name


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    directory = Path(data_dir if data_dir is not None else os.environ.get("DATA_DIR", "/data"))
    holder = EncoderHolder(directory)
    app = FastAPI(title="immich-ml-fallback", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.holder = holder

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"message": "Immich ML fallback proxy (text search only)"})

    @app.get("/ping")
    def ping() -> PlainTextResponse:
        if holder.get() is None:
            return PlainTextResponse(holder.error or "no vectors loaded", status_code=503)
        return PlainTextResponse("pong")

    @app.get("/info")
    def info() -> JSONResponse:
        encoder = holder.get()
        if encoder is None:
            return JSONResponse({"loaded": False, "error": holder.error, "data_dir": str(directory)}, status_code=503)
        store = encoder.store
        return JSONResponse(
            {
                "loaded": True,
                "version": __version__,
                "model": store.model,
                "dim": store.dim,
                "entries": len(store),
                "phrases": sum(1 for e in store.entries if " " in e),
                "meta": store.meta,
            }
        )

    @app.get("/explain")
    def explain(q: str) -> JSONResponse:
        """Debug helper: which words of `q` are used, dropped or unknown."""
        encoder = holder.get()
        if encoder is None:
            raise ApiError(503, holder.error or "no vectors loaded")
        a = encoder.analyse(q)
        return JSONResponse({"used": a.used, "stop_words": a.stopped, "unknown": a.unknown})

    @app.post("/predict")
    async def predict(request: Request) -> Response:
        started = time.perf_counter()
        form = await request.form()
        entries = form.get("entries")
        if not isinstance(entries, str):
            raise ApiError(422, "Invalid request format.")
        requested_model = _parse_entries(entries)
        if isinstance(form.get("image"), UploadFile):
            raise ApiError(400, "Image input is not supported by the fallback proxy.")
        text = form.get("text")
        if not isinstance(text, str):
            raise ApiError(400, "Either image or text must be provided")

        encoder = holder.get()
        if encoder is None:
            raise ApiError(503, f"No vectors loaded: {holder.error or 'run the generator and copy the data directory'}")
        if clean_name(requested_model) != clean_name(encoder.store.model):
            raise ApiError(
                409,
                f"Vectors were generated for model '{encoder.store.model}' but Immich asked for "
                f"'{requested_model}'. Regenerate the vectors or change the CLIP model back.",
            )
        try:
            result = encoder.encode(text)
        except NoKnownWordsError as exc:
            log.info("query %r -> no usable word", text)
            raise ApiError(422, str(exc)) from None

        a = result.analysis
        log.info(
            "query %r -> used=%s stop=%s unknown=%s (%.1f ms)",
            text, list(a.used), list(a.stopped), list(a.unknown), (time.perf_counter() - started) * 1000,
        )
        embedding = orjson.dumps(result.vector, option=orjson.OPT_SERIALIZE_NUMPY).decode()
        return Response(orjson.dumps({"clip": embedding}), media_type="application/json")

    return app


app = create_app()
