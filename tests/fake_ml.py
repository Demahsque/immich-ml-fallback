"""A tiny stand-in for the Immich ML server: same wire format, 'bag of words' embeddings."""

from __future__ import annotations

import json

import orjson
from fastapi import FastAPI, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse

from conftest import bag_of_words_embedding

app = FastAPI()
calls: list[str] = []


@app.get("/ping")
def ping():
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(entries: str = Form(), text: str | None = Form(default=None), image: bytes | None = File(default=None)):
    request = json.loads(entries)
    if "clip" not in request or text is None:
        return JSONResponse({"detail": "unsupported"}, status_code=400)
    calls.append(text)
    embedding = orjson.dumps(bag_of_words_embedding(text), option=orjson.OPT_SERIALIZE_NUMPY).decode()
    return JSONResponse({"clip": embedding})
