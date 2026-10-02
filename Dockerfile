FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    PORT=3003 \
    LOG_LEVEL=info

WORKDIR /app

# Dependencies first (cached layer). Pinned versions ship wheels for amd64 and arm64.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

RUN useradd --system --uid 10001 --no-create-home app && mkdir -p /data
USER app
VOLUME /data
EXPOSE 3003

# /ping answers 503 until vectors are loaded, so "unhealthy" means "no usable data".
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/ping' % os.environ.get('PORT','3003'), timeout=3)"

CMD ["python", "-m", "immich_ml_fallback", "serve"]
