FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BIOEVIDENCE_DATA_DIR=/app/data/corpora/demo \
    BIOEVIDENCE_EMBEDDING_CACHE_DIR=/tmp/bioevidence-cache

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir ".[serve]"

COPY interfaces ./interfaces
COPY data ./data

RUN mkdir -p /tmp/bioevidence-cache \
    && chown -R appuser:appuser /app /tmp/bioevidence-cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).read()"

CMD ["uvicorn", "interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
