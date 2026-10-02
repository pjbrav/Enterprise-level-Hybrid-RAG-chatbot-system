FROM python:3.12-slim

# System deps for spaCy / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer). CPU-only torch keeps the image small.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Application code only. Data is NOT baked into the image: mount ./data at /app/data
# (corpora, state, and optionally the evaluation database + questions parquet).
COPY app.py v24.py setup.py routing_config.example.yaml /app/
COPY rag/ /app/rag/
COPY scripts/ /app/scripts/

ENV DATA_DIR=/app/data \
    SQLITE_DATABASE_PATH=/app/data/hybrid_rag_v22.db \
    GRADIO_BIND=0.0.0.0 \
    HOST_BIND=0.0.0.0 \
    PYTHONUNBUFFERED=1
VOLUME /app/data

# 7860: admin / search UI    8765: read-only host API (only used when you click "Start hosting")
EXPOSE 7860 8765

CMD ["python", "app.py"]
