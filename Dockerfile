FROM python:3.12-slim

# System deps for spaCy and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt


# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY v24.py app.py /app/
COPY rag/ /app/rag/
COPY scripts/ /app/scripts/

# Copy the pre-ingested database (downloaded by CI from GitHub Releases)
COPY data/hybrid_rag_v22.db /app/data/hybrid_rag_v22.db
COPY data/questionaskerbigset.parquet /app/data/questionaskerbigset.parquet

EXPOSE 7860

# Serve the Gradio web UI
CMD ["python", "app.py"]
