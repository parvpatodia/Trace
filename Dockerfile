FROM python:3.11-slim

# System libs needed by sentence-transformers / torch CPU wheels.
# build-essential is dropped after install to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer cache survives source changes.
COPY pyproject.toml README.md* ./
COPY trace/ trace/

# Optional: skip optional extras (Kalibr, Apify, PRAW, Scalekit, Redis, etc.)
# at build time. The hackathon path doesn't need them and they bloat the image.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --upgrade pip \
    && pip install -e .

# Pre-bake the sentence-transformers model into the image. Without this,
# the first /v2/ingest or /v2/context call would trigger a ~80 MB download
# and a 30+ second cold-start latency.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Drop build deps now that wheels are installed.
RUN apt-get purge -y --auto-remove build-essential

# Cloud Run injects PORT; our config reads it via os.getenv("PORT").
ENV PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# uvicorn binds to 0.0.0.0 so Cloud Run's load balancer can reach it.
CMD exec uvicorn trace.delivery.api:app --host 0.0.0.0 --port ${PORT}
