FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Install system build dependencies for scientific packages and the frontend build
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    liblapacke-dev \
    pkg-config \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache-buster: bump this value whenever the model corpus must be re-fetched.
# Render's Docker builds reuse layer cache, and a stale cache previously skipped
# the weights-download step below; this ENV invalidates that cache chain.
ENV MODEL_CACHEBUST=1

# Copy requirements and install Python dependencies
COPY backend/requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-baked model artifacts (transformer detector + LR fallback).
# The transformer weights are tracked via git LFS, and Render's clone step does
# not always run the LFS smudge filter, which leaves LFS pointer text on disk.
# The real weights are therefore downloaded from the pinned GitHub release to
# guarantee the image always contains the full 267 MB safetensors file.
COPY backend/models/ models/
RUN for f in $(find models -type f); do head -c 20 "$f" 2>/dev/null | grep -q 'version https://git-lfs' && echo "INFO: $f is an LFS pointer, replacing from release"; done \
 && curl -fsSL -o /tmp/ai_detector_transformer.tar.gz \
      https://github.com/SparshM8/VeriPaper/releases/download/models-v1/ai_detector_transformer.tar.gz \
 && tar xzf /tmp/ai_detector_transformer.tar.gz -C models/ \
 && rm -f /tmp/ai_detector_transformer.tar.gz \
 && ls -la models/ai_detector_transformer/model.safetensors

# Corpus data (arXiv abstracts used by the similarity search)
COPY backend/data/ data/

# Frontend source + build it once (snapshot for the /static mount)
COPY backend/frontend/ frontend/
RUN cd frontend && rm -rf dist && npm ci && npm run build \
 && rm -rf node_modules package-lock.json

# Application code (kept last so code edits don't bust the pip/model cache)
COPY backend/app/ app/

RUN mkdir -p /app/reports

# Expose API port
EXPOSE 8000

# Health check — Render free tier requires a passing check within ~90s
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
