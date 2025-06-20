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
ENV MODEL_CACHEBUST=3

# Copy requirements and install Python dependencies
COPY backend/requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "onnxruntime==1.18.1" && \
    pip install --no-cache-dir --force-reinstall "torch==2.4.0+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu --no-deps

# Pre-baked model artifacts (LR fallback only).
# The quantized ONNX transformer graph (65 MB) is too heavy for Render's clone
# step and its Docker build cache quirks, so it is fetched here from the pinned
# GitHub release instead of being shipped through git.
COPY backend/models/ai_detector.joblib models/
COPY backend/models/ai_detector_metrics.json models/
COPY backend/models/ai_detector_winner.json models/
RUN mkdir -p models/ai_detector_transformer \
 && curl -fsSL -o /tmp/ai_detector_transformer_onnx.tar.gz \
      https://github.com/SparshM8/VeriPaper/releases/download/models-onnx-v1/ai_detector_transformer_onnx.tar.gz \
 && tar xzf /tmp/ai_detector_transformer_onnx.tar.gz -C models/ai_detector_transformer/ \
 && rm -f /tmp/ai_detector_transformer_onnx.tar.gz \
 && ls -la models/ai_detector_transformer/model_quantized.onnx

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
