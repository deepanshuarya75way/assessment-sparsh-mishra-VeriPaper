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

# Copy requirements under a distinct name and pin a unique build marker so that
# the pip install layer can never be served from a stale cache entry.
# Render's BuildKit cache previously reused a cached pip layer from April 2026
# (full GPU torch, no onnxruntime) across later builds, breaking the AI engine;
# both the input hash (deps-onnx.txt) and the command string (build marker)
# are now unique per revision, and the ARG before the layer further forces a
# cache miss whenever bumped.
ARG PY_DEP_BUILD_MARKER=render-deps-v5-$(date +%s)
COPY backend/requirements.txt deps-onnx.txt
COPY pyproject.toml ./

RUN echo "DEP_BUILD_MARKER=$PY_DEP_BUILD_MARKER" && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r deps-onnx.txt && \
    pip install --no-cache-dir --force-reinstall "onnxruntime==1.18.1" && \
    python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"

# Pre-baked model artifacts (LR fallback only).
# The quantized ONNX transformer graph (65 MB) is too heavy for Render's clone
# step and its Docker build cache quirks, so it is fetched here from the pinned
# GitHub release instead of being shipped through git.
COPY backend/models/ai_detector.joblib models/
COPY backend/models/ai_detector_metrics.json models/
COPY backend/models/ai_detector_winner.json models/
# Public release download URLs on this repo have been observed to return 404
# after assets are re-issued. Try the v2 public URL first, then fall back to
# the authenticated GitHub API asset endpoint when GH_TOKEN is provided.
ARG GH_TOKEN=
RUN mkdir -p models/ai_detector_transformer \
 && (curl -fsSL -o /tmp/ai_detector_transformer_onnx.tar.gz \
      https://github.com/SparshM8/VeriPaper/releases/download/models-onnx-v2/ai_detector_transformer_onnx.tar.gz \
 || curl -fsSL -o /tmp/ai_detector_transformer_onnx.tar.gz \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "Accept: application/octet-stream" \
      https://api.github.com/repos/SparshM8/VeriPaper/releases/assets/522326158) \
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
