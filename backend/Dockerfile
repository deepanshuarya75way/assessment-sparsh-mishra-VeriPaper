FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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

# Copy requirements and install Python dependencies
COPY backend/requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-baked model artifacts (transformer detector + LR fallback)
COPY backend/models/ models/

# Corpus data (arXiv abstracts used by the similarity search)
COPY backend/data/ data/

# Frontend source + build it once (snapshot for the /static mount)
COPY frontend/ frontend/
RUN cd frontend && npm ci --ignore-scripts && npm run build \
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
