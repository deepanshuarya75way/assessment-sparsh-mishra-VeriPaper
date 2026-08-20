# VeriPaper — Research Paper Verification Platform

**A production-ready, full-stack platform that verifies the authenticity and integrity of research papers across five independent analysis modules.**

[**Live Demo**](https://veripaper.onrender.com) · [API Docs](https://veripaper.onrender.com/docs) · [Quick Start](QUICKSTART.md) · [Development Guide](DEVELOPMENT.md)

VeriPaper goes beyond simple AI and plagiarism checks. It performs a comprehensive verification of research papers, covering **AI-generated content detection**, **plagiarism analysis**, **citation validation**, **statistical integrity**, and **academic writing standards** (IEEE / IMRaD) — and returns a single, easy-to-read credibility verdict with detailed per-module breakdowns.

![VeriPaper](https://veripaper.onrender.com/static/og.png)

## Why VeriPaper

Manuscripts face scrutiny from editors, peer reviewers, and increasingly, automated desk-reject systems. VeriPaper combines a **fine-tuned transformer model deployed as a quantized ONNX runtime** with deterministic NLP pipelines so that a single PDF, DOCX, or TXT upload yields a complete verification report — scores, flagged excerpts, invalid citations, statistical red flags, and writing-standards findings — in under two seconds of pure inference on free-tier hardware.

## Analysis Modules

| # | Module | Technique | What It Checks |
|---|--------|-----------|----------------|
| 1 | **AI Detection** | Fine-tuned DistilBERT → int8 quantized ONNX (65 MB, torch-free) | Whether the manuscript was written by a human or an LLM, paragraph by paragraph |
| 2 | **Plagiarism** | TF-IDF vectorization over a 2,500-paper arXiv corpus | Textual overlap and semantic similarity against known papers |
| 3 | **Citation Validation** | CrossRef API lookup | Whether referenced DOIs actually exist and metadata matches |
| 4 | **Statistical Integrity** | Pattern analysis | Suspicious p-values, repeated decimal patterns, implausible statistics |
| 5 | **Writing Standards** | Structural heuristics | IEEE / IMRaD structure compliance, section presence, academic conventions |

Results are combined into an **Overall Research Credibility** score (0–100), and every module returns its own score, explanations, and — where applicable — the actual suspicious passages so you can verify the findings instead of taking them on faith.

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/SparshM8/VeriPaper.git && cd VeriPaper

# Backend (Terminal 1)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Frontend (Terminal 2)
cd ../frontend && npm install && npm run dev
```

Then open `http://localhost:5173`. On Windows you can instead run `start-dev.bat` from the project root.

See [QUICKSTART.md](QUICKSTART.md) for the 5-minute guide and [DEVELOPMENT.md](DEVELOPMENT.md) for the full architecture.

## Production Deployment

The platform is already deployed to the **Render free tier** at [veripaper.onrender.com](https://veripaper.onrender.com) — the same single Dockerfile in this repo builds everything:

```bash
docker build -t veripaper .
docker run -p 8000:8000 --env-file backend/.env veripaper
```

Deployment notes for free-tier hosting:

- The **transformer engine runs entirely on CPU via ONNX Runtime** with memory arenas disabled, holding a flat ~342 MB RSS even under 5+ concurrent analyses — well inside a 512 MB container.
- An **asynchronous analysis queue** (`POST /api/analyze/async` + status polling) returns a task ID immediately so long papers never block the client.
- **Sentry** (EU region) captures frontend exceptions in real time, and **UptimeRobot** keeps the free-tier instance warm with 5-minute health pings.
- The in-memory task store is intentional for a single-instance free deployment; swap in Redis/DB persistence for multi-instance scaling.

## Features

| Area | Capabilities |
|------|-------------|
| **Input formats** | PDF, DOCX, TXT (up to 15 MB) |
| **AI detection** | Fine-tuned DistilBERT (F1 0.797), int8 quantized ONNX, paragraph-level verdicts with confidence |
| **Plagiarism** | TF-IDF similarity against a curated arXiv corpus, top match highlighting |
| **Citations** | DOI extraction, live CrossRef validation, year-mismatch detection |
| **Statistics** | Red-flag patterns in p-values and repeated decimals |
| **Writing quality** | IEEE / IMRaD structural compliance scoring |
| **Frontend** | React 18 + Vite + Tailwind, live score gauges, dark/light theme, responsive |
| **Async queue** | Non-blocking analysis with queued → processing → done progress states |
| **History** | Full per-analysis record replay from server storage, with an ErrorBoundary crash fallback |
| **Reports** | PDF, CSV, and JSON export for every analysis |
| **Observability** | Sentry error tracking, health endpoints, stress-tested (10/10 concurrent requests) |

## API Quick Reference

Interactive Swagger UI: `https://veripaper.onrender.com/docs` (rate-limited to 20 req/min).

```bash
# Health
GET /api/health

# Synchronous analysis (small papers)
POST /api/analyze          # multipart/form-data, field: file

# Asynchronous analysis (recommended)
POST /api/analyze/async    # returns { "task_id": "..." } immediately
GET  /api/analyze/{task_id}/status   # polls queued → processing → done/error

# History
GET /api/history
GET /api/history/{id}

# Engine status
GET /api/detector/status
GET /api/detector/config
```

A full analysis response includes per-module scores, credibility verdict, flagged paragraphs, invalid DOIs, writing-standards findings, and a downloadable report path.

## Test Suite

```bash
cd backend
python -m pytest backend/tests/ -q    # 20/20 passing (API + service tests)
python backend/scripts/verify_api.py backend/data/sample_paper.txt
```

Stress testing (5 concurrent × 2 uploads, production): **10/10 successes, 100% transformer-finetuned engine, zero heuristic fallbacks**, ~409 MB RSS with comfortable headroom under the 512 MB cap.

## Project Structure

```
VeriPaper/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py          # FastAPI app, lifespan (model warm-up), CORS
│   │   ├── api/routes.py    # Sync + async analysis, history, detector status
│   │   └── services/        # ai_detection.py, plagiarism.py, citations.py,
│   │                        # statistics.py, writing.py, reports.py
│   ├── frontend/            # Backend-embedded React build (Render builds here)
│   ├── tests/               # pytest: API + service coverage
│   ├── scripts/             # Training, validation, and verification scripts
│   ├── models/              # model_quantized.onnx (int8 DistilBERT, 65 MB)
│   └── requirements.txt
├── frontend/                 # React 18 + Vite source (mirrored to backend/frontend)
│   ├── src/
│   │   ├── App.jsx          # Dashboard, async flow, history replay, ErrorBoundary
│   │   ├── api.js           # HTTP client incl. pollUntilDone
│   │   └── sentry.js        # Sentry SDK initialization
│   └── package.json
├── Dockerfile                # Single-stage production build
├── QUICKSTART.md             # 5-minute setup
├── DEVELOPMENT.md            # Architecture deep-dive
├── DEPLOYMENT.md             # Docker Compose / PostgreSQL deployment
└── README.md                 # This file
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI 0.115, Uvicorn, SQLAlchemy + SQLite |
| ML / Inference | tokenizers (standalone), ONNX Runtime 1.18.1 (CPU), int8 DistilBERT |
| Frontend | React 18.3, Vite 5, Tailwind CSS 3, Recharts, Axios |
| Observability | Sentry (EU), UptimeRobot, Docker health checks |
| Infrastructure | Render free tier, Docker, GitHub Actions CI |

## CI / Quality

GitHub Actions runs the backend test suite and frontend production build on every push and pull request. Commits follow a conventional style and the project enforces reproducible builds via a pinned Dockerfile.

## License

MIT — free for academic and commercial use.

## Contributing

Contributions are welcome. Please read [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details, and run the test suite before opening a pull request. When adding frontend changes, remember that Render builds from `backend/frontend/` — keep both copies in sync.

## Support

- Setup issues → [QUICKSTART.md](QUICKSTART.md)
- Architecture questions → [DEVELOPMENT.md](DEVELOPMENT.md)
- Bugs → open a [GitHub issue](https://github.com/SparshM8/VeriPaper/issues)

---

**Version** 1.0.0 · **Status**: Production — live at [veripaper.onrender.com](https://veripaper.onrender.com)
> **Deployment note (June 2026):** the quantized model weights are delivered at
> build time from the public `SparshM8/veripaper-assets` CDN repository
> (release `models-onnx-v2`), because asset download URLs on this repository
> intermittently return 404. A runtime restore path in `backend/app/main.py`
> fetches the same archive on demand if the image copy is missing.
