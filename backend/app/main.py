import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .api.routes import router as api_router
from .core.config import settings
from .core.logging_config import configure_logging

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize and clean up application resources."""
    try:
        from .core.database import init_db

        # Ensure the database parent directory exists (e.g. /app/data on Render)
        if settings.database_url.startswith("sqlite://"):
            db_path = settings.database_url[len("sqlite://") :]
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Don't crash the app - allow degraded mode

    # The transformer weights may be an unresolved git-LFS pointer in the image
    # (Render's clone step does not always smudge LFS files). Restore them from
    # the pinned GitHub release at startup when that happens.
    _ensure_transformer_weights()

    # Load analysis engines (failures keep the app running in degraded mode)
    from .services import ai_detection
    from .services import plagiarism as plagiarism_svc

    if ai_detection.load_trained_model(settings.MODEL_PATH):
        logger.info("✅ Trained AI detector registered (loaded on first analysis request)")
    else:
        logger.warning("No trained AI model available; heuristic engine will be used")
    if plagiarism_svc.load_corpus():
        logger.info("✅ Plagiarism similarity corpus registered (index built on first analysis request)")
    else:
        logger.warning("Similarity corpus not found; internal duplication check only")

    yield

    try:
        from .core.database import close_db

        close_db()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"⚠️ Error closing database: {e}")


_TRANSFORMER_RELEASE_URL = (
    "https://github.com/SparshM8/VeriPaper/releases/download"
    "/models-v1/ai_detector_transformer.tar.gz"
)


def _ensure_transformer_weights() -> None:
    """Download the transformer weights at runtime if the image copy is missing
    or is an unresolved git-LFS pointer (~100 bytes of text)."""
    model_dir = settings.TRANSFORMER_MODEL_DIR
    weights = model_dir / "model.safetensors"
    if weights.exists() and weights.stat().st_size > 1_000_000:
        return
    if not model_dir.exists():
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot create transformer model dir {model_dir}: {e}")
            return
    logger.warning("Transformer weights missing or invalid; downloading from GitHub release...")
    try:
        import io
        import tarfile

        import requests

        resp = requests.get(_TRANSFORMER_RELEASE_URL, timeout=600, stream=True)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
            # The archive wraps files under `ai_detector_transformer/`; extract
            # its members directly into the model directory (strip the prefix).
            members = []
            for member in tar.getmembers():
                name = member.name.split("/", 1)[-1] if "/" in member.name else member.name
                if not name:
                    continue
                member_copy = tarfile.TarInfo(name=name)
                member_copy.size = member.size
                member_copy.mtime = member.mtime
                member_copy.mode = member.mode
                member_copy.type = member.type
                member_copy.linkname = member.linkname
                members.append(member_copy)
            tar.extractall(path=model_dir, members=members)
        if weights.exists() and weights.stat().st_size > 1_000_000:
            logger.info(f"Transformer weights restored at {weights} ({weights.stat().st_size/1e6:.0f} MB)")
        else:
            logger.error("Downloaded archive did not contain valid transformer weights")
    except Exception as e:
        logger.error(f"Failed to download transformer weights at runtime: {e}")


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

# Add CORS middleware BEFORE other routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for PDF reports
reports_dir = settings.REPORTS_DIR
reports_dir.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(reports_dir)), name="static")

# Mount frontend static files under /static to avoid shadowing API routes
frontend_dir = settings.ROOT_DIR / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

app.include_router(api_router)


@app.get("/")
def root():
    """Serve the frontend index if available, otherwise return API info."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(index_path), media_type="text/html")

    return {
        "message": "VeriPaper AI Research Authenticity Platform API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "version": settings.VERSION, "environment": settings.ENVIRONMENT}


@app.get("/ready")
def readiness_check() -> dict:
    from .services import ai_detection as ai_detection_mod
    from .services import plagiarism as plagiarism_mod
    checks = {
        "reports_dir_exists": reports_dir.exists(),
        "model_available": settings.MODEL_PATH.exists(),
        "similarity_corpus": plagiarism_mod._corpus_loaded,
        "trained_ai_model": ai_detection_mod._model is not None or ai_detection_mod._transformer is not None,
    }
    required = checks["reports_dir_exists"]
    return {"status": "ready" if required else "degraded", "checks": checks}


@app.get("/api/test")
def test_endpoint() -> dict:
    return {"message": "Backend is working!"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled server exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

