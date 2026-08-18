"""Shared fixtures: preload the analysis engines so service and API tests use the real models."""
import sys

import pytest

sys.path.insert(0, "/home/ubuntu/VeriPaper/backend")

from app.core.config import settings


@pytest.fixture(scope="session", autouse=True)
def load_engines():
    """Load the trained AI detector (logistic regression + transformer) and the
    plagiarism similarity corpus once for the whole test session."""
    from app.services import ai_detection
    from app.services import plagiarism

    ai_detection.load_trained_model(settings.MODEL_PATH)
    if getattr(ai_detection, "load_transformer_detector", None):
        ai_detection.load_transformer_detector(
            str(settings.ROOT_DIR / "models" / "ai_detector_transformer")
        )
    plagiarism.load_corpus()
    plagiarism.ensure_index()
    ai_detection._load_on_demand()
    yield
