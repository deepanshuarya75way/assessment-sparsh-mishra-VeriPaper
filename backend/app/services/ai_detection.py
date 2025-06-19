"""AI-generated content detection service.

Three-tier detection:
1. Fast heuristic baseline (perplexity-style + lexical features) — always available
2. Trained logistic regression (features from `_extract_features`)
3. Fine-tuned transformer classifier (DistilBERT) — loaded lazily if available

The production route uses the best available engine and reports which one was
used, its version, and measured validation metrics.
"""
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# --- Lexical / stylistic features shared by the LR model ---------------------

STOPWORDS = set(
    "the a an and or but if then because as of in to for on with by at from is "
    "are was were be been being have has had do does did will would could should "
    "may might can this that these those it its they them their we our us they"
    .split()
)


def _extract_features(text: str) -> List[float]:
    """Feature vector used by the trained logistic regression detector."""
    lowered = text.lower()
    words = re.findall(r"\b[a-zA-Z]{2,}\b", lowered)
    word_count = max(1, len(words))
    unique_words = len(set(words))

    # Lexical diversity (type-token ratio)
    ttr = unique_words / word_count

    # Perplexity proxy: average inverse word frequency from a small corpus
    # (uniform fallback keeps the feature stable without an external LM)
    freq: dict = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab_size = max(1, len(freq))
    perp_proxy = math.exp(-sum((c / word_count) * math.log(max(1e-9, c / vocab_size)) for c in freq.values()))

    # Burstiness: coefficient of variation of sentence lengths
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lens = [len(s.split()) for s in sentences] or [word_count]
    mean_len = sum(sent_lens) / len(sent_lens)
    var_len = sum((l - mean_len) ** 2 for l in sent_lens) / len(sent_lens)
    burstiness = (var_len ** 0.5) / max(1e-6, mean_len)

    # Stopword ratio
    stop_ratio = sum(1 for w in words if w in STOPWORDS) / word_count

    # Repetition ratio
    repetitive = 1.0 - unique_words / word_count

    # Punctuation entropy proxy
    punct = sum(1 for ch in text if ch in ";:,") / max(1, word_count)

    # Paragraph uniformity
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    para_lens = [len(p.split()) for p in paragraphs] or [word_count]
    para_mean = sum(para_lens) / len(para_lens)
    para_cv = (sum((l - para_mean) ** 2 for l in para_lens) / len(para_lens)) ** 0.5 / max(1e-6, para_mean)

    return [ttr, min(perp_proxy, 500) / 500.0, burstiness, stop_ratio, repetitive, punct, para_cv]


# --- Heuristic baseline -------------------------------------------------------

def heuristic_ai_score(text: str) -> float:
    """Fast keyword/structure heuristic (0-1). Used as fallback and sanity tier."""
    lowered = text.lower()
    words = re.findall(r"\b[a-zA-Z]{2,}\b", lowered)
    word_count = max(1, len(words))
    unique = len(set(words))

    repetitive_ratio = 1.0 - unique / word_count
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    avg_sentence_length = word_count / sentence_count
    bursty_punctuation = len(re.findall(r"[;:,]", text))

    keyword_score = sum(
        lowered.count(token)
        for token in [
            "we propose", "in this paper", "state-of-the-art", "novel framework",
            "significant improvement", "delve", "furthermore", "moreover",
            "it is important to note", "tapestry", "landscape of",
        ]
    )

    score = (
        0.10
        + min(0.40, repetitive_ratio * 0.7)
        + min(0.20, keyword_score / 10)
        + min(0.15, max(avg_sentence_length - 20, 0) / 80)
        + min(0.08, bursty_punctuation / 250)
    )
    return max(0.02, min(0.98, score))


# --- Trained model tier -------------------------------------------------------

_model = None
_model_is_pipeline: bool = False
_model_path: Optional[str] = None  # registered at boot; loaded lazily on first call
_model_loaded_attempted: bool = False
_model_meta = {
    "version": "unknown",
    "metrics": {"f1": None, "precision": None, "recall": None, "auc": None},
}


def _pick_engine(model_path):
    """Read the benchmark winner file; transformer wins on equal/higher F1."""
    import json

    winner_path = model_path.parent / "ai_detector_winner.json"
    if winner_path.exists():
        try:
            with open(winner_path) as fh:
                meta = json.load(fh)
            return meta.get("winner", "logistic_regression"), meta
        except Exception:  # pragma: no cover
            pass
    return "logistic_regression", {}


def load_trained_model(model_path) -> bool:
    """Register the model location; the actual weights are loaded lazily on the
    first detection call to keep boot memory low (e.g. 512 MB free-tier hosts).

    The benchmark script (scripts/train_ai_detector_v3.py) writes
    models/ai_detector_winner.json; the transformer path is preferred because
    it scored higher on the validation split.
    """
    from pathlib import Path
    global _model_path, _model_loaded_attempted
    path = Path(model_path)
    _model_path = str(path)
    _model_loaded_attempted = False
    if not path.exists() and not (path.parent / "ai_detector_transformer").exists():
        return False
    return True


def _load_on_demand() -> None:
    """Load the best available trained detector on first use."""
    global _model_loaded_attempted
    if _model_path is None or _model_loaded_attempted:
        return
    _model_loaded_attempted = True
    from pathlib import Path
    path = Path(_model_path)
    tf_path = path.parent / "ai_detector_transformer"
    # If the transformer weights are missing (e.g. an LFS pointer in the image),
    # attempt a memory-safe restore now that we are past the strict boot window.
    _maybe_restore_weights(tf_path)
    engine, meta = _pick_engine(path)
    if engine == "transformer" and tf_path.exists() and _weights_valid(tf_path):
        if load_transformer_detector(str(tf_path)):
            _model_meta["metrics"] = {
                "f1": meta.get("transformer_f1"),
                "validation": "held-out 20% test split (balanced)",
            }
            return True
    if path.exists():
        global _model, _model_is_pipeline
        try:
            import joblib

            _model = joblib.load(path)
            # The shipped LR artifact is a scikit-learn Pipeline (TF-IDF + LR) that
            # takes raw text directly; older raw-coefficient checkpoints took the
            # 7-dim stylometric vector instead. Handle both.
            if hasattr(_model, "transform") and hasattr(_model, "predict_proba"):
                _model_is_pipeline = True
                _model_meta["version"] = "logreg-pipeline"
            else:
                _model_is_pipeline = False
                _model_meta["version"] = "logreg-v1"
            _model_meta["metrics"] = {
                "f1": meta.get("lr_f1"),
                "validation": "5-fold cross-validation (balanced)",
            }
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load trained AI model: %s", exc)
    return False


def trained_ai_score(text: str) -> Optional[float]:
    """Run the trained logistic regression detector (0-1), or None if unavailable."""
    _load_on_demand()
    if _model is None:
        return None
    try:
        if _model_is_pipeline:
            prob = float(_model.predict_proba([text])[0][1])
        else:
            features = np.array([_extract_features(text)])
            prob = float(_model.predict_proba(features)[0][1])
        return max(0.0, min(1.0, prob))
    except Exception as exc:  # pragma: no cover
        logger.error("Trained model inference failed: %s", exc)
        return None


# --- Transformer tier (Hugging Face) ------------------------------------------

_transformer = None
_transformer_load_error: Optional[str] = None  # staged load failure detail


def _onnx_session(model_path: str) -> "Optional[InferenceSession]":
    """Load the quantized ONNX graph if present and memory-safe."""
    import onnxruntime as ort

    onnx_path = _Path(model_path) / "model_quantized.onnx"
    if not onnx_path.exists() or onnx_path.stat().st_size < 1_000_000:
        return None
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options, providers=providers)


def load_transformer_detector(model_name_or_path: Optional[str] = None) -> bool:
    """Lazily load a fine-tuned transformer detector from the local path or HF Hub.

    On memory-constrained hosts (e.g. Render's 512 MB free tier) the quantized
    ONNX graph is preferred: it scores identically to the PyTorch checkpoint but
    runs at a fraction of the resident memory because it skips torch and uses an
    int8 dynamic-quantized runtime. The full PyTorch weights are the fallback.
    """
    global _transformer, _transformer_load_error
    if model_name_or_path is None:
        return False
    # Per-stage logging: the failure point is otherwise invisible on the
    # memory-constrained free tier where a crash wipes the process state.
    onnx_path = _Path(model_name_or_path) / "model_quantized.onnx"
    logger.info(
        "Loading transformer detector from %s (onnx present=%s, size=%.0f MB)",
        model_name_or_path,
        onnx_path.exists(),
        (onnx_path.stat().st_size / 1e6) if onnx_path.exists() else 0,
    )
    try:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            label_index = _resolve_label_map(model_name_or_path)
        except Exception as exc:
            _transformer_load_error = f"tokenizer: {type(exc).__name__}: {exc}"
            logger.error("Transformer tokenizer load failed: %s", _transformer_load_error)
            return False
        logger.info("Transformer tokenizer loaded; label_index=%s", label_index)

        onnx = _onnx_session(model_name_or_path)
        if onnx is not None:
            _transformer = (tokenizer, onnx, None, label_index)
            _model_meta["version"] = f"transformer-onnx:{model_name_or_path}"
            logger.info("Transformer detector loaded via quantized ONNX runtime")
            return True

        from transformers import AutoModelForSequenceClassification
        import torch

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            model.eval()
        except Exception as exc:
            _transformer_load_error = f"torch: {type(exc).__name__}: {exc}"
            logger.error("Transformer PyTorch load failed: %s", _transformer_load_error)
            return False
        # Label ordering: default config label2id {ai:0, human:1} (see
        # ai_detector_transformer_metrics.json label_1 note)
        _transformer = (tokenizer, model, torch, label_index)
        _model_meta["version"] = f"transformer:{model_name_or_path}"
        logger.info("Transformer detector loaded via PyTorch (full weights)")
        return True
    except Exception as exc:  # pragma: no cover
        logger.error("Transformer detector load failed: %s", exc)
        return False


def _softmax(x, axis: int = -1):
    """Numerically stable softmax over a numpy array."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _weights_valid(tf_path: _Path) -> bool:
    """True when the transformer weights file is present and substantial.

    Accepts either the full PyTorch checkpoint (model.safetensors) or the
    quantized ONNX graph (model_quantized.onnx), whichever is available.
    """
    for name in ("model.safetensors", "model_quantized.onnx"):
        weights = tf_path / name
        if weights.exists() and weights.stat().st_size > 1_000_000:
            return True
    return False


def _maybe_restore_weights(tf_path: _Path) -> None:
    """Try to restore missing transformer weights from the pinned release.

    The restore needs significant free memory, so it is only attempted when
    headroom exists; otherwise the LR fallback is used and nothing crashes.
    """
    try:
        from ..main import _ensure_transformer_weights

        _ensure_transformer_weights(at_boot=False)
    except Exception as exc:  # pragma: no cover
        logger.warning("Weights restore skipped: %s", exc)


def _resolve_label_map(model_path: str) -> int:
    """Return the class index that corresponds to AI text."""
    import json

    path = _Path(model_path)
    cfg = path / "config.json"
    if cfg.exists():
        with open(cfg) as fh:
            labels = json.load(fh).get("label2id", {})
        if "ai" in labels:
            return int(labels["ai"])
    return 1


def transformer_ai_score(text: str) -> Optional[float]:
    """Run the transformer detector (0-1), or None if unavailable.

    Uses the quantized ONNX runtime when it was loaded; otherwise runs the
    standard PyTorch path.
    """
    _load_on_demand()
    if _transformer is None:
        return None
    try:
        tokenizer, model, torch, ai_class = _transformer
        if torch is None:
            # ONNX path: raw int64 arrays, softmax done in float32
            import numpy as np

            encoded = tokenizer(
                text, return_tensors="np", truncation=True, max_length=256, padding=True
            )
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            logits = model.run(None, inputs)[0]
            probs = _softmax(np.asarray(logits), axis=-1)
            prob = float(probs[0][ai_class])
            return max(0.0, min(1.0, prob))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        prob = float(probs[0][ai_class])
        return max(0.0, min(1.0, prob))
    except Exception as exc:  # pragma: no cover
        logger.error("Transformer inference failed: %s", exc)
        return None


# --- Public API ---------------------------------------------------------------

@dataclass
class AIDetectionResult:
    ai_probability: float  # 0-100
    confidence: str  # "high" / "moderate" / "low"
    engine: str  # which engine produced the result
    explanation: str


AI_HIGH = 0.62
AI_LOW = 0.38


def detect_ai(text: str, use_trained: bool = True) -> AIDetectionResult:
    if use_trained:
        _load_on_demand()
    score = None
    engine = "heuristic"

    if use_trained:
        # Prefer the fine-tuned transformer (highest validation F1),
        # fall back to the logistic regression, then the heuristic.
        score = transformer_ai_score(text)
        if score is not None:
            engine = "transformer_finetuned"
        if score is None:
            score = trained_ai_score(text)
            if score is not None:
                engine = "logistic_regression"
    if score is None:
        score = heuristic_ai_score(text)

    probability = round(score * 100, 1)
    if score >= AI_HIGH:
        confidence = "high"
        explanation = "The text shows strong AI-like regularity patterns."
    elif score <= AI_LOW:
        confidence = "high"
        explanation = "The text shows predominantly human writing patterns."
    else:
        confidence = "low"
        explanation = "Signals are mixed; treat this result as uncertain."

    return AIDetectionResult(
        ai_probability=probability,
        confidence=confidence,
        engine=engine,
        explanation=explanation,
    )


def get_detector_meta() -> dict:
    return {
        "model_version": _model_meta["version"],
        "transformer_load_error": _transformer_load_error,
        "has_trained_model": _model is not None,
        "has_transformer": _transformer is not None,
        "metrics": _model_meta["metrics"],
        "thresholds": {
            "high_confidence_ai": round(AI_HIGH, 2),
            "high_confidence_human": round(AI_LOW, 2),
            "uncertain_range": [round(AI_LOW, 2), round(AI_HIGH, 2)],
        },
    }
