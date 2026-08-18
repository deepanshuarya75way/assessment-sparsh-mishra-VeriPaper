"""Semantic plagiarism detection service.

Uses SBERT embeddings + FAISS similarity search against a bundled corpus of
open academic abstracts, plus an internal duplicate-paragraph detector for
self-plagiarism. Returns per-section similarity evidence.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_PAPER_WORDS = 60          # below this, plagiarism signals are unreliable
CHUNK_MIN_WORDS = 35          # sentence-group chunk size for embedding search
FAISS_SEARCH_TOP_K = 5
FAISS_SIMILARITY_THRESHOLD = 0.75  # cosine similarity above this = a candidate match
FAISS_MATCH_THRESHOLD = 0.82      # only reported as a confident match above this


@dataclass
class SimilarityMatch:
    source_title: str
    source_corpus: str
    similarity: float
    matched_text: str


@dataclass
class PlagiarismResult:
    plagiarism_score: int  # 0-100, higher = more similarity found
    summary: str
    matches: List[SimilarityMatch] = field(default_factory=list)
    duplicate_paragraphs: int = 0
    embeddable_sections: int = 0


_index = None          # faiss IndexFlatIP
_embeddings = None     # np.ndarray
_titles = None         # np.ndarray of titles
_corpus_labels = None  # np.ndarray of corpus labels
_embedder = None       # sentence-transformers model
_corpus_path: Optional[str] = None  # resolves corpus lazily to save boot memory
_corpus_loaded: bool = False
_corpus_failed: bool = False

_corpuses: Dict[str, str] = {
    "arxiv": "arXiv open abstract corpus",
    "self": "internal (within-paper) duplication",
}


def _load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the SBERT embedder (~90 MB)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(model_name)
            logger.info("SBERT embedder loaded")
        except Exception as exc:  # pragma: no cover
            logger.error("SBERT load failed: %s", exc)
            raise


def _embed_batch(texts: List[str]) -> np.ndarray:
    _load_embedder()
    vectors = _embedder.encode(texts, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def load_corpus(corpus_path: Optional[str] = None) -> bool:
    """Register (or reload) the similarity corpus path.

    On memory-constrained hosts (e.g. free-tier Render with 512 MB), the
    embedding model and FAISS index are built lazily on the first analysis
    request instead of at boot.

    JSONL format: {"title": "...", "text": "...", "corpus": "arxiv"}
    """
    global _index, _embeddings, _titles, _corpus_labels, _corpus_path
    global _corpus_loaded, _corpus_failed
    _index = None
    _embeddings = None
    _titles = None
    _corpus_labels = None
    _corpus_loaded = False
    _corpus_failed = False
    if corpus_path is None:
        corpus_path = os.environ.get(
            "VERIPAPER_CORPUS_PATH",
            str(Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"),
        )
    path = Path(corpus_path)
    if not path.exists():
        logger.warning("Corpus file not found: %s", path)
        return False
    _corpus_path = str(path)
    _corpus_loaded = True
    return True


def ensure_index() -> bool:
    """Lazily build the FAISS index on first use (saves boot memory)."""
    global _corpus_loaded, _corpus_failed
    if _index is not None:
        return True
    if _corpus_failed or not _corpus_loaded or not _corpus_path:
        return False
    _corpus_failed = True  # avoid re-raise loops; reset if a fresh load_corpus is called
    return _build_index(_corpus_path)


def _build_index_from_texts(titles, labels, texts, path: Path) -> bool:
    """Build the embedding matrix and FAISS index from parsed corpus texts."""
    global _index, _embeddings, _corpus_path, _corpus_failed
    logger.info("Building FAISS index over %d corpus entries...", len(texts))
    try:
        _embeddings = _embed_batch(texts)
    except Exception as exc:
        logger.error("Corpus embedding failed: %s", exc)
        _corpus_failed = True
        return False
    dimension = _embeddings.shape[1]
    import faiss

    _index = faiss.IndexFlatIP(dimension)
    _index.add(_embeddings)
    logger.info("FAISS index ready (%s)", _index.ntotal)
    return True


def _build_index(path_str: str) -> bool:
    """Parse the registered corpus JSONL and build the index lazily."""
    global _corpus_failed, _corpus_loaded
    path = Path(path_str)
    if not path.exists():
        logger.warning("Corpus file not found: %s", path)
        _corpus_failed = True
        _corpus_loaded = False
        return False

    titles, texts, labels = [], [], []
    import json

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid corpus line %d", lineno)
                continue
            title = item.get("title") or f"Entry {lineno}"
            text = (item.get("text") or "").strip()
            if len(text.split()) < 15:
                continue
            titles.append(title)
            texts.append(text)
            labels.append(item.get("corpus") or "arxiv")

    if not texts:
        logger.warning("Corpus empty after parsing")
        _corpus_failed = True
        _corpus_loaded = False
        return False
    ok = _build_index_from_texts(titles, labels, texts, path)
    _corpus_failed = not ok
    return ok


def _chunk_section_text(text: str) -> List[str]:
    """Split section text into overlapping sentence-group chunks."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 8]
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for sent in sentences:
        cur.append(sent)
        cur_words += len(sent.split())
        if cur_words >= CHUNK_MIN_WORDS:
            chunks.append(". ".join(cur) + ".")
            cur = cur[-2:]  # overlap: keep last two sentences
            cur_words = sum(len(c.split()) for c in cur)
    if cur:
        chunks.append(". ".join(cur) + ".")
    return chunks


def _search_chunk(chunk: str) -> Optional[SimilarityMatch]:
    ensure_index()
    if _index is None:
        return None
    query = _embed_batch([chunk])
    scores, indices = _index.search(query, FAISS_SEARCH_TOP_K)
    best_sim = float(scores[0][0])
    best_idx = int(indices[0][0])
    if best_sim >= FAISS_MATCH_THRESHOLD and best_idx >= 0:
        return SimilarityMatch(
            source_title=str(_titles[best_idx]),
            source_corpus=_corpuses.get(str(_corpus_labels[best_idx]), str(_corpus_labels[best_idx])),
            similarity=round(float(best_sim) * 100, 1),
            matched_text=chunk[:180],
        )
    return None


def _jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    return inter / float(len(a_tokens | b_tokens))


def _detect_internal_duplication(sections: List["Section"]) -> List[SimilarityMatch]:
    """Find near-duplicate content across sections (self-plagiarism / copy-paste).

    Uses near-exact token overlap (Jaccard >= 0.70) rather than embedding
    similarity: academic papers are topically consistent, so embedding-based
    checks between the paper's own sections over-flag normal writing.
    """
    from .parsing import Section  # noqa: delayed import to avoid cycles
    matches: List[SimilarityMatch] = []
    token_sets: List[tuple] = []  # (text, tokens set, section index)
    for si, section in enumerate(sections):
        for chunk in _chunk_section_text(section.text):
            token_sets.append((chunk, set(chunk.lower().split()), si))
    reported: set = set()
    for i in range(len(token_sets)):
        text_i, toks_i, si = token_sets[i]
        for j in range(i + 1, len(token_sets)):
            toks_j = token_sets[j][1]
            if len(toks_j) < 25 or len(toks_i) < 25:
                continue
            if len(toks_j) < 0.7 * len(toks_i) or len(toks_i) < 0.7 * len(toks_j):
                continue
            jacc = _jaccard(toks_i, toks_j)
            if jacc >= 0.85:
                key = hash(text_i + text_i if False else text_i[:150])
                if key in reported:
                    continue
                reported.add(key)
                matches.append(
                    SimilarityMatch(
                        source_title="Other section of this paper",
                        source_corpus="self",
                        similarity=round(jacc * 100, 1),
                        matched_text=text_i[:180],
                    )
                )
    return matches


def analyze_plagiarism(sections: List["Section"], word_count: int) -> PlagiarismResult:
    """Full plagiarism analysis: corpus search + internal duplication."""
    results: List[SimilarityMatch] = []
    embeddable = 0

    for section in sections:
        chunks = _chunk_section_text(section.text)
        embeddable += 1
        for chunk in chunks:
            match = _search_chunk(chunk)
            if match:
                results.append(match)

    # Internal duplication: near-exact repetition across the paper's sections.
    dup_matches: List[SimilarityMatch] = []
    try:
        dup_matches = _detect_internal_duplication(sections)
    except Exception as exc:  # pragma: no cover
        logger.error("Internal duplication check failed: %s", exc)

    # Deduplicate corpus matches by source title
    seen: set = set()
    unique_matches: List[SimilarityMatch] = []
    for match in results:
        key = match.source_title
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    unique_matches.sort(key=lambda m: m.similarity, reverse=True)
    top_matches = unique_matches[:6]

    score = 0
    if top_matches:
        score = int(round(min(100, top_matches[0].similarity * 0.6 + len(top_matches) * 6)))
    elif dup_matches:
        score = min(100, len(dup_matches) * 12 + 20)

    score = max(0, min(100, score))
    if score <= 5:
        summary = "No significant similarity to the indexed open corpus."
    elif score <= 25:
        summary = "Low similarity detected; some passages resemble indexed abstracts."
    elif score <= 50:
        summary = "Moderate similarity — several passages match indexed sources."
    else:
        summary = "High similarity detected — substantial overlap with indexed sources."

    return PlagiarismResult(
        plagiarism_score=score,
        summary=summary,
        matches=top_matches,
        duplicate_paragraphs=len(dup_matches),
        embeddable_sections=embeddable,
    )
