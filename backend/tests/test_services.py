"""Production tests for the VeriPaper analysis services."""
import sys
from pathlib import Path

import pytest

# Portable resolution: works on local sandboxes and GitHub Actions runners.
_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from app.services import parsing
from app.services import ai_detection
from app.services import plagiarism
from app.services import citations
from app.services import statistics
from app.services import writing_quality
from app.services import scoring


# ---------------- Parsing ----------------

# Committed sample papers live under backend/data/ so tests run anywhere
# (GitHub Actions, CI, or a fresh checkout) without local /tmp artifacts.
SAMPLE_PDF = str(_backend / "data" / "sample_paper.pdf")
SAMPLE_TXT = str(_backend / "data" / "sample_paper.txt")
SAMPLE_DOCX = str(_backend / "data" / "sample_paper.docx")


@pytest.mark.parametrize(
    "path",
    [SAMPLE_PDF, SAMPLE_TXT, SAMPLE_DOCX],
    ids=["pdf", "txt", "docx"],
)
def test_parse_document_real_formats(path):
    payload = open(path, "rb").read()
    ext = "pdf" if path.endswith(".pdf") else "txt" if path.endswith(".txt") else "docx"
    doc = parsing.parse_document("paper." + ext, payload)
    assert doc.word_count > 100
    assert doc.extension in (".pdf", ".txt", ".docx")
    labels = {s.label for s in doc.sections}
    assert "abstract" in labels and "references" in labels, [s.label for s in doc.sections]


def test_rejects_non_document_extensions():
    with pytest.raises((ValueError, Exception)):
        parsing.parse_document("malware.exe", b"MZ" + b"\x00" * 1000)


def test_parse_empty_file_rejected():
    with pytest.raises((ValueError, Exception)):
        parsing.parse_document("empty.pdf", b"")


def test_parse_unreadable_pdf_raises():
    with pytest.raises((ValueError, Exception)):
        parsing.parse_document("broken.pdf", b"%PDF-1.4 fake content not a real pdf at all")


# ---------------- AI detection ----------------

def test_ai_detection_real_paper_scored_low():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    full_text = " ".join(s.text for s in doc.sections)
    result = ai_detection.detect_ai(full_text, use_trained=True)
    # Hand-written test paper should score as human (below 60)
    assert result.ai_probability < 60, result
    assert result.engine in ("transformer_finetuned", "trained_model", "heuristic")
    assert result.confidence in ("high", "low")
    assert result.explanation


def test_ai_detection_ai_like_text_scored_high():
    """Text full of AI hallmarks (uniform sentences, hedging chains) should score high."""
    ai_text = ". ".join(["It is important to note that this highlights the significance of the findings"] * 8)
    result = ai_detection.detect_ai(ai_text, use_trained=True)
    assert result.ai_probability >= 70, result


# ---------------- Plagiarism ----------------

def test_plagiarism_clean_paper_no_corpus_matches():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    result = plagiarism.analyze_plagiarism(doc.sections, doc.word_count)
    assert 0 <= result.plagiarism_score <= 100
    # Clean original paper must not report corpus matches against the arXiv corpus
    assert result.matches == [], result.matches


def test_plagiarism_self_duplication_detected():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    if len(doc.sections) > 3:
        doc.sections[3].text = doc.sections[1].text
    result = plagiarism.analyze_plagiarism(doc.sections, doc.word_count)
    assert result.duplicate_paragraphs >= 2, result


# ---------------- Citations ----------------

def test_citations_invalid_dois_flagged():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    full_text = " ".join(s.text for s in doc.sections)
    result = citations.analyze_citations(full_text)
    assert 0 <= result.validity_score <= 100
    assert result.total_dois == result.valid_dois + len(result.invalid_dois)


# ---------------- Statistics ----------------

def test_statistics_suspicious_numbers_flagged():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    full_text = " ".join(s.text for s in doc.sections)
    result = statistics.analyze_statistics(full_text)
    # Sample paper contains rounded-percentage claims (89%, 76%, 95%)
    assert result.risk_score > 0
    assert len(result.findings) > 0


# ---------------- Writing ----------------

def test_writing_checks_sections_and_flags_missing_citations():
    payload = open(SAMPLE_PDF, "rb").read()
    doc = parsing.parse_document("paper.pdf", payload)
    result = writing_quality.analyze_writing_quality(doc)
    assert 0 <= result.score <= 100
    assert len(result.checks) >= 4
    names = {c.name for c in result.checks}
    assert "Citation-text linkage" in names, names
    linkage = next(c for c in result.checks if c.name == "Citation-text linkage")
    assert not linkage.passed, linkage


# ---------------- Scoring ----------------

def _fake_modules(ai_probability: float, plagiarism_score: float, validity_score: float,
                   risk_score: float, writing_score: float):
    """Build service-layer result objects (dataclasses) matching scoring.compute_credibility's API."""
    from app.services.ai_detection import AIDetectionResult
    from app.services.citations import CitationResult
    from app.services.plagiarism import PlagiarismResult
    from app.services.statistics import StatisticalResult
    from app.services.writing_quality import WritingQualityResult
    return (
        AIDetectionResult(ai_probability=ai_probability, confidence="high",
                          explanation="t", engine="heuristic"),
        PlagiarismResult(plagiarism_score=int(plagiarism_score), summary="s", matches=[], duplicate_paragraphs=0),
        CitationResult(validity_score=int(validity_score), summary="s", total_dois=0, valid_dois=0),
        StatisticalResult(risk_score=int(risk_score), summary="s", findings=[], p_values_found=[]),
        WritingQualityResult(score=int(writing_score), grade="Meets standard", checks=[], citation_style="none", section_map={}),
    )


def test_scoring_verdict_categories():
    args = _fake_modules(ai_probability=10, plagiarism_score=5, validity_score=80,
                         risk_score=5, writing_score=80)
    cred = scoring.compute_credibility(*args)
    assert cred.credibility_score >= 70
    assert cred.verdict in ("Credible", "Needs review", "High risk")


def test_scoring_high_ai_yields_review_verdict():
    args = _fake_modules(ai_probability=80, plagiarism_score=5, validity_score=80,
                         risk_score=5, writing_score=80)
    cred = scoring.compute_credibility(*args)
    assert cred.verdict != "Credible", cred
