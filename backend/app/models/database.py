"""Database models for storing analysis results and history.
Works with PostgreSQL and SQLite via SQLAlchemy ORM.
"""
import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnalysisResult(Base):
    """Store complete paper analysis results with metadata."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)

    # File metadata
    filename = Column(String(255), nullable=False, index=True)
    author_name = Column(String(255), nullable=True)
    institution = Column(String(255), nullable=True)
    full_text = Column(JSON, nullable=True)
    file_size = Column(Integer, nullable=False)  # bytes
    file_hash = Column(String(64), unique=True, index=True)  # SHA256
    word_count = Column(Integer, nullable=False, default=0)

    # Module scores
    plagiarism_score = Column(Integer, nullable=False)  # 0-100
    ai_probability = Column(Float, nullable=False)  # 0-100
    ai_confidence = Column(String(50), nullable=False)
    ai_engine = Column(String(100), nullable=True)
    citation_validity_score = Column(Integer, nullable=False)  # 0-100
    statistical_risk_score = Column(Integer, nullable=False)  # 0-100
    writing_quality_score = Column(Integer, nullable=False)  # 0-100

    # Overall credibility
    overall_research_credibility = Column(Integer, nullable=False)  # 0-100
    verdict = Column(String(50), nullable=False)

    # Detailed evidence as JSON blobs
    plagiarism_matches = Column(JSON, nullable=True)
    citation_details = Column(JSON, nullable=True)  # {invalid_dois, references_without_doi}
    statistical_findings = Column(JSON, nullable=True)
    writing_checks = Column(JSON, nullable=True)
    sections = Column(JSON, nullable=True)

    # Provenance Chain Verification (PCV) layers — nullable for back-compat
    # with older records created before PCV shipped.
    retraction_contamination_score = Column(Integer, nullable=True)
    citation_graph_anomaly_score = Column(Integer, nullable=True)
    claim_alignment_score = Column(Integer, nullable=True)
    methodology_fingerprint_score = Column(Integer, nullable=True)
    provenance_score = Column(Integer, nullable=True)
    pcv_details = Column(JSON, nullable=True)

    # Report generation
    report_path = Column(String(255), nullable=True)
    report_generated = Column(Boolean, default=False)

    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "author_name": self.author_name,
            "institution": self.institution,
            "full_text": self.full_text,
            "file_size": self.file_size,
            "word_count": self.word_count,
            "plagiarism_score": self.plagiarism_score,
            "ai_probability": self.ai_probability,
            "ai_confidence": self.ai_confidence,
            "ai_engine": self.ai_engine,
            "citation_validity_score": self.citation_validity_score,
            "statistical_risk_score": self.statistical_risk_score,
            "writing_quality_score": self.writing_quality_score,
            "overall_research_credibility": self.overall_research_credibility,
            "verdict": self.verdict,
            "report_path": self.report_path,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "provenance_score": self.provenance_score,
            "retraction_contamination_score": self.retraction_contamination_score,
            "citation_graph_anomaly_score": self.citation_graph_anomaly_score,
            "claim_alignment_score": self.claim_alignment_score,
            "methodology_fingerprint_score": self.methodology_fingerprint_score,
            "provenance_details": json.loads(self.pcv_details) if self.pcv_details else None,
        }
