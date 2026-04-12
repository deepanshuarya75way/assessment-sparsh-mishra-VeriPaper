from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PlagiarismMatchSchema(BaseModel):
    title: str
    similarity: float = Field(..., ge=0, le=100)
    source: str
    matched_text: str


class PlagiarismModuleResult(BaseModel):
    score: float = Field(..., ge=0, le=100)
    summary: str
    matches: List[PlagiarismMatchSchema]
    duplicate_paragraphs: int = 0


class AIDetectionModuleResult(BaseModel):
    ai_probability: float = Field(..., ge=0, le=100)
    confidence: str  # high / low (uncertain)
    explanation: str
    engine: str  # detection engine used
    model_version: str


class CitationModuleResult(BaseModel):
    validity_score: float = Field(..., ge=0, le=100)
    summary: str
    total_dois: int
    valid_dois: int
    invalid_dois: List[str]
    references_without_doi: int = 0
    verified_dois: List[Dict[str, object]]  # {doi, title, year}


class StatisticalModuleResult(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    summary: str
    findings: List[Dict[str, str]]  # {category, severity, detail}
    p_values_found: List[float]


class WritingCheck(BaseModel):
    name: str
    passed: bool
    score: float = Field(..., ge=0, le=100)
    detail: str
    suggestions: List[str]


class WritingModuleResult(BaseModel):
    score: int = Field(..., ge=0, le=100)
    grade: str  # Meets standard / Minor issues / Needs revision
    checks: List[WritingCheck]
    section_map: Dict[str, str]


class SectionEvidence(BaseModel):
    label: str
    heading: str
    word_count: int


class ScoreContributions(BaseModel):
    ai_detection: float
    plagiarism: float
    citation_validity: float
    statistical_integrity: float
    writing_standards: float


class AnalysisResult(BaseModel):
    filename: str
    analyzed_at: str
    word_count: int
    section_count: int
    sections: List[SectionEvidence]
    # Module results
    plagiarism: PlagiarismModuleResult
    ai_detection: AIDetectionModuleResult
    citation: CitationModuleResult
    statistics: StatisticalModuleResult
    writing: WritingModuleResult
    # Aggregate
    overall_research_credibility: float = Field(..., ge=0, le=100)
    verdict: str  # Credible / Needs review / High risk
    verdict_detail: str
    contributions: ScoreContributions
    risk_triggers: List[str]
    action_items: List[str]
    flags: List[str]
    # Evidence export
    report_path: str
    download_url: Optional[str] = None
