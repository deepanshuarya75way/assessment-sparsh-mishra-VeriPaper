from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PlagiarismMatchSchema(BaseModel):
    title: str
    similarity: float = Field(..., ge=0, le=100)
    source: str
    matched_text: str
    match_type: str = "uncited"  # 'cited', 'uncited', or 'internal'



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


class ProvenanceModuleResult(BaseModel):
    """Provenance Chain Verification (PCV) layer results."""
    score: int = Field(..., ge=0, le=100)
    summary: str
    fingerprint_score: int = Field(..., ge=0, le=100)
    fingerprint_summary: str
    fingerprint_notes: Dict[str, str]  # benford / p_curve / precision / consistency
    alignment_score: int = Field(..., ge=0, le=100)
    alignment_summary: str
    alignment_verdicts: List[Dict[str, object]]  # {doi, alignment, overlap, reason}
    contamination_score: int = Field(..., ge=0, le=100)
    contamination_summary: str
    retracted_dois: List[Dict[str, object]]  # {doi, retraction_date, reason, notice_type}
    graph_anomaly_score: int = Field(..., ge=0, le=100)
    graph_anomaly_summary: str
    graph_density: float = 0.0
    isolated_references: int = 0
    web_matches: List[Dict[str, object]] = []  # {url, title, snippet, matched_text, similarity}
    web_summary: str = ""
    web_available: bool = False
    web_provider: str = ""


class ScoreContributions(BaseModel):
    ai_detection: float
    plagiarism: float
    citation_validity: float
    statistical_integrity: float
    writing_standards: float


class AnalysisResult(BaseModel):
    filename: str
    author_name: Optional[str] = None
    institution: Optional[str] = None
    full_text: Optional[str] = None
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
    # Provenance Chain Verification
    provenance: Optional[ProvenanceModuleResult] = None
    # Evidence export
    report_path: str
    download_url: Optional[str] = 
    
class ReportChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class ReportChatResponse(BaseModel):
    report_id: int
    answer: str
    evidence: list[dict]

