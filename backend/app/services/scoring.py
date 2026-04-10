"""Unified research credibility scoring.

Combines the five module results (AI detection, plagiarism, citations,
statistics, writing quality) into a single explainable credibility score
with per-metric weights, risk triggers, and action items.
"""
from dataclasses import dataclass
from typing import List

from .ai_detection import AIDetectionResult
from .citations import CitationResult
from .plagiarism import PlagiarismResult
from .statistics import StatisticalResult
from .writing_quality import WritingQualityResult

# Weights sum to 1.0
WEIGHTS = {
    "ai": 0.25,
    "plagiarism": 0.20,
    "citations": 0.25,
    "statistics": 0.10,
    "writing": 0.20,
}


@dataclass
class ScoreBreakdown:
    credibility_score: int  # 0-100
    verdict: str  # "Credible" / "Needs review" / "High risk"
    verdict_detail: str
    contributions: dict
    risk_triggers: List[str]
    action_items: List[str]
    flags: List[str]  # uncertainty/ESL caveats


def _verdict(score: int) -> tuple:
    if score >= 75:
        return "Credible", "This paper appears structurally reliable and is suitable for standard peer review."
    if score >= 50:
        return "Needs review", "This paper shows promise but requires focused manual validation before acceptance."
    return "High risk", "This paper exhibits elevated integrity risks and requires prioritized investigation."


def compute_credibility(
    ai: AIDetectionResult,
    plagiarism: PlagiarismResult,
    citations: CitationResult,
    statistics: StatisticalResult,
    writing: WritingQualityResult,
) -> ScoreBreakdown:
    # Normalize each module to a 0-100 "health" value (higher = healthier).
    ai_health = 100 - ai.ai_probability
    plagiarism_health = 100 - plagiarism.plagiarism_score
    citation_health = citations.validity_score
    stats_health = 100 - statistics.risk_score
    writing_health = writing.score

    contributions = {
        "ai_detection": round(ai_health * WEIGHTS["ai"], 1),
        "plagiarism": round(plagiarism_health * WEIGHTS["plagiarism"], 1),
        "citation_validity": round(citation_health * WEIGHTS["citations"], 1),
        "statistical_integrity": round(stats_health * WEIGHTS["statistics"], 1),
        "writing_standards": round(writing_health * WEIGHTS["writing"], 1),
    }
    raw = sum(contributions.values())
    score = int(round(raw))

    verdict, verdict_detail = _verdict(score)

    risk_triggers: List[str] = []
    if ai.ai_probability >= 60:
        risk_triggers.append(f"AI-generation likelihood is high ({ai.ai_probability}%).")
    if plagiarism.plagiarism_score >= 25:
        risk_triggers.append(f"Similarity score is elevated ({plagiarism.plagiarism_score}%).")
    if citations.validity_score < 60:
        risk_triggers.append(f"Citation verification is weak ({citations.validity_score}%).")
    if statistics.risk_score >= 20:
        risk_triggers.append(f"Statistical anomalies detected (risk {statistics.risk_score}%).")
    if writing.score < 55:
        risk_triggers.append("Academic writing standards are not met.")

    action_items: List[str] = []
    if ai.ai_probability >= 60:
        action_items.append("Request draft history or writing-process notes to verify authorship.")
    if plagiarism.plagiarism_score >= 20:
        action_items.append("Run an external plagiarism check on the highlighted overlapping sections.")
    if citations.invalid_dois:
        action_items.append("Manually review references with invalid or unverified DOIs.")
    if citations.references_without_doi > 2:
        action_items.append("Several references lack DOIs; verify them in the original sources.")
    if statistics.risk_score >= 20:
        action_items.append("Audit reported p-values and independently verify key statistical claims.")
    if not action_items:
        action_items.append("No critical risk triggers detected; proceed with standard peer review.")

    flags: List[str] = []
    if ai.confidence == "low":
        flags.append("AI-detection confidence is low — treat the AI score as uncertain.")
    if plagiarism.matches:
        flags.append("Corpus matches come from open abstracts only; a full-internet check was not performed.")
    flags.append(
        "Formal or non-native writing can raise AI-likelihood scores; interpret the AI score "
        "alongside authorship evidence."
    )
    flags.append("Results are assistive signals, not conclusive proof of misconduct.")

    return ScoreBreakdown(
        credibility_score=score,
        verdict=verdict,
        verdict_detail=verdict_detail,
        contributions=contributions,
        risk_triggers=risk_triggers,
        action_items=action_items,
        flags=flags,
    )
