"""Academic writing quality & standards checker.

Evaluates a paper the way a journal reviewer would:
- Section structure compliance (IMRaD / IEEE standard patterns)
- Tone & register (formal academic style)
- Citation style consistency (APA vs IEEE) and citation-text linkage
- Figure/table reference integrity
- Heading hierarchy
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List

from .parsing import ParsedDocument, Section, CITATION_STYLE_IEEE, CITATION_STYLE_APA

logger = logging.getLogger(__name__)

# Standard expected sections in a research paper (IEEE / IMRaD conventions)
EXPECTED_SECTIONS = ["abstract", "introduction", "methods", "results", "conclusion"]

COLLOQUIAL_PATTERNS = [
    re.compile(r"\b(gonna|wanna|gotta|kids|stuff|things like|a lot of)\b", re.I),
    re.compile(r"\b(very nice|pretty good|super important|huge deal)\b", re.I),
]
FIRST_PERSON = re.compile(r"\b(I|we|my|our|ours)\b")
HEDGING = re.compile(r"\b(might|could|possibly|perhaps|seems to|suggests that|may)\b")
EXCLAMATIONS = re.compile(r"!")
ABBREVIATIONS = re.compile(r"\b(etc\.|e\.g\.|i\.e\.)\b")


@dataclass
class CheckResult:
    name: str
    passed: bool
    score: float  # 0-100 contribution
    detail: str
    suggestions: List[str] = field(default_factory=list)


@dataclass
class WritingQualityResult:
    score: int  # 0-100
    grade: str  # "Meets standard" / "Minor issues" / "Needs revision"
    checks: List[CheckResult]
    citation_style: str  # "ieee", "apa", "mixed", "none"
    section_map: dict


def _check_structure(doc: ParsedDocument) -> CheckResult:
    present = {s.label for s in doc.sections}
    missing = [label for label in EXPECTED_SECTIONS if label not in present]
    # Discussion is common but not strictly required; reward its presence.
    bonus = 1.0 if "discussion" in present else 0.85

    if not missing:
        return CheckResult(
            "Section structure",
            True,
            100,
            "All standard sections present (Abstract, Introduction, Methods, Results, Conclusion).",
        )
    penalty = len(missing) * 18
    score = max(0, 100 - penalty) * bonus
    return CheckResult(
        "Section structure",
        False,
        score,
        f"Missing expected sections: {', '.join(missing)}. Standard IMRaD/IEEE papers should include them.",
        suggestions=[
            f"Add a clearly-headed '{m.capitalize()}' section." for m in missing
        ],
    )


def _check_tone(doc: ParsedDocument) -> CheckResult:
    text = doc.full_text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return CheckResult("Academic tone", True, 100, "No analyzable text found.")

    issues: List[str] = []
    exclaim = sum(1 for s in sentences if EXCLAMATIONS.search(s))
    if exclaim:
        issues.append(f"{exclaim} exclamation mark(s) — informal for research writing")
    colloq_hits = sum(len(p.findall(text)) for p in COLLOQUIAL_PATTERNS)
    if colloq_hits:
        issues.append(f"{colloq_hits} colloquial expression(s) detected")
    first_person_ratio = len(FIRST_PERSON.findall(text)) / max(1, len(re.findall(r"\b\w+\b", text)))
    if first_person_ratio > 0.03:
        issues.append("Frequent first-person pronouns; passive/impersonal voice is standard in IEEE-style papers")

    score = max(0, 100 - len(issues) * 22)
    passed = len(issues) == 0
    return CheckResult(
        "Academic tone",
        passed,
        score,
        "Tone is formal and appropriate." if passed else "Informal language patterns detected.",
        suggestions=issues,
    )


def _check_citation_style(doc: ParsedDocument) -> CheckResult:
    body = " ".join(s.text for s in doc.sections if s.label != "references")
    refs = [s.text for s in doc.sections if s.label == "references"]
    ref_text = "\n".join(refs)

    ieee_hits = len(CITATION_STYLE_IEEE.findall(body))
    apa_hits = len(CITATION_STYLE_APA.findall(body))

    style = "none"
    if ieee_hits and apa_hits:
        style = "mixed"
    elif ieee_hits:
        style = "ieee"
    elif apa_hits:
        style = "apa"

    suggestions: List[str] = []
    if style == "none" and refs:
        return CheckResult(
            "Citation style consistency",
            False,
            30,
            "References are listed but no in-text citations were found.",
            suggestions=["Add in-text citations (e.g. [1] for IEEE or (Author, 2024) for APA)."],
        )
    if style == "mixed":
        suggestions.append(
            f"Mixed citation styles detected ({ieee_hits} IEEE-style, {apa_hits} APA-style). Pick one and be consistent."
        )
        score = 55
    elif style == "ieee":
        score = 95
    elif style == "apa":
        score = 95
    else:
        score = 70  # no references section at all
        suggestions.append("Consider adding a References section for verifiable claims.")

    passed = style in ("ieee", "apa")
    return CheckResult(
        "Citation style consistency",
        passed,
        score,
        f"Detected style: {style.upper() if style != 'none' else 'NONE'}." + (
            " Styles are consistent." if passed else " Style usage is inconsistent or missing."
        ),
        suggestions=suggestions,
    )


def _check_citation_linkage(doc: ParsedDocument) -> CheckResult:
    """Every reference should be mentioned in the body, and vice versa."""
    body = " ".join(s.text for s in doc.sections if s.label != "references")
    refs = [s.text for s in doc.sections if s.label == "references"]
    ref_text = "\n".join(refs)

    ref_ids = CITATION_STYLE_IEEE.findall(ref_text)
    body_ids = CITATION_STYLE_IEEE.findall(body)
    orphan_refs = sorted(set(ref_ids) - set(body_ids))
    unused = len(orphan_refs)
    score = 100 if unused == 0 else max(10, 100 - unused * 15)
    passed = unused == 0
    return CheckResult(
        "Citation-text linkage",
        passed,
        score,
        f"All {len(ref_ids)} cited references appear in the body." if passed
        else f"{unused} reference(s) never cited in the body: {', '.join(orphan_refs[:5])}.",
        suggestions=[
            "Cite every listed reference in the body, or remove it from the reference list.",
        ] if not passed else [],
    )


def _check_figure_table_refs(doc: ParsedDocument) -> CheckResult:
    fig_decl = re.findall(r"\b(Figure|Fig\.|Table)\s+\d+", doc.full_text)
    body = " ".join(s.text for s in doc.sections if s.label != "references")
    fig_calls = re.findall(r"\b(?:Figure|Fig\.|Table)\s+\d+", body)
    missing = 0
    suggestions: List[str] = []
    # Heuristic: captions usually appear as "Figure N:" or standalone; if count
    # of declarations exceeds in-text calls substantially, flag it.
    if len(fig_decl) > len(fig_calls) + 1:
        missing = len(fig_decl) - len(fig_calls)
        suggestions.append(
            f"{missing} figure(s)/table(s) declared but not referenced in the text. "
            "Every figure and table should be called out in the body."
        )
    score = 100 if missing == 0 else max(20, 100 - missing * 20)
    return CheckResult(
        "Figure & table integrity",
        missing == 0,
        score,
        f"{len(fig_decl)} figure/table mentions found; all referenced in body." if missing == 0
        else f"{len(fig_decl)} figure/table declarations but only {len(fig_calls)} in-text references.",
        suggestions=suggestions,
    )


def _check_heading_hierarchy(doc: ParsedDocument) -> CheckResult:
    numbered = [s for s in doc.sections if re.match(r"^\d", s.heading.strip())]
    unnumbered = [s for s in doc.sections if s.heading.strip() and not re.match(r"^\d", s.heading.strip())]
    issues: List[str] = []
    if numbered and unnumbered:
        issues.append("Mix of numbered and unnumbered headings — pick one convention (IEEE uses numbered).")
    if len(numbered) >= 2:
        nums = [int(re.match(r"(\d+)", s.heading.strip()).group(1)) for s in numbered]
        if nums != sorted(range(1, len(nums) + 1)):
            issues.append("Section numbers are not sequential.")
    score = 100 if not issues else 100 - len(issues) * 25
    return CheckResult(
        "Heading hierarchy",
        not issues,
        max(0, score),
        "Heading structure is consistent." if not issues else "; ".join(issues),
        suggestions=issues,
    )


WEIGHTS = {
    "Section structure": 0.30,
    "Academic tone": 0.20,
    "Citation style consistency": 0.20,
    "Citation-text linkage": 0.15,
    "Figure & table integrity": 0.075,
    "Heading hierarchy": 0.075,
}


def analyze_writing_quality(doc: ParsedDocument) -> WritingQualityResult:
    checks = [
        _check_structure(doc),
        _check_tone(doc),
        _check_citation_style(doc),
        _check_citation_linkage(doc),
        _check_figure_table_refs(doc),
        _check_heading_hierarchy(doc),
    ]
    raw = sum(WEIGHTS[c.name] * c.score for c in checks)
    score = int(round(raw))
    if score >= 80:
        grade = "Meets standard"
    elif score >= 55:
        grade = "Minor issues"
    else:
        grade = "Needs revision"
    style = "ieee" if any(c.name == "Citation style consistency" and c.passed and "IEEE" in c.detail for c in checks) else "none"
    return WritingQualityResult(
        score=score,
        grade=grade,
        checks=checks,
        citation_style=style,
        section_map={s.label: s.heading for s in doc.sections},
    )
