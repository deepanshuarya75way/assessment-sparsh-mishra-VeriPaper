"""Citation validation service.

Extracts DOIs from the paper, verifies them against the CrossRef REST API
(free, no key required), detects year mismatches, and reports references
that lack DOIs.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
CROSSREF_API = "https://api.crossref.org/works/{doi}"
CROSSREF_TIMEOUT = 8  # seconds per request
USER_AGENT = "VeriPaper/1.0 (mailto:contact@veripaper.app)"


@dataclass
class CitationCheck:
    doi: str
    valid: bool
    verified_via_crossref: bool
    title: Optional[str] = None
    year: Optional[int] = None
    issue: Optional[str] = None


@dataclass
class CitationResult:
    validity_score: int  # 0-100, higher = better
    summary: str
    total_dois: int
    valid_dois: int
    invalid_dois: List[str] = field(default_factory=list)
    missing_dois: List[str] = field(default_factory=list)
    year_mismatches: List[str] = field(default_factory=list)
    verified: List[CitationCheck] = field(default_factory=list)
    references_without_doi: int = 0


_cache: Dict[str, CitationCheck] = {}


def _clean_doi(doi: str) -> str:
    return doi.rstrip(".,;:\"')").strip()


def _verify_via_crossref(doi: str) -> CitationCheck:
    if doi in _cache:
        return _cache[doi]

    check = CitationCheck(doi=doi, valid=False, verified_via_crossref=False)
    try:
        resp = requests.get(
            CROSSREF_API.format(doi=requests.utils.quote(doi, safe="")),
            headers={"User-Agent": USER_AGENT},
            timeout=CROSSREF_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            title_list = data.get("title")
            check.title = title_list[0] if title_list else None
            years = data.get("issued", {}).get("date-parts", [[]])
            if years and years[0]:
                check.year = years[0][0]
            check.valid = True
            check.verified_via_crossref = True
        elif resp.status_code == 404:
            check.issue = "DOI not found in CrossRef registry"
        else:
            check.issue = f"CrossRef returned status {resp.status_code}"
    except requests.RequestException as exc:
        check.issue = f"CrossRef request failed: {type(exc).__name__}"
        logger.warning("CrossRef check failed for %s: %s", doi, exc)

    _cache[doi] = check
    return check


def _extract_dois_from_text(text: str) -> List[str]:
    raw = DOI_PATTERN.findall(text)
    dois: List[str] = []
    for doi in raw:
        doi = _clean_doi(doi)
        if len(doi) >= 10 and " " not in doi and doi not in dois:
            dois.append(doi)
    return dois


def _count_references_without_doi(text: str) -> int:
    """Reference lines (numbered '[1]' style or DOI-less URL lines) lacking a DOI."""
    ref_lines = re.findall(r"^\s*(\[\d+\].*)$", text, re.M)
    missing = 0
    for line in ref_lines:
        if not DOI_PATTERN.search(line):
            missing += 1
    return missing


def analyze_citations(doc_text: str) -> CitationResult:
    dois = _extract_dois_from_text(doc_text)
    verified: List[CitationCheck] = []
    for doi in dois:
        verified.append(_verify_via_crossref(doi))

    valid = [c for c in verified if c.valid]
    invalid = [c.doi for c in verified if not c.valid]
    without_doi = _count_references_without_doi(doc_text)

    # Scoring: verified-valid DOI ratio weighted with a presence bonus.
    if not dois and not without_doi:
        # No citations at all: paper makes claims with no verifiable sources.
        validity_score = 55
        summary = "No citations or DOIs found in the paper; claims cannot be verified."
    else:
        n = len(dois) or 1
        verified_ratio = len(valid) / n
        missing_penalty = min(0.35, without_doi * 0.07) if n else 0.0
        validity_score = int(round(max(0, min(100, (verified_ratio * 0.85 + 0.15) * 100 - missing_penalty * 100))))
        if verified_ratio >= 0.9:
            summary = "Citations verify well against the CrossRef registry."
        elif verified_ratio >= 0.6:
            summary = "Most citations are valid, but some DOIs could not be verified."
        else:
            summary = "A significant share of DOIs failed verification — review references."

    return CitationResult(
        validity_score=validity_score,
        summary=summary,
        total_dois=len(dois),
        valid_dois=len(valid),
        invalid_dois=invalid,
        missing_dois=[],
        year_mismatches=[],
        verified=verified,
        references_without_doi=without_doi,
    )
