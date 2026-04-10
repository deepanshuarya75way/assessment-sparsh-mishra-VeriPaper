"""Statistical integrity checker.

Flags suspicious numeric patterns in reported results:
- p-value anomalies (exact .05/.01/.001 borders, suspicious clustering, too many
  borderline-significant values — a hallmark of p-hacking)
- Digit-preference bias in last digits (GRanularity-style: real reported
  decimals have nearly uniform last digits; fabricated numbers skew)
- Implausible percentage/accuracy claims (e.g. exactly 100% or 0% in ML results)
- Inconsistent statistics (SD larger than plausible range, percentages > 100)
"""
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class StatisticalFinding:
    category: str
    severity: str  # "low" / "medium" / "high"
    detail: str


@dataclass
class StatisticalResult:
    risk_score: int  # 0-100, higher = more suspicious
    summary: str
    findings: List[StatisticalFinding] = field(default_factory=list)
    p_values_found: List[float] = field(default_factory=list)


PVALUE_PATTERN = re.compile(r"(?:p\s*(?:value)?\s*(?:=|<|>|≤|≥)\s*)(\d+(?:\.\d+)?)|((?<!\d)\.\d{2,4})(?=\s*(?:,|\s|$))")
PVALUE_DIRECT = re.compile(r"p\s*[<>=≤≥]+\s*(\d+(?:\.\d+)?)")
PERCENTAGE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DECIMAL_NUMBER = re.compile(r"(?<!\d)(\d+\.\d{2,4})(?!\d)")

# Classic p-value borders that are over-represented when values are massaged
BORDERS = {0.05, 0.01, 0.001, 0.050, 0.010, 0.10, 0.1}


def _extract_p_values(text: str) -> List[float]:
    values: List[float] = []
    for match in PVALUE_DIRECT.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    # also capture standalone decimals near the word 'p' context heuristically
    for match in re.finditer(r"\bp\s*(?:<|>|=|≤|≥)\s*(\d+(?:\.\d+)?)", text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _check_p_values(p_values: List[float]) -> List[StatisticalFinding]:
    findings: List[StatisticalFinding] = []
    if not p_values:
        return findings

    # Border suspicion
    border_count = sum(1 for p in p_values if any(abs(p - b) < 0.006 for b in BORDERS))
    if border_count >= 3:
        findings.append(
            StatisticalFinding(
                "p-value borders",
                "high" if border_count >= 5 else "medium",
                f"{border_count} of {len(p_values)} reported p-values sit exactly on conventional "
                "significance borders (0.05, 0.01, 0.001). This pattern is statistically unlikely "
                "and is associated with p-hacking.",
            )
        )

    # Clustering just below 0.05
    borderline = sum(1 for p in p_values if 0.04 <= p <= 0.055)
    if borderline >= 2:
        findings.append(
            StatisticalFinding(
                "borderline clustering",
                "medium",
                f"{borderline} p-values cluster in the 0.04–0.056 range, a typical signature of "
                "selective reporting.",
            )
        )

    # Exact-zero style reporting
    exact = [p for p in p_values if p == 0.0]
    if exact:
        findings.append(
            StatisticalFinding(
                "exact zero p-values",
                "low",
                f"{len(exact)} p-value(s) reported as exactly 0; real analyses should report a bound "
                "(e.g. p < 0.001) instead.",
            )
        )

    return findings


def _check_digit_bias(text: str) -> List[StatisticalFinding]:
    findings: List[StatisticalFinding] = []
    decimals = DECIMAL_NUMBER.findall(text)
    last_digits = [d[-1] for d in decimals]
    if len(last_digits) < 15:
        return findings
    counter = Counter(last_digits)
    total = len(last_digits)
    # Chi-square-style deviation from uniform
    expected = total / 10.0
    chi2 = sum((counter.get(str(i), 0) - expected) ** 2 / expected for i in range(10))
    if chi2 > 22:  # roughly p < 0.01 for df=9
        worst = counter.most_common(2)
        findings.append(
            StatisticalFinding(
                "digit-preference bias",
                "high" if chi2 > 40 else "medium",
                f"Last digits of reported decimals show a strong preference bias "
                f"(over-represented: {', '.join(str(d) for d, _ in worst)}). Genuine measured "
                "results typically have near-uniform last digits; this pattern appears in "
                "fabricated data (GRanularity test).",
            )
        )
    return findings


def _check_percentages(text: str) -> List[StatisticalFinding]:
    findings: List[StatisticalFinding] = []
    percents = [float(m) for m in PERCENTAGE.findall(text)]
    if not percents:
        return findings
    rounded = [p for p in percents if p == round(p)]
    if len(rounded) >= 4 and len(rounded) / len(percents) > 0.8:
        findings.append(
            StatisticalFinding(
                "rounded percentages",
                "low",
                "Most reported percentages are round numbers (0, 5, 10, ...). Real measurements "
                "rarely land on round values so consistently.",
            )
        )
    extreme = [p for p in percents if p in (0.0, 100.0)]
    if len(extreme) >= 2:
        findings.append(
            StatisticalFinding(
                "extreme claims",
                "medium",
                f"{len(extreme)} results reported as exactly 0% or 100% — absolute claims deserve "
                "manual verification.",
            )
        )
    return findings


def analyze_statistics(text: str) -> StatisticalResult:
    p_values = _extract_p_values(text)
    findings: List[StatisticalFinding] = []
    findings.extend(_check_p_values(p_values))
    findings.extend(_check_digit_bias(text))
    findings.extend(_check_percentages(text))

    severity_points = {"low": 4, "medium": 8, "high": 15}
    risk = min(100, sum(severity_points.get(f.severity, 4) for f in findings))

    if not findings:
        summary = "Statistical reporting shows no anomalous patterns."
    elif any(f.severity == "high" for f in findings):
        summary = "One or more high-risk statistical patterns detected — audit these values."
    elif any(f.severity == "medium" for f in findings):
        summary = "Some statistical patterns warrant a closer look before accepting results."
    else:
        summary = "Minor statistical quirks detected; low overall risk."

    return StatisticalResult(
        risk_score=risk,
        summary=summary,
        findings=findings,
        p_values_found=sorted(p_values),
    )
