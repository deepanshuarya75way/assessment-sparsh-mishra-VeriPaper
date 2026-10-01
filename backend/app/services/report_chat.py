import json


def _get_json(value, default=None):
    if not value:
        return default if default is not None else {}

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default if default is not None else {}


def answer_report_question(record, questions):
    q = questions.strip().lower()

    # ---------------------------------------------------------
    # Overall credibility
    # ---------------------------------------------------------
    if "overall" in q or "credibility" in q:
        score = record.overall_research_credibility

        return {
            "answer": (
                f"Overall research credibility is {score}/100."
            ),
            "evidence": [
                {
                    "label": "Overall credibility",
                    "value": f"{score}/100",
                },
                {
                    "label": "Verdict",
                    "value": record.verdict,
                },
            ],
        }

    # ---------------------------------------------------------
    # Plagiarism / similarity
    # ---------------------------------------------------------
    if "plagiarism" in q or "similarity" in q:
        score = record.plagiarism_score
        matches = _get_json(record.plagiarism_matches, [])

        evidence = [
            {
                "label": "Similarity & Plagiarism score",
                "value": f"{score}/100",
            }
        ]

        if matches:
            for match in matches[:3]:
                evidence.append(
                    {
                        "label": "Matched source",
                        "value": (
                            f"{match.get('title', 'Unknown source')} "
                            f"({match.get('similarity', 0)}% similarity)"
                        ),
                    }
                )

        return {
            "answer": (
                f"Similarity & Plagiarism score is {score}/100."
            ),
            "evidence": evidence,
        }

    # ---------------------------------------------------------
    # AI-written content
    # ---------------------------------------------------------
    if (
        "ai" in q
        or "artificial intelligence" in q
        or "ai-written" in q
        or "ai written" in q
    ):
        score = record.ai_probability

        return {
            "answer": (
                f"The AI-written content likelihood score is "
                f"{score}/100. This is a likelihood signal, "
                f"not proof that the text was AI-generated."
            ),
            "evidence": [
                {
                    "label": "AI detection score",
                    "value": f"{score}/100",
                },
                {
                    "label": "Confidence",
                    "value": record.ai_confidence,
                },
                {
                    "label": "Detection engine",
                    "value": record.ai_engine or "Unknown",
                },
            ],
        }

    # ---------------------------------------------------------
    # Citation / references
    # ---------------------------------------------------------
    if (
        "citation" in q
        or "reference" in q
        or "references" in q
        or "doi" in q
    ):
        score = record.citation_validity_score
        details = _get_json(record.citation_details, {})

        total_dois = details.get("total_dois", 0)
        valid_dois = details.get("valid_dois", 0)
        invalid_dois = details.get("invalid_dois", [])

        return {
            "answer": (
                f"Citation Authenticity score is {score}/100."
            ),
            "evidence": [
                {
                    "label": "Citation validity",
                    "value": f"{score}/100",
                },
                {
                    "label": "Total DOIs",
                    "value": str(total_dois),
                },
                {
                    "label": "Valid DOIs",
                    "value": str(valid_dois),
                },
                {
                    "label": "Invalid DOIs",
                    "value": str(len(invalid_dois)),
                },
            ],
        }

    # ---------------------------------------------------------
    # Statistical integrity
    # ---------------------------------------------------------
    if (
        "statistic" in q
        or "statistical" in q
        or "numeric" in q
        or "number" in q
    ):
        integrity_score = 100 - record.statistical_risk_score
        findings = _get_json(record.statistical_findings, [])

        evidence = [
            {
                "label": "Statistical integrity",
                "value": f"{integrity_score}/100",
            },
            {
                "label": "Risk score",
                "value": f"{record.statistical_risk_score}/100",
            },
        ]

        for finding in findings[:3]:
            evidence.append(
                {
                    "label": finding.get("category", "Statistical finding"),
                    "value": (
                        f"{finding.get('severity', 'Unknown')}: "
                        f"{finding.get('detail', '')}"
                    ),
                }
            )

        return {
            "answer": (
                f"Statistical Integrity score is "
                f"{integrity_score}/100."
            ),
            "evidence": evidence,
        }

    # ---------------------------------------------------------
    # Writing quality
    # ---------------------------------------------------------
    if (
        "writing" in q
        or "grammar" in q
        or "grammer" in q
        or "standards" in q
    ):
        score = record.writing_quality_score

        return {
            "answer": (
                f"Writing & Standards score is {score}/100."
            ),
            "evidence": [
                {
                    "label": "Writing quality",
                    "value": f"{score}/100",
                }
            ],
        }

    # ---------------------------------------------------------
    # Unsupported question
    # ---------------------------------------------------------
    return {
        "answer": (
            "I can only answer questions using information "
            "stored in this report's analysis."
        ),
        "evidence": [],
    }