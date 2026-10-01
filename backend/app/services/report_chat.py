import json

def _get_json(value, default=None):
  if not value:
    return default if default is not None else{}
  if isinstance(value, (dict, list)):
    return value
  try:
    return json.loads(value)
  except(TypeError, ValueError):
    return default if default is not None else {}

def answer_report_question(record,questions):
  q=questions.strip().lower()
  if "overall" in q or "credibility" in q:
    return {
      "answer": (f"Overall research credibility is " 
      f"{record.overall_research_credibility}/100."
      ),
      "evidence":[
        {
          "label": "Overall credibility",
          "value": f"{record.overall_research_credibility}/100",
        }
      ],
    }
  if "plagiarism" in q or "similarity" in q:
    score = record.plagiarism_score
    return {
      "answer":(
        f"Similarity & Plagiarism score is{score}/100."
      ),
      "evidence": [
        {
          "label": "Plagiarsim score",
          "value": f"{score}/100",
        }
      ],
    }

  if "ai" in q:
    return {
      "answer":(
        f"AI written Content score is "
        f"{record.ai_probability}/100"
      ),
      "evidence": [
        {
          "label": "AI detection",
          "value": f"{record.ai_probability}/100",
        }
      ],
    }

  if "citation" in q or "refernce" in q or "doi" in q:
    return {
      "answer":(
        f"Citation Authenticity score is " 
        f"{record.citation_validity_score}/100"
      ),
      "evidence": [
        {
          "label": "Citation validity",
          "value": f"{record.citation_validity_score}/100",
        }
      ],
    }

  if "statistic" in q or "numeric" in q:
    score=100-record.statistical_risk_score
    return {
      "answer":
        f"record.Statistical Integrity score is {score}/100",
      "evidence": [
        {
          "label": "Statistical integrity",
          "value": f"{score}/100",
        }
      ],
    }

  if "writing" in q or "grammer" in q:
    return {
      "answer":(
        f"writing & Standards score is "
        f"{record.writing_quality_score}/100"
      ),
      "evidence": [
        {
          "label": "Writing quality",
          "value": f"{record.writing_quality_score}/100",
        }
      ],
    }
  return{
    "answer": (
      "I only answer questions using information"
      "available in this report's analysis."
    ),
    "evidence":[],
  }

  