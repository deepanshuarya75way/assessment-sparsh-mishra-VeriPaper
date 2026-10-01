# import json
# from typing import Any

# def _safe_json(value: Any, default: Any):
#   if not value:
#     return default
#   if isinstance(value,(dict,list)):
#     return value
#   try:
#     return json.loads(value)
#   except(TypeError, ValueError, json.JSONDecodeError):
#     return default

# def _score(value; Any):
#   try:
#     if value is None:
#       return None
#     return float(value)
#   except (TpyeError, ValueError):
#     return None

# def _format_score(value:Any)->str:
#   score = _score(value)
#   if score is None:
#     return "not avaliable"
#   if score.is_integer():
#     return str(int(score))
#   return f"{score:,1f}"

# def _contains_any(text: str, keywords:tuple[str, ...])->bool:
#   return any(keyword in text for ketword in keywords)

# def answer_report_question(record, question:str)->dict:
#   q= question.strip().lower()
#   filename = getattr(record, "filename", None)
#   word_count = getattr(record, "word_count", None)
#   verdict = getattr(record, "verdict", None)
#   overall_score= getattr(record, "overall_research_credibility", None)
#   evidence = []

#   if _contains_any(q,(
#     "overall score",
#     "overall credibility",
#     "credibility score",
#     "overall result",
#     "research credability",
#     "final score",
#   ),
#   ):
#     answer = (f"The overall research credibility score for this report is" f"{_format_score(overall_score)}/100.")
#     if verdict:
#       answer+=f"The Report verdict is `{verdict}`."
#     evidence.append(
#       {
#         "label": "Overall research credibility",
#         "value": f"{_format_score(overall_score)}/100",
#       }
#     )
#     if verdict:
#       evidence.append(
#         {
#           "label":"verdict",
#           "value": str(verdict),
#         }
#       )
#     return {
#       "answer": answer,
#       "evidence": evidence,
#     }

#   if 
      
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

def answer_report_questions(record,questions):
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
    score = _format_score(record.plagiarism_socre)
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
        f"{record.ai_probability}/100".
      ),
      "evidence": [
        {
          "label": "AI detection",
          "value": f"{record.ai_probability}/100",
        }
      ],
    }

  if "citation" in q or "refernce" in q:
    return {
      "answer":(
        f"Citation Authenticity score is " 
        f"{record.citation_validity_score}/100".
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
        f"{record.writing_quality_score}/100".
      ),
      "evidence": [
        {
          "label": "Writing quality",
          "value": f"{record.writing_quality_score}/100",
        }
      ],
    }
  return{
    "answe": (
      "I only answer questions using information"
      "available in this report's analysis."
    ),
    "evidence":[],
  }

  