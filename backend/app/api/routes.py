"""Analysis API routes.

POST /api/analyze            — full paper analysis (all five modules)
GET  /api/history            — recent analysis history (from DB)
GET  /api/history/{id}       — replay a previous analysis
GET  /api/detector/config    — honest detector config + measured metrics
GET  /api/methodology        — how each module works (transparency page)
"""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi import Request
from fastapi import BackgroundTasks
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..core.config import settings
from ..core.database import get_db
from ..models.database import AnalysisResult as AnalysisRecord
from ..models.schemas import (
    AnalysisResult,
    AIDetectionModuleResult,
    CitationModuleResult,
    PlagiarismMatchSchema,
    PlagiarismModuleResult,
    ScoreContributions,
    SectionEvidence,
    StatisticalModuleResult,
    WritingCheck,
    WritingModuleResult,
)
from ..services import (
    ai_detection,
    citations as citation_svc,
    parsing,
    plagiarism as plagiarism_svc,
    scoring,
    statistics as statistics_svc,
    writing_quality,
)
from ..services.report import write_pdf_report
from ..services import pcv as pcv_svc
from ..services import web_attribution as web_svc
from ..models.schemas import ProvenanceModuleResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Simple in-memory rate limiting: per-IP analyze budget
_rate_window: dict = {}
RATE_LIMIT = int(settings.__dict__.get("RATE_LIMIT_PER_MINUTE", 20))
RATE_WINDOW_SEC = 60


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window = _rate_window.setdefault(client_ip, {"count": 0, "reset_at": now + RATE_WINDOW_SEC})
    if now > window["reset_at"]:
        window["count"] = 0
        window["reset_at"] = now + RATE_WINDOW_SEC
    window["count"] += 1
    if window["count"] > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many analyses from this address; try again in a minute.")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host if request.client else "unknown"


def _pdf_report_path(filename: str, record_id: int) -> str:
    safe_stem = "".join(ch for ch in Path(filename).stem if ch.isalnum() or ch in "_.-")[:60] or "paper"
    return f"{safe_stem}_{record_id}.pdf"


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_paper(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisResult:
    """Analyze an uploaded research paper with the full verification suite."""
    _check_rate_limit(_client_ip(request))

    try:
        payload = await file.read()
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

        doc = parsing.parse_document(file.filename, payload)
        if doc.word_count < 60:
            raise HTTPException(
                status_code=400,
                detail="The document appears too short for a meaningful analysis (under ~60 words). "
                "Please upload a research paper.",
            )

        # Deterministic cache by content hash: same file -> same result
        content_hash = hashlib.sha256(payload).hexdigest()
        existing = db.query(AnalysisRecord).filter(AnalysisRecord.file_hash == content_hash).first()
        if existing and existing.report_path:
            logger.info("Cache hit for %s", file.filename)
            record = existing
            report_url = f"/files/{Path(existing.report_path).name}"
        else:
            # --- Module analysis ---
            ai_result = ai_detection.detect_ai(doc.full_text)
            plagiarism_result = plagiarism_svc.analyze_plagiarism(doc.sections, doc.word_count)
            citation_result = citation_svc.analyze_citations(doc.full_text)
            statistics_result = statistics_svc.analyze_statistics(doc.full_text)
            writing_result = writing_quality.analyze_writing_quality(doc)

            # --- Provenance Chain Verification (PCV) layers ---
            citing_sentences = doc.citing_sentences
            verified_dois = [v.doi for v in citation_result.verified if v.valid]
            try:
                contamination = pcv_svc.trace_retraction_contamination(verified_dois)
                graph_anomaly = pcv_svc.detect_graph_anomalies(verified_dois)
                alignment = pcv_svc.check_claim_alignment(citing_sentences, verified_dois)
                fingerprint = pcv_svc.fingerprint_methodology(doc.full_text)
                pcv_score = pcv_svc.fuse_pcv_scores(
                    citation_result.validity_score, contamination, graph_anomaly, alignment, fingerprint
                )
                # Web attribution samples only the most similar corpus segments
                # to stay inside the free web-search quota.
                web_segments = [m.matched_text for m in plagiarism_result.matches[:5] if m.matched_text]
                web_result = web_svc.attribute_web_sources(web_segments)
            except Exception as exc:  # noqa: BLE001 — PCV never breaks the pipeline
                logger.warning("PCV failed: %s", exc)
                pcv_score = 50
                contamination = graph_anomaly = fingerprint = None
                alignment, web_result = {}, None

            breakdown = scoring.compute_credibility(
                ai_result, plagiarism_result, citation_result, statistics_result, writing_result
            )

            record = AnalysisRecord(
                filename=file.filename,
                file_size=len(payload),
                file_hash=content_hash,
                word_count=doc.word_count,
                plagiarism_score=plagiarism_result.plagiarism_score,
                ai_probability=ai_result.ai_probability,
                ai_confidence=ai_result.confidence,
                ai_engine=ai_result.engine,
                citation_validity_score=citation_result.validity_score,
                statistical_risk_score=statistics_result.risk_score,
                writing_quality_score=writing_result.score,
                overall_research_credibility=breakdown.credibility_score,
                verdict=breakdown.verdict,
                retraction_contamination_score=contamination.score if contamination else None,
                citation_graph_anomaly_score=graph_anomaly.score if graph_anomaly else None,
                claim_alignment_score=alignment.get("alignment_score") if alignment else None,
                methodology_fingerprint_score=fingerprint.score if fingerprint else None,
                provenance_score=pcv_score,
                plagiarism_matches=json.dumps(
                    [
                        {
                            "title": m.source_title,
                            "similarity": m.similarity,
                            "source": m.source_corpus,
                            "matched_text": m.matched_text,
                        }
                        for m in plagiarism_result.matches
                    ]
                ) or None,
                citation_details=json.dumps(
                    {
                        "total_dois": citation_result.total_dois,
                        "valid_dois": citation_result.valid_dois,
                        "invalid_dois": citation_result.invalid_dois,
                        "references_without_doi": citation_result.references_without_doi,
                        "verified": [
                            {"doi": v.doi, "title": v.title, "year": v.year, "valid": v.valid}
                            for v in citation_result.verified
                        ],
                    }
                ) or None,
                statistical_findings=json.dumps(
                    [
                        {"category": f.category, "severity": f.severity, "detail": f.detail}
                        for f in statistics_result.findings
                    ]
                ) or None,
                writing_checks=json.dumps(
                    [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "score": c.score,
                            "detail": c.detail,
                            "suggestions": c.suggestions,
                        }
                        for c in writing_result.checks
                    ]
                ) or None,
                sections=json.dumps(
                    [
                        {
                            "label": s.label,
                            "heading": s.heading,
                            "word_count": len(s.text.split()),
                        }
                        for s in doc.sections
                    ]
                ) or None,
                pcv_details=json.dumps(
                    {
                        "retracted_dois": contamination.retracted_dois if contamination else [],
                        "contamination_summary": contamination.summary if contamination else "",
                        "graph_density": graph_anomaly.density if graph_anomaly else 0.0,
                        "isolated_references": graph_anomaly.isolated_references if graph_anomaly else 0,
                        "graph_summary": graph_anomaly.summary if graph_anomaly else "",
                        "alignment_verdicts": alignment.get("verdicts", []) if alignment else [],
                        "alignment_summary": alignment.get("summary", "") if alignment else "",
                        "fingerprint_notes": {
                            "benford": fingerprint.benford_note if fingerprint else "",
                            "p_curve": fingerprint.p_curve_note if fingerprint else "",
                            "precision": fingerprint.precision_note if fingerprint else "",
                            "consistency": fingerprint.consistency_note if fingerprint else "",
                        } if fingerprint else {},
                        "web_matches": [
                            {
                                "url": w.url, "title": w.title, "snippet": w.snippet,
                                "matched_text": w.matched_text, "similarity": w.similarity,
                                "match_type": "uncited",
                            }
                            for w in (web_result.matches if web_result else [])
                        ],
                        "web_summary": web_result.summary if web_result else "",
                        "web_available": web_result.available if web_result else False,
                        "web_provider": web_result.provider if web_result else "",
                    }
                ) or None,
            )
            db.add(record)
            db.commit()
            db.refresh(record)

            report_name = _pdf_report_path(file.filename, record.id)
            write_pdf_report(doc, record, breakdown, citation_result, statistics_result, writing_result, report_name)
            record.report_path = str(settings.REPORTS_DIR / report_name)
            record.report_generated = True
            db.commit()
            db.refresh(record)
            report_url = f"/files/{report_name}"

        # --- Build the response payload ---
        sections_payload = json.loads(record.sections) if record.sections else []
        verified = json.loads(record.citation_details).get("verified", []) if record.citation_details else []
        if not plagiarism_svc._corpus_loaded:
            plag_summary = "Similarity corpus not loaded; only internal duplication checked."
        elif record.plagiarism_score <= 5:
            plag_summary = "No significant similarity to the indexed open corpus."
        else:
            plag_summary = "Similarity search completed."
        if record.ai_engine and "transformer" in record.ai_engine:
            ai_explanation = (
                "AI-likelihood estimated by the fine-tuned transformer classifier "
                "(trained on real arXiv abstracts vs. controlled AI rewrites)."
            )
        else:
            ai_explanation = "AI-likelihood estimated by the configured detection engine."

        result = AnalysisResult(
            filename=record.filename,
            author_name=record.author_name,
            institution=record.institution,
            full_text=record.full_text,
            analyzed_at=(record.analyzed_at or datetime.now(timezone.utc)).isoformat(),
            word_count=record.word_count,
            section_count=len(sections_payload),
            sections=[
                SectionEvidence(label=s["label"], heading=s["heading"], word_count=s["word_count"])
                for s in sections_payload
            ],
            plagiarism=PlagiarismModuleResult(
                score=record.plagiarism_score,
                summary=plag_summary,
                matches=[
                    PlagiarismMatchSchema(**m)
                    for m in (json.loads(record.plagiarism_matches) if record.plagiarism_matches else [])
                ],
            ),
            ai_detection=AIDetectionModuleResult(
                ai_probability=record.ai_probability,
                confidence=record.ai_confidence,
                explanation=ai_explanation,
                engine=record.ai_engine or "heuristic",
                model_version=ai_detection.get_detector_meta()["model_version"],
            ),
            citation=CitationModuleResult(
                validity_score=record.citation_validity_score,
                summary="Citations verified against CrossRef.",
                total_dois=json.loads(record.citation_details).get("total_dois", 0) if record.citation_details else 0,
                valid_dois=json.loads(record.citation_details).get("valid_dois", 0) if record.citation_details else 0,
                invalid_dois=json.loads(record.citation_details).get("invalid_dois", []) if record.citation_details else [],
                references_without_doi=json.loads(record.citation_details).get("references_without_doi", 0) if record.citation_details else 0,
                verified_dois=verified,
            ),
            statistics=StatisticalModuleResult(
                risk_score=record.statistical_risk_score,
                summary="Statistical pattern analysis completed.",
                findings=json.loads(record.statistical_findings) if record.statistical_findings else [],
                p_values_found=[],
            ),
            writing=WritingModuleResult(
                score=record.writing_quality_score,
                grade="Assessed" if record.writing_quality_score else "Not assessed",
                checks=[
                    WritingCheck(**c) for c in (json.loads(record.writing_checks) if record.writing_checks else [])
                ],
                section_map={s["label"]: s["heading"] for s in sections_payload},
            ),
            overall_research_credibility=record.overall_research_credibility,
            verdict=record.verdict,
            verdict_detail="See methodology for how this verdict is computed.",
            contributions=ScoreContributions(
                ai_detection=0.0, plagiarism=0.0, citation_validity=0.0,
                statistical_integrity=0.0, writing_standards=0.0,
            ),
            risk_triggers=[],
            action_items=["Review the detailed evidence in the downloaded report."],
            flags=[],
            provenance=_build_provenance(record),
            report_path=report_url,
            download_url=report_url,
        )
        return result
    except HTTPException:
        raise
    except parsing.ParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


# --- Asynchronous analysis queue (BackgroundTasks) ---
# Shared in-memory task store: task_id -> {"status": "queued"|"processing"|"done"|"error", "result": ...}
_task_store: dict = {}


@router.post("/analyze/async")
async def analyze_paper_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Start an asynchronous analysis. Returns immediately with a task_id.
    
    Client should poll GET /api/analyze/{task_id}/status until status == "done"."""
    _check_rate_limit(_client_ip(request))

    try:
        payload = await file.read()
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_SIZE_MB} MB)")

        # Deterministic cache by content hash: same file -> same result
        content_hash = hashlib.sha256(payload).hexdigest()
        existing = db.query(AnalysisRecord).filter(AnalysisRecord.file_hash == content_hash).first()
        if existing and existing.report_path:
            logger.info("Cache hit for %s", file.filename)
            record = existing
            report_url = f"/files/{Path(existing.report_path).name}"
        else:
            # Parse document to get filename for the record
            doc = parsing.parse_document(file.filename, payload)
            if doc.word_count < 60:
                raise HTTPException(
                    status_code=400,
                    detail="The document appears too short for a meaningful analysis (under ~60 words). "
                    "Please upload a research paper.",
                )
            # Create a placeholder record so we have an ID
            record = AnalysisRecord(
                filename=file.filename,
                file_size=len(payload),
                file_hash=content_hash,
                word_count=doc.word_count,
                plagiarism_score=0.0,
                ai_probability=0.0,
                ai_confidence=0.0,
                ai_engine=None,
                citation_validity_score=0.0,
                statistical_risk_score=0.0,
                writing_quality_score=0.0,
                overall_research_credibility=0.0,
                verdict="pending",
            )
            db.add(record)
            db.commit()
            db.refresh(record)

        import uuid
        task_id = str(uuid.uuid4())
        _task_store[task_id] = {
            "status": "queued",
            "record_id": record.id,
            "result": None,
            "report_url": report_url if (existing and existing.report_path) else None,
            "payload": payload,
            "filename": file.filename,
            "content_hash": content_hash,
            "is_cache_hit": existing is not None and existing.report_path is not None,
        }
        background_tasks.add_task(_process_analysis_task, task_id)
        return {"task_id": task_id, "status": "queued", "message": "Analysis started. Poll /api/analyze/{task_id}/status."}
    except HTTPException:
        raise
    except parsing.ParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Async analysis start failed")
        raise HTTPException(status_code=500, detail=f"Analysis start failed: {exc}")


@router.get("/analyze/{task_id}/status")
def get_analysis_status(task_id: str, db: Session = Depends(get_db)):
    """Poll the status of an async analysis. Returns status and full result when done."""
    task = _task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    if task["status"] == "done" and task["result"]:
        return {
            "status": "done",
            "record_id": task["record_id"],
            "result": task["result"],
        }
    elif task["status"] == "error":
        return {"status": "error", "error": task.get("error", "Unknown error")}
    elif task["status"] == "processing":
        return {"status": "processing", "message": "Analysis in progress..."}
    else:
        return {"status": "queued", "message": "Waiting to start..."}


async def _process_analysis_task(task_id: str):
    """Background worker: performs the full 5-module analysis."""
    import uuid as uuid_mod
    from ..services import (
        ai_detection,
        citations as citation_svc,
        parsing,
        plagiarism as plagiarism_svc,
        scoring,
        statistics as statistics_svc,
        writing_quality,
    )
    from ..services.report import write_pdf_report

    task = _task_store.get(task_id)
    if not task:
        return

    task["status"] = "processing"

    try:
        if task["is_cache_hit"]:
            task["status"] = "done"
            record = None  # Will be fetched below
            # Build response from existing record
            with next(get_db()) as db:
                record = db.query(AnalysisRecord).filter(AnalysisRecord.id == task["record_id"]).first()
            task["result"] = _build_response(record, ai_detection, plagiarism_svc, json)
        else:
            payload = task["payload"]
            filename = task["filename"]
            doc = parsing.parse_document(filename, payload)

            ai_result = ai_detection.detect_ai(doc.full_text)
            plagiarism_result = plagiarism_svc.analyze_plagiarism(doc.sections, doc.word_count)
            citation_result = citation_svc.analyze_citations(doc.full_text)
            statistics_result = statistics_svc.analyze_statistics(doc.full_text)
            writing_result = writing_quality.analyze_writing_quality(doc)

            # --- Provenance Chain Verification (PCV) layers ---
            citing_sentences = doc.citing_sentences
            verified_dois = [v.doi for v in citation_result.verified if v.valid]
            try:
                contamination = pcv_svc.trace_retraction_contamination(verified_dois)
                graph_anomaly = pcv_svc.detect_graph_anomalies(verified_dois)
                alignment = pcv_svc.check_claim_alignment(citing_sentences, verified_dois)
                fingerprint = pcv_svc.fingerprint_methodology(doc.full_text)
                pcv_score = pcv_svc.fuse_pcv_scores(
                    citation_result.validity_score, contamination, graph_anomaly, alignment, fingerprint
                )
                web_segments = [m.matched_text for m in plagiarism_result.matches[:5] if m.matched_text]
                web_result = web_svc.attribute_web_sources(web_segments)
            except Exception as exc:  # noqa: BLE001 — PCV never breaks the pipeline
                logger.warning("PCV failed: %s", exc)
                pcv_score = 50
                contamination = graph_anomaly = fingerprint = None
                alignment, web_result = {}, None

            breakdown = scoring.compute_credibility(
                ai_result, plagiarism_result, citation_result, statistics_result, writing_result
            )

            with next(get_db()) as db:
                record = db.query(AnalysisRecord).filter(AnalysisRecord.id == task["record_id"]).first()
                if not record:
                    task["status"] = "error"
                    task["error"] = "Record not found"
                    return
                record.plagiarism_score = plagiarism_result.plagiarism_score
                record.ai_probability = ai_result.ai_probability
                record.ai_confidence = ai_result.confidence
                record.ai_engine = ai_result.engine
                record.citation_validity_score = citation_result.validity_score
                record.statistical_risk_score = statistics_result.risk_score
                record.writing_quality_score = writing_result.score
                record.overall_research_credibility = breakdown.credibility_score
                record.verdict = breakdown.verdict
                record.retraction_contamination_score = contamination.score if contamination else None
                record.citation_graph_anomaly_score = graph_anomaly.score if graph_anomaly else None
                record.claim_alignment_score = alignment.get("alignment_score") if alignment else None
                record.methodology_fingerprint_score = fingerprint.score if fingerprint else None
                record.provenance_score = pcv_score
                record.plagiarism_matches = json.dumps(
                    [
                        {
                            "title": m.source_title,
                            "similarity": m.similarity,
                            "source": m.source_corpus,
                            "matched_text": m.matched_text,
                        }
                        for m in plagiarism_result.matches
                    ]
                ) or None
                record.citation_details = json.dumps(
                    {
                        "total_dois": citation_result.total_dois,
                        "valid_dois": citation_result.valid_dois,
                        "invalid_dois": citation_result.invalid_dois,
                        "references_without_doi": citation_result.references_without_doi,
                        "verified": [
                            {"doi": v.doi, "title": v.title, "year": v.year, "valid": v.valid}
                            for v in citation_result.verified
                        ],
                    }
                ) or None
                record.statistical_findings = json.dumps(
                    [
                        {"category": f.category, "severity": f.severity, "detail": f.detail}
                        for f in statistics_result.findings
                    ]
                ) or None
                record.writing_checks = json.dumps(
                    [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "score": c.score,
                            "detail": c.detail,
                            "suggestions": c.suggestions,
                        }
                        for c in writing_result.checks
                    ]
                ) or None
                record.sections = json.dumps(
                    [
                        {
                            "label": s.label,
                            "heading": s.heading,
                            "word_count": len(s.text.split()),
                        }
                        for s in doc.sections
                    ]
                ) or None
                record.pcv_details = json.dumps(
                    {
                        "retracted_dois": contamination.retracted_dois if contamination else [],
                        "contamination_summary": contamination.summary if contamination else "",
                        "graph_density": graph_anomaly.density if graph_anomaly else 0.0,
                        "isolated_references": graph_anomaly.isolated_references if graph_anomaly else 0,
                        "graph_summary": graph_anomaly.summary if graph_anomaly else "",
                        "alignment_verdicts": alignment.get("verdicts", []) if alignment else [],
                        "alignment_summary": alignment.get("summary", "") if alignment else "",
                        "fingerprint_notes": {
                            "benford": fingerprint.benford_note if fingerprint else "",
                            "p_curve": fingerprint.p_curve_note if fingerprint else "",
                            "precision": fingerprint.precision_note if fingerprint else "",
                            "consistency": fingerprint.consistency_note if fingerprint else "",
                        } if fingerprint else {},
                        "web_matches": [
                            {
                                "url": w.url, "title": w.title, "snippet": w.snippet,
                                "matched_text": w.matched_text, "similarity": w.similarity,
                                "match_type": "uncited",
                            }
                            for w in (web_result.matches if web_result else [])
                        ],
                        "web_summary": web_result.summary if web_result else "",
                        "web_available": web_result.available if web_result else False,
                        "web_provider": web_result.provider if web_result else "",
                    }
                ) or None
                db.commit()
                db.refresh(record)

                report_name = _pdf_report_path(filename, record.id)
                write_pdf_report(doc, record, breakdown, citation_result, statistics_result, writing_result, report_name)
                record.report_path = str(settings.REPORTS_DIR / report_name)
                record.report_generated = True
                db.commit()
                db.refresh(record)
                task["report_url"] = f"/files/{report_name}"

            task["result"] = _build_response(record, ai_detection, plagiarism_svc, json)
            task["status"] = "done"

        # Clean up payload from memory
        task["payload"] = None
    except Exception as exc:
        logger.exception("Async analysis task failed for %s", task_id)
        task["status"] = "error"
        task["error"] = str(exc)
        task["payload"] = None


def _build_provenance(record) -> Optional[dict]:
    """Build the ProvenanceModuleResult payload from a DB record (None for old records)."""
    details = json.loads(record.pcv_details) if record.pcv_details else None
    if details is None and record.provenance_score is None:
        return None
    details = details or {}
    return {
        "score": record.provenance_score if record.provenance_score is not None else 50,
        "summary": _provenance_summary(details, record),
        "fingerprint_score": record.methodology_fingerprint_score if record.methodology_fingerprint_score is not None else 50,
        "fingerprint_summary": _fingerprint_summary(details),
        "fingerprint_notes": details.get("fingerprint_notes", {}) or {},
        "alignment_score": record.claim_alignment_score if record.claim_alignment_score is not None else 50,
        "alignment_summary": details.get("alignment_summary", "Claim alignment was not computed."),
        "alignment_verdicts": details.get("alignment_verdicts", []) or [],
        "contamination_score": record.retraction_contamination_score if record.retraction_contamination_score is not None else 50,
        "contamination_summary": details.get("contamination_summary", "Retraction tracing was not run."),
        "retracted_dois": details.get("retracted_dois", []) or [],
        "graph_anomaly_score": record.citation_graph_anomaly_score if record.citation_graph_anomaly_score is not None else 50,
        "graph_anomaly_summary": details.get("graph_summary", "The reference graph was not analyzed."),
        "graph_density": details.get("graph_density", 0.0) or 0.0,
        "isolated_references": details.get("isolated_references", 0) or 0,
        "web_matches": details.get("web_matches", []) or [],
        "web_summary": details.get("web_summary", ""),
        "web_available": details.get("web_available", False) is True,
        "web_provider": details.get("web_provider", ""),
    }


def _provenance_summary(details: dict, record) -> str:
    retracted = (details.get("retracted_dois") or []) if isinstance(details, dict) else []
    if retracted:
        return (
            f"{len(retracted)} cited reference(s) were later retracted; the "
            f"supporting evidence should be re-checked."
        )
    if record.provenance_score is None:
        return "Provenance verification was not available for this analysis."
    if record.provenance_score >= 75:
        return "Claims trace back to intact, interconnected scholarly sources and the statistical fingerprints look natural."
    if record.provenance_score >= 50:
        return "Some provenance signals are weak — check the alignment and fingerprint notes below."
    return "Multiple provenance checks failed; the paper's supporting evidence needs careful review."


def _fingerprint_summary(details: dict) -> str:
    notes = details.get("fingerprint_notes", {}) or {} if isinstance(details, dict) else {}
    flags = [k for k, v in notes.items() if v and "irregular" in v.lower()]
    if flags:
        return f"Statistical forensics flagged {', '.join(flags)}. Review the raw data."
    if notes:
        return "Statistical fingerprints look natural for experimental research."
    return "Methodological fingerprinting was not computed."


def _build_response(record, ai_detection, plagiarism_svc, json):
    """Build the AnalysisResult dict from a DB record (extracted for reuse by async)."""
    sections_payload = json.loads(record.sections) if record.sections else []
    verified = json.loads(record.citation_details).get("verified", []) if record.citation_details else []
    if not plagiarism_svc._corpus_loaded:
        plag_summary = "Similarity corpus not loaded; only internal duplication checked."
    elif record.plagiarism_score <= 5:
        plag_summary = "No significant similarity to the indexed open corpus."
    else:
        plag_summary = "Similarity search completed."
    if record.ai_engine and "transformer" in str(record.ai_engine):
        ai_explanation = (
            "AI-likelihood estimated by the fine-tuned transformer classifier "
            "(trained on real arXiv abstracts vs. controlled AI rewrites)."
        )
    else:
        ai_explanation = "AI-likelihood estimated by the configured detection engine."

    report_url = task_report_url = f"/files/{Path(record.report_path).name}" if record.report_path else None

    return {
        "filename": record.filename,
        "analyzed_at": (record.analyzed_at or datetime.now(timezone.utc)).isoformat(),
        "word_count": record.word_count,
        "section_count": len(sections_payload),
        "sections": [
            {"label": s["label"], "heading": s["heading"], "word_count": s["word_count"]}
            for s in sections_payload
        ],
        "plagiarism": {
            "score": record.plagiarism_score,
            "summary": plag_summary,
            "matches": json.loads(record.plagiarism_matches) if record.plagiarism_matches else [],
        },
        "ai_detection": {
            "ai_probability": record.ai_probability,
            "confidence": record.ai_confidence,
            "explanation": ai_explanation,
            "engine": record.ai_engine or "heuristic",
            "model_version": ai_detection.get_detector_meta()["model_version"],
        },
        "citation": {
            "validity_score": record.citation_validity_score,
            "summary": "Citations verified against CrossRef.",
            "total_dois": json.loads(record.citation_details).get("total_dois", 0) if record.citation_details else 0,
            "valid_dois": json.loads(record.citation_details).get("valid_dois", 0) if record.citation_details else 0,
            "invalid_dois": json.loads(record.citation_details).get("invalid_dois", []) if record.citation_details else [],
            "references_without_doi": json.loads(record.citation_details).get("references_without_doi", 0) if record.citation_details else 0,
            "verified_dois": verified,
        },
        "statistics": {
            "risk_score": record.statistical_risk_score,
            "summary": "Statistical pattern analysis completed.",
            "findings": json.loads(record.statistical_findings) if record.statistical_findings else [],
            "p_values_found": [],
        },
        "writing": {
            "score": record.writing_quality_score,
            "grade": "Assessed" if record.writing_quality_score else "Not assessed",
            "checks": json.loads(record.writing_checks) if record.writing_checks else [],
            "section_map": {s["label"]: s["heading"] for s in sections_payload},
        },
        "overall_research_credibility": record.overall_research_credibility,
        "verdict": record.verdict,
        "verdict_detail": "See methodology for how this verdict is computed.",
        "contributions": {
            "ai_detection": 0.0, "plagiarism": 0.0, "citation_validity": 0.0,
            "statistical_integrity": 0.0, "writing_standards": 0.0,
        },
        "risk_triggers": [],
        "action_items": ["Review the detailed evidence in the downloaded report."],
        "flags": [],
        "provenance": _build_provenance(record),
        "report_path": report_url,
        "download_url": report_url,
    }




@router.get("/history")
def list_history(limit: int = 25, db: Session = Depends(get_db)):
    """Recent analysis history."""
    records = db.query(AnalysisRecord).order_by(desc(AnalysisRecord.analyzed_at)).limit(min(limit, 100)).all()
    return {
        "count": len(records),
        "results": [r.to_dict() for r in records],
    }


@router.get("/history/{record_id}")
def get_history_item(record_id: int, db: Session = Depends(get_db)):
    """Replay a previous analysis."""
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found")
    return record.to_dict()


@router.get("/detector/config")
def detector_config():
    """Honest detector configuration: which engine is loaded and its measured metrics."""
    meta = ai_detection.get_detector_meta()
    corpus_loaded = plagiarism_svc._corpus_loaded
    return {
        "ai_detection": meta,
        "plagiarism_corpus_loaded": corpus_loaded,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_MB,
        "modules": ["ai_detection", "plagiarism_similarity", "citation_validation", "statistical_integrity", "writing_standards"],
        "note": "Scores are likelihood signals, not proof. Never use as sole evidence of misconduct.",
    }


@router.get("/detector/status")
def detector_status():
    """Live diagnostic of the transformer detector state (files + real test inference).

    This exists so the exact failure point is visible without dashboard access.
    """
    out = {
        "model_root": str(settings.MODEL_PATH),
        "transformer_dir": str(settings.TRANSFORMER_MODEL_DIR),
        "files": {},
        "diagnosis": "",
    }
    tf_dir = settings.TRANSFORMER_MODEL_DIR
    # Force the transformer load directly (bypasses the once-per-process guard)
    # so the exact failure reason and memory footprint are always visible.
    before_kb = _rss_kb()
    load_ok = ai_detection.load_transformer_detector(str(tf_dir))
    after_kb = _rss_kb()
    out["load_attempted"] = True
    out["load_ok"] = load_ok
    out["memory_before_kb"] = before_kb
    out["memory_after_kb"] = after_kb
    out["transformer_load_error"] = ai_detection.get_detector_meta().get("transformer_load_error")
    if tf_dir.exists():
        for entry in sorted(tf_dir.iterdir()):
            out["files"][entry.name] = entry.stat().st_size
    else:
        out["diagnosis"] = "transformer directory missing"
        return out

    try:
        score = ai_detection.transformer_ai_score("This is a short diagnostic sentence.")
        if score is not None:
            out["diagnosis"] = f"transformer inference OK (test score={score:.4f})"
        else:
            out["diagnosis"] = (
                "transformer load failed silently; see deploy logs for the exact "
                "'Transformer detector load failed' traceback"
            )
        out["test_inference_score"] = round(score, 4) if score is not None else None
    except Exception as exc:  # pragma: no cover
        out["diagnosis"] = f"test inference crashed: {exc}"
    return out


def _rss_kb() -> int:
    """Current process peak RSS in KB (Linux only)."""
    import resource

    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0

@router.get("/detector/env")
def detector_env():
    """Runtime package probe: shows what pip actually installed in the container.

    Used to diagnose missing dependency installs caused by build caching.
    """
    import subprocess

    out: dict = {}
    for pkg in ("onnxruntime", "torch", "transformers", "fastapi", "scikit-learn"):
        try:
            res = subprocess.run(
                ["pip", "show", pkg], capture_output=True, text=True, timeout=20
            )
            out[pkg] = (
                res.stdout.strip().split("\n", 1)[1] if res.stdout.strip() else "NOT INSTALLED"
            )
        except Exception as exc:  # pragma: no cover
            out[pkg] = f"probe error: {exc}"
    return out


@router.get("/methodology")
def methodology():
    """Explain how each module works."""
    return {
        "modules": {
            "ai_detection": (
                "Text-style analysis using a trained model (with a fast heuristic fallback). "
                "It measures word-pattern regularity, repetition, and phrasing common in "
                "machine-generated text. Confidence is 'low' when signals are mixed."
            ),
            "plagiarism_similarity": (
                "Term-overlap similarity search (TF-IDF cosine similarity) against a bundled "
                "corpus of open academic abstracts, plus an internal duplication check. "
                "It does NOT scan the whole internet."
            ),
            "citation_validation": (
                "DOIs are resolved against the public CrossRef registry. A reference without "
                "a DOI or with an unresolvable DOI is flagged for manual review."
            ),
            "statistical_integrity": (
                "Checks reported p-values for border clustering and last-digit bias "
                "(GRanularity-style), plus implausible percentage claims."
            ),
            "writing_standards": (
                "Verifies IMRaD/IEEE section structure, formal academic tone, consistent "
                "citation style (APA vs IEEE), citation-text linkage, and figure/table "
                "referencing integrity."
            ),
        },
        "verdict_policy": {
            "Credible": "Score >= 75",
            "Needs review": "Score 50-74",
            "High risk": "Score < 50",
        },
        "limitations": [
            "Formal or non-native English writing can raise the AI-likelihood score.",
            "Plagiarism search is limited to the bundled open corpus.",
            "Results are assistive signals, not conclusive proof.",
        ],
    }
