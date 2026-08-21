"""PDF evidence report generation with per-module findings."""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from ..core.config import settings

logger = logging.getLogger(__name__)

def write_pdf_report(doc, record, breakdown, citation_result, statistics_result, writing_quality_result, filename):
    """Generate a professional research integrity report with cover page."""
    output_path = settings.REPORTS_DIR / filename
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    doc_pdf = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=2*cm, rightMargin=2*cm, 
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=24, spaceAfter=30, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'], fontSize=14, spaceAfter=12, alignment=TA_CENTER, textColor=colors.grey)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading1'], fontSize=16, spaceBefore=20, spaceAfter=12, textColor=colors.HexColor("#1e293b"))
    
    story = []

    # --- COVER PAGE ---
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("RESEARCH INTEGRITY REPORT", title_style))
    story.append(Paragraph("VeriPaper Autonomous Verification Platform", subtitle_style))
    story.append(Spacer(1, 2 * cm))
    
    meta_data = [
        [Paragraph("<b>Document:</b>", styles['Normal']), Paragraph(record.filename, styles['Normal'])],
        [Paragraph("<b>Author:</b>", styles['Normal']), Paragraph(record.author_name or "Not Specified", styles['Normal'])],
        [Paragraph("<b>Institution:</b>", styles['Normal']), Paragraph(record.institution or "Not Specified", styles['Normal'])],
        [Paragraph("<b>Analysis Date:</b>", styles['Normal']), Paragraph(record.analyzed_at.strftime('%B %d, %Y'), styles['Normal'])],
        [Paragraph("<b>Report ID:</b>", styles['Normal']), Paragraph(f"VP-{record.id:06d}", styles['Normal'])],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 10 * cm])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    
    story.append(Spacer(1, 4 * cm))
    
    # Credibility Badge
    score_color = colors.green if record.overall_research_credibility >= 75 else (colors.orange if record.overall_research_credibility >= 50 else colors.red)
    badge_data = [[Paragraph(f"<font size=12 color=white>OVERALL CREDIBILITY</font><br/><font size=36 color=white><b>{record.overall_research_credibility}%</b></font>", ParagraphStyle('Badge', alignment=TA_CENTER))]]
    badge = Table(badge_data, colWidths=[6 * cm])
    badge.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), score_color),
        ('ROUNDEDCORNERS', [10, 10, 10, 10]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
    ]))
    story.append(badge)
    
    story.append(PageBreak())

    # --- EXECUTIVE SUMMARY ---
    story.append(Paragraph("1. Integrity Overview", heading_style))
    
    summary_text = f"The document '{record.filename}' has undergone a comprehensive multi-module verification. "
    if record.overall_research_credibility >= 75:
        summary_text += "The analysis indicates a high level of research integrity with standard citation patterns and natural statistical distributions."
    elif record.overall_research_credibility >= 50:
        summary_text += "The analysis identifies moderate risks that require manual review, particularly in the areas of citation alignment and statistical consistency."
    else:
        summary_text += "Significant integrity risks were detected. Multiple forensic modules flagged irregularities that deviate from standard academic writing and reporting."
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.5 * cm))
    
    # Verdict Box
    verdict_data = [[Paragraph(f"<b>VERDICT: {record.verdict.upper()}</b>", ParagraphStyle('Verdict', textColor=colors.white, alignment=TA_CENTER))]]
    verdict_table = Table(verdict_data, colWidths=[14 * cm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), score_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 1 * cm))

    # --- MODULE DETAILS ---
    story.append(Paragraph("2. Detailed Module Findings", heading_style))
    
    modules = [
        ("AI Detection", f"{record.ai_probability}% probability (Engine: {record.ai_engine})"),
        ("Similarity Check", f"{record.plagiarism_score}% overlap with indexed corpus"),
        ("Citation Validity", f"{record.citation_validity_score}% CrossRef validation score"),
        ("Statistical Integrity", f"{record.statistical_risk_score}% risk based on distribution forensics"),
        ("Writing Quality", f"{record.writing_quality_score}% adherence to academic standards"),
    ]
    
    mod_table = Table(modules, colWidths=[5 * cm, 9 * cm])
    mod_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(mod_table)

    # --- SIMILARITY TAXONOMY ---
    story.append(Paragraph("3. Similarity & Attribution Taxonomy", heading_style))
    plag_matches = json.loads(record.plagiarism_matches) if record.plagiarism_matches else []
    if plag_matches:
        match_data = [["Source Title", "Similarity", "Type"]]
        for m in plag_matches[:10]:
            m_type = m.get("match_type", "uncited").capitalize()
            match_data.append([m["title"], f"{m['similarity']}%", m_type])
        
        t = Table(match_data, colWidths=[8 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No significant similarities detected.", styles['Normal']))

    # Footer
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("<i>Generated by VeriPaper Autonomous Platform. This report is intended for academic assistance only.</i>", styles['Normal']))

    doc_pdf.build(story)
    
    # Emergency memory cleanup: clear large lists and trigger GC
    story.clear()
    import gc
    gc.collect()
    
    logger.info("Professional report written: %s", output_path)
