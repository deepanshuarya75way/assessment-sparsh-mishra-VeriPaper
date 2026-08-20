"""Document parsing service.

Extracts text from PDF, DOCX, and TXT uploads, verifies file magic bytes,
and segments the document into labeled sections for per-section analysis.
"""
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Magic byte signatures ---------------------------------------------------
MAGIC = {
    b"%PDF": ".pdf",
    b"PK\x03\x04": ".docx",  # Office Open XML containers are ZIP files
}

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ParseError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass
class Section:
    """A labeled segment of the paper (e.g. Abstract, Introduction, ...)."""

    label: str  # "abstract", "introduction", "methods", "results", "discussion", "conclusion", "references", "other"
    heading: str  # Original heading text, if any
    text: str


@dataclass
class ParsedDocument:
    filename: str
    extension: str
    sections: List[Section] = field(default_factory=list)
    full_text: str = ""
    word_count: int = 0

    @property
    def section_texts(self) -> List[str]:
        return [s.text for s in self.sections if s.text.strip()]

    @property
    def citing_sentences(self) -> List[str]:
        """Claim-bearing sentences suitable for provenance/alignment checks.

        Pulls sentences from the body sections (abstract, introduction,
        methods, results, discussion, conclusion) — where claims live —
        skips the reference list, and caps the output at 40 sentences so
        downstream CrossRef lookups stay inside polite-pool budgets.
        """
        body_labels = {"abstract", "introduction", "methods", "results", "discussion", "conclusion"}
        body_text = " ".join(s.text for s in self.sections if s.label in body_labels)
        if not body_text.strip():
            body_text = self.full_text
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body_text.replace("\n", " "))]
        candidates = [s for s in sentences if 12 <= len(s.split()) <= 60]
        # Prefer sentences that cite something or state a claim.
        prioritized = [
            s for s in candidates
            if CITATION_STYLE_IEEE.search(s) or CITATION_STYLE_APA.search(s) or re.search(r"\b(10\.\d{4,9}/\S+)\b", s)
        ]
        prioritized = [s for s in prioritized if s not in {""}] if prioritized else []
        pool = prioritized + [s for s in candidates if s not in prioritized]
        seen, out = set(), []
        for s in pool:
            key = s.lower()[:80]
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out[:40]


# --- Section heading patterns ------------------------------------------------
SECTION_LABELS = [
    ("abstract", re.compile(r"^\s*(a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\s*\.?\s*$", re.I)),
    ("introduction", re.compile(r"^\s*(\d*\s*\.?\s*)?(intro|background|motivation)\w*\s*$", re.I)),
    ("methods", re.compile(r"^\s*(\d*\s*\.?\s*)?(methods|methodology|materials and methods|approach|system|model|experimental setup)\w*\s*$", re.I)),
    ("results", re.compile(r"^\s*(\d*\s*\.?\s*)?(results|findings|experiments?|evaluation|performance)\s*$", re.I)),
    ("discussion", re.compile(r"^\s*(\d*\s*\.?\s*)?(discussion|analysis|limitations)\s*$", re.I)),
    ("conclusion", re.compile(r"^\s*(\d*\s*\.?\s*)?(conclusion|concluding remarks|summary)\s*$", re.I)),
    ("references", re.compile(r"^\s*(references|bibliography|works cited)\s*\.?\s*$", re.I)),
    ("acknowledgements", re.compile(r"^\s*(acknowledg(e|ement)s?|funding|disclosure)\s*$", re.I)),
]

CITATION_STYLE_IEEE = re.compile(r"\[\d+\]")
CITATION_STYLE_APA = re.compile(r"\([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)?,?\s+\d{4}\)")


def detect_file_type(payload: bytes) -> Tuple[str, str]:
    """Return (detected_extension, error_reason). Extension from magic bytes."""
    for signature, extension in MAGIC.items():
        if payload[: len(signature)] == signature:
            return extension, ""
    if all(c < 128 or c in b"\r\n\t" or (32 <= c < 127) for c in payload[:1024]):
        return ".txt", ""
    return "", "Unrecognized or unsupported file content"


def validate_upload(filename: str, payload: bytes) -> str:
    """Validate extension + magic bytes. Raises ParseError on mismatch."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ParseError(f"Unsupported file type: {extension or '(no extension)'}. Allowed: {allowed}")
    detected, reason = detect_file_type(payload)
    if detected and detected != extension:
        raise ParseError(
            f"File content does not match the extension .{extension.lstrip('.')}. "
            "Please upload the correct file format."
        )
    if not detected and reason:
        raise ParseError(f"Unable to read file: {reason}")
    return extension or detected


def _extract_pdf(payload: bytes) -> str:
    import pymupdf

    doc = pymupdf.open(stream=payload, filetype="pdf")
    parts: List[str] = []
    for page in doc:
        text = page.get_text("text")
        parts.append(text)
    doc.close()
    extracted = "\n".join(parts)
    if len(extracted.split()) < 20:
        raise ParseError(
            "Very little text could be extracted from this PDF. "
            "The file may be image-based (scanned); OCR is not available in this version."
        )
    return extracted


def _extract_docx(payload: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(payload))
    parts: List[str] = []
    for block in document.element.body.iter():
        tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag
        if tag == "p":
            # Determine style
            style_el = block.find(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr/"
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle"
            )
            text = "".join(node.text or "" for node in block.iter()
                           if node.tag.endswith("}t"))
            if not text.strip():
                continue
            is_heading = False
            if style_el is not None:
                style_name = style_el.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", ""
                )
                is_heading = style_name.lower().startswith("heading") or style_name.lower().startswith("title")
            if is_heading:
                parts.append("\n" + text.strip() + "\n")
            else:
                parts.append(text.strip())
    return "\n".join(parts)


def _extract_txt(payload: bytes) -> str:
    return payload.decode("utf-8", errors="ignore")


def extract_text(filename: str, payload: bytes) -> str:
    """Extract raw text from an uploaded file."""
    extension = validate_upload(filename, payload)
    if extension == ".pdf":
        return _extract_pdf(payload)
    if extension == ".docx":
        return _extract_docx(payload)
    return _extract_txt(payload)


def _normalize_label(heading: str) -> Optional[str]:
    for label, pattern in SECTION_LABELS:
        if pattern.match(heading):
            return label
    return None


def _merge_short_sections(sections: List[Section]) -> List[Section]:
    """Merge very small trailing 'other' sections into the previous section."""
    merged: List[Section] = []
    for section in sections:
        if section.label == "other" and len(section.text.split()) < 15 and merged:
            merged[-1].text = (merged[-1].text + "\n\n" + section.text).strip()
            continue
        merged.append(section)
    return merged


def segment_text(text: str) -> List[Section]:
    """Split full paper text into labeled sections using headings."""
    lines = text.split("\n")
    sections: List[Section] = []
    current_label: Optional[str] = None
    current_heading = ""
    current_lines: List[str] = []

    def flush():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body or current_label:
            sections.append(
                Section(label=current_label or "other", heading=current_heading, text=body)
            )
        current_lines = []

    # Candidate heading detection: short lines, possibly numbered
    candidate_heading = re.compile(
        r"^\s*(\d+\s*\.?\s*)?([A-Z][A-Za-z &\-]{2,45})\s*$"
    )
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        match = candidate_heading.match(line)
        if match and len(line.split()) <= 6:
            label = _normalize_label(line)
            # Treat as heading only if it looks structural (numbered, or known label)
            if label or match.group(1):
                flush()
                current_label = label
                current_heading = line
                continue
        current_lines.append(raw_line)
    flush()

    if not sections:
        sections.append(Section(label="other", heading="", text=text.strip()))
    # Merge tiny unlabeled leading fragments (usually the paper title) into the
    # first real section instead of polluting the section list.
    if len(sections) >= 2 and sections[0].label == "other" and sections[0].heading == "" and len(sections[0].text.split()) < 20:
        sections[1].text = (sections[0].text + "\n\n" + sections[1].text).strip()
        sections = sections[1:]
    return _merge_short_sections(sections)


def parse_document(filename: str, payload: bytes) -> ParsedDocument:
    """Full parse: validate, extract, segment."""
    text = extract_text(filename, payload)
    sections = segment_text(text)
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    return ParsedDocument(
        filename=filename,
        extension=Path(filename).suffix.lower() or detect_file_type(payload)[0],
        sections=sections,
        full_text=text,
        word_count=len(words),
    )
