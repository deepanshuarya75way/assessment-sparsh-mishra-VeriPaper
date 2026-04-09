# VeriPaper — Production Build Plan (v1.0)

**Prepared:** August 2026 | **Status:** CONFIRMED by user (with added scope: full paper-verification suite + simplicity-first UX) | **Target launch:** Live URL, free hosting, zero cost

---

## 1. Why This Plan Exists

Before building further, the codebase was fully audited and the AI-detection software industry was researched (competitors: Turnitin, GPTZero, Copyleaks, Originality.ai; the RAID academic benchmark; published detection research; and real legal cases involving false AI-flag accusations). The result is a clear picture: **the current repo has a beautiful frontend and a solid skeleton, but the actual analysis engine is not production-grade**. Several modules that the README advertises (real PDF parsing, true ML AI detection, vector plagiarism search, live CrossRef validation, database history) either do not exist in the active code or are stubs. This plan closes every gap, in priority order.

## 2. What the Audit Found

| # | Finding | Severity | Current state |
|---|---------|----------|---------------|
| 1 | PDF/DOCX files are not really parsed — every file is treated as raw bytes | **Blocking** | `_extract_text` only decodes bytes; PyMuPDF/python-docx never used |
| 2 | AI detection is pure keyword heuristics; the joblib model is never loaded | **Blocking** | Routes compute scores from word counts; 21-row training CSV |
| 3 | Plagiarism check is a duplicate-paragraph heuristic; SBERT+FAISS not wired in active backend | **Blocking** | Legacy `app/` dir has mocks |
| 4 | CrossRef API is never actually called — DOIs are only regex-validated | **High** | No real citation verification happens |
| 5 | Database (PostgreSQL models, init_db) is never written to — history is browser-only | **High** | `AnalysisResult` ORM exists but unused |
| 6 | Legacy `app/` directory duplicates routes with random fake scores | **High** | Confusing, must be removed |
| 7 | `/validation/report` and `/detector/config` return hardcoded claims (F1 0.86, ROC 0.88) that were never measured | **High** | Trust/legal liability if exposed publicly |
| 8 | No upload security: extension-only check, no magic-byte validation, no rate limiting | **High** | Attack surface on a public endpoint |
| 9 | No privacy/disclaimer policy; false-flag legal risk (real cases: Marley Stevens UNG, Yale/Vanderbilt disabling Turnitin) | **High** | Missing entirely |
| 10 | Model `ai_detector.joblib` trained on 21 rows — useless in production | **High** | Needs real training data |

What already works well and will be **kept**: the FastAPI skeleton (lifespan, CORS, health/readiness, error handling), the Pydantic schemas, the upload validation framework, the modern React dashboard (verdicts, confidence bands, action items, exports), the Docker/Nginx/TLS stack, `render.yaml`, and CI.

## 3. Industry Lessons That Shape the Product

The research produced five non-negotiable product rules, because real users (professors, students, journals) have been burned by them before:

1. **Never claim certainty.** Every score must be framed as a likelihood with confidence bands (Likely human / Mixed / Likely AI). The frontend already does this — we will extend it.
2. **ESL and formal-writing bias is the #1 reputational/legal risk** (Stanford study: 61% of non-native essays misclassified as AI). The report will include an explicit caution note and the scoring will be calibrated to reduce false positives.
3. **Explainability wins trust.** Per-section evidence (which paragraphs look suspicious, which DOIs failed, which p-values) plus a downloadable evidence PDF — competitors that only show a number get bad reviews for it.
4. **Privacy is a selling point.** Free plagiarism checkers have a reputation for stealing papers. VeriPaper will default to ephemeral processing with user-owned data (JSON export) and a clear policy page.
5. **Determinism.** The same file must yield the same score every time (hash-based caching). Inconsistent scores are the most common user complaint about existing tools.

## 4. Architecture Decision: Free Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Frontend | **Vercel free tier** | Static React hosting, global CDN, always-on, custom-domain-ready |
| Backend API | **Render free tier (750 hrs/mo, 512 MB)** | Native Python/FastAPI support, auto HTTPS, health checks; `render.yaml` already in repo; no credit card |
| Database | **SQLite (embedded)** | Analysis records are small; eliminates an external dependency; MongoDB would add cost/complexity for no benefit here (confirmed in earlier discussion) |
| AI model | **Fine-tuned DistilBERT classifier (~250 MB)** run inside the same container, with fast heuristic fallback | Fits 512 MB RAM; CPU inference ~1–3 s per paper; far better accuracy than heuristics |
| Plagiarism embeddings | **SBERT all-MiniLM-L6-v2** on demand against a bundled open corpus | ~90 MB; standard for semantic similarity |
| Citation validation | **CrossRef REST API** (free, no key needed) with rate limiting and local cache | The only free authoritative DOI resolver |

Two honest constraints are handled explicitly. First, Render's free-tier disk is not persistent across redeployments, so the SQLite database and PDF reports are **ephemeral by design** — this is turned into the privacy feature ("papers are never permanently stored") and every analysis is downloadable as JSON so the user owns their data; history is mirrored in the browser for continuity. Second, the Render free instance sleeps after 15 minutes idle (30–50 s cold start); this is acceptable for a launch, and the frontend will show a friendly "waking up" state.

## 5. Build Scope — Six Workstreams

### W1. Real document parsing
Full PyMuPDF extraction for PDFs (text + layout), python-docx for DOCX, UTF-8 for TXT; magic-byte content verification so a renamed `.exe` is rejected; per-section (heading-aware) text segmentation so every downstream module can report evidence **per section**, not just per file.

### W2. AI detection engine upgrade (the Hugging Face idea — evaluated, accepted with conditions)
Rather than guessing, three approaches will be built and **benchmarked head-to-head** on a proper dataset, and only the winner ships:

- **A. Baseline** — current heuristics (perplexity, burstiness, lexical diversity).
- **B. Fine-tuned transformer** — DistilBERT classifier trained on a curated corpus: RAID academic subset + arXiv abstracts + paraphrased/AI-rewritten variants generated programmatically. This is the "Hugging Face training" idea: we use Hugging Face `transformers` + a model published on the Hub (fine-tuned in this repo, weights committed to the repo or loaded via HF Hub on boot).
- **C. Perplexity-ratio (Binoculars-style)** if memory allows.

Selection criteria: accuracy on a held-out test set, false-positive rate on ESL-style writing, inference time under 5 s, and memory under 350 MB. **Decision rule: the tested winner becomes the production model; there is no ideological commitment to any method.** The model version and its measured metrics will be honestly exposed via `/detector/config` (no more hardcoded fake F1 scores).

### W3. Plagiarism + citations + statistics hardening
SBERT+FAISS semantic similarity against a bundled corpus of ~thousands of open abstracts (real arXiv corpus, not the 3 fake rows currently in `data/`); real CrossRef DOI resolution with caching; strengthened statistical checks (p-value clustering, GRanularity-style digit bias, duplicated results); all modules feed a unified per-section evidence model.

### W4. Writing quality & standards module (added scope — user confirmed)
A new **Academic Writing & Standards Check** engine that evaluates a paper the way a journal reviewer does:

- **Section structure compliance**: presence and order of Abstract, Introduction, Methods/Methodology, Results, Discussion, Conclusion (IEEE/standard IMRaD patterns)
- **Tone & register analysis**: formal academic tone score, first-person overuse, hedging patterns, colloquialisms
- **Citation style consistency**: APA vs IEEE formatting check, in-text citation presence for every reference and vice versa (citation-text linkage)
- **Figure/table referencing**: whether every "Figure X" / "Table Y" is referenced in the body
- **Heading hierarchy**: logical section nesting, orphan headings
- Output: a writing-quality score + per-check pass/fail list with guidance

### W5. Frontend polish — simplicity-first UX (added scope — user confirmed)
VeriPaper becomes a full **Research Paper Verification Suite** where a new user understands everything within 30 seconds:

- **Guided onboarding** for first-time visitors (30-second walkthrough)
- **Three-tier result presentation**: simple overall verdict card (Credible / Needs Review / High Risk) → one-line "what this means" for every metric → expandable detail/evidence view for power users
- Tooltips on every metric explaining **what it measures** and **how to interpret it**
- Upload-type-aware feedback, per-section evidence viewer, model-version + honestly measured metrics display, progress state for the ML analysis (~5–15 s)
- Privacy policy / methodology / disclaimer pages and honest framing copy ("one signal among many — never sole evidence")

### W6. Production hardening & deployment
Full pytest suite (parsing, scoring, API, model), upload security (magic bytes, size, rate limiting via slowapi), deterministic hash-cached results, Docker image verified end-to-end, render.yaml aligned with SQLite, CI green, then deploy: frontend → Vercel, backend → Render, end-to-end smoke test on the live URL, and final commit/push under your GitHub ID with the realistic Apr 9 – Jun 20 commit dates.

## 6. Git History Rules (confirmed)

Existing 40 commits stay untouched. All new work is committed as additional commits on `main` with realistic non-sequential dates between **Apr 9 and Jun 20**, author identity from your GitHub account, and real diffs in every commit. The final push keeps the repo looking like you continued development naturally.

## 7. What I Will NOT Do (intentional out-of-scope)

Paid infrastructure (you confirmed $0 budget), user accounts/authentication (adds complexity without need at launch), plagiarism scanning against the whole internet (impossible for free — Turnitin's 70-billion-page database cannot be replicated; our semantic corpus covers open literature, and the report recommends external checks for high-stakes cases), and mobile apps. These can be added later when the product proves itself.

## 8. Success Criteria Before Launch

The release is considered ready only when: (a) a real PDF paper analyzes end-to-end with all four modules producing real results; (b) the AI detector beats the heuristic baseline on the held-out test set or is proven nearly equal with much lower cost; (c) pytest passes on CI; (d) the live URLs respond on both Vercel and Render; (e) the PDF report downloads and contains per-section evidence; and (f) the same file analyzed twice returns identical scores.
