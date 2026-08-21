# VeriPaper System Architecture

VeriPaper is an autonomous research integrity platform designed to provide a free, high-performance alternative to commercial plagiarism and AI detection tools. The architecture is optimized for low-memory environments (Render Free Tier) while maintaining forensic-grade analysis capabilities.

## 1. Core Modules

VeriPaper operates through six specialized analysis modules that work in parallel to evaluate document credibility:

| Module | Methodology | Key Technologies |
| :--- | :--- | :--- |
| **AI Detection** | Transformer-based stylometry & Logistic Regression | ONNX Runtime, DistilBERT, Scikit-learn |
| **Plagiarism** | Multi-source similarity matching | TF-IDF, DuckDuckGo Zero-Cost Attribution |
| **Citation Validity** | Metadata cross-referencing | CrossRef API, PyMuPDF |
| **Statistical Integrity** | Distribution forensics & P-value analysis | SciPy, NumPy |
| **Writing Quality** | IEEE/IMRaD standard adherence | Custom Linguistic Rules |
| **PCV Suite** | Citation-graph anomaly & Methodological fingerprinting | NetworkX, Custom Forensic Logic |

## 2. Technical Stack

*   **Backend**: FastAPI (Python 3.11) with Uvicorn.
*   **Frontend**: React 18, Vite, Tailwind CSS.
*   **Database**: SQLAlchemy with SQLite (Production) / PostgreSQL (Optional).
*   **ML Runtime**: ONNX Runtime (CPU) for memory-efficient transformer inference.
*   **Reporting**: ReportLab for professional PDF generation.
*   **Monitoring**: Sentry for real-time error tracking and performance monitoring.

## 3. Data Flow

1.  **Ingestion**: User uploads a research paper (PDF/DOCX) via the React frontend.
2.  **Preprocessing**: The backend extracts text and metadata using PyMuPDF and python-docx.
3.  **Analysis**: The text is passed to the six core modules. AI detection uses a quantized ONNX model to stay within the 512MB RAM limit.
4.  **Taxonomy**: Plagiarism matches are categorized into "Cited" and "Uncited" based on the document's bibliography.
5.  **Aggregation**: A final "Credibility Score" is calculated based on weighted inputs from all modules.
6.  **Delivery**: Results are presented in an interactive dashboard with colored full-text overlays and a professional PDF report.

## 4. Memory Optimization Strategies

To operate on the Render Free Tier, VeriPaper employs several emergency memory strategies:
*   **Lazy Loading**: The AI model is only loaded into RAM during inference.
*   **Aggressive GC**: Explicit garbage collection is triggered after every heavy inference and PDF generation task.
*   **Task Eviction**: In-memory async tasks are evicted after 10 minutes to prevent memory leaks.
*   **Standalone Tokenizers**: Uses the Rust-based `tokenizers` library instead of the full `transformers` package to save ~300MB of RAM.

## 5. Security & Privacy

*   **Local Processing**: Documents are processed in a transient environment and are not stored for model training.
*   **Encrypted Transport**: All API communication is secured via TLS.
*   **Error Masking**: Production errors are logged to Sentry without exposing sensitive user data.
