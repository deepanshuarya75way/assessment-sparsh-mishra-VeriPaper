# VeriPaper: Autonomous Research Integrity Platform

VeriPaper is a forensic-grade, multi-module verification platform designed to ensure the authenticity and integrity of academic research papers. It serves as a comprehensive, free alternative to commercial tools like Turnitin, providing deep insights into AI generation, plagiarism, and methodological consistency.

[![CI Status](https://github.com/SparshM8/VeriPaper/actions/workflows/ci.yml/badge.svg)](https://github.com/SparshM8/VeriPaper/actions/workflows/ci.yml)
[![Deployment](https://img.shields.io/badge/Deployed%20on-Render-blue)](https://veripaper.onrender.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🚀 Core Capabilities

VeriPaper analyzes documents through six specialized forensic dimensions:

*   **AI Detection**: Identifies synthetic text using a fine-tuned DistilBERT engine, optimized via int8 quantization for high-speed, low-memory inference.
*   **Zero-Cost Plagiarism Attribution**: Detects overlaps with the global web corpus using an intelligent DuckDuckGo-based attribution engine—no expensive API keys required.
*   **Intent-Based Taxonomy**: Automatically distinguishes between "Cited" and "Uncited" similarities, helping researchers identify missing attributions vs. legitimate citations.
*   **Citation Validation**: Cross-references every DOI and reference against the CrossRef global database to detect "hallucinated" or retracted citations.
*   **Statistical Integrity**: Analyzes P-value distributions and statistical patterns to identify potential data manipulation or reporting anomalies.
*   **Professional Reporting**: Generates industry-standard PDF reports with executive summaries, credibility badges, and interactive full-text overlays.

## 🛠 Architecture & Optimization

VeriPaper is built for maximum efficiency on constrained environments:

*   **Engine**: FastAPI + React 18.
*   **ML Stack**: ONNX Runtime + standalone Rust tokenizers (optimized for 512MB RAM).
*   **Monitoring**: Integrated Sentry error tracking and automated health monitoring.
*   **Deployment**: Fully containerized (Docker) and optimized for Render's free tier.

For a deep dive into the system design, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 📥 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   Python 3.11+
*   Node.js 20+

### Quick Start (Docker)
```bash
docker-compose up --build
```
The platform will be available at `http://localhost:8000`.

### Manual Setup
Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed local development instructions.

## 📊 Roadmap

*   [x] **Phase 1**: Core Analysis Infrastructure (AI, Plagiarism, Citations).
*   [x] **Phase 2**: Professional Reporting Engine (PDF, Overlays, Taxonomy).
*   [x] **Phase 3**: PCV Suite (Methodological Fingerprinting & Anomaly Detection).
*   [ ] **Phase 4**: Visual Forensic Analysis (Image manipulation detection).
*   [ ] **Phase 5**: Multi-language support for global research standards.

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

---
**Disclaimer**: VeriPaper is an automated assistant. Final integrity decisions should always be made by qualified human reviewers.
