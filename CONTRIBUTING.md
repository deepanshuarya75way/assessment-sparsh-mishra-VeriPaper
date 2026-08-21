# Contributing to VeriPaper

Thank you for your interest in contributing to VeriPaper! We welcome contributions from the academic and software engineering communities to help make research verification accessible to everyone.

## Code of Conduct

By participating in this project, you agree to maintain a professional and respectful environment. We follow standard academic integrity guidelines.

## How to Contribute

1.  **Report Bugs**: Use GitHub Issues to report any irregularities in analysis or system errors.
2.  **Suggest Features**: We are always looking to expand our analysis modules (e.g., Image Forensic analysis).
3.  **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a feature branch (`feature/your-feature`).
    *   Ensure all backend tests pass (`pytest`).
    *   Provide a clear description of the changes.

## Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd backend/frontend
npm install
npm run dev
```

## Pull Request Guidelines

*   **Commit Messages**: Use descriptive, imperative commit messages (e.g., "Add P-value distribution check").
*   **Documentation**: Update `README.md` or `ARCHITECTURE.md` if your changes alter the core system logic.
*   **Testing**: New features should include corresponding unit or integration tests in the `backend/tests` directory.

## Academic Integrity

VeriPaper is designed to *assist* in research verification. Contributors should ensure that all features prioritize accuracy and transparency to prevent the platform from being misused.
