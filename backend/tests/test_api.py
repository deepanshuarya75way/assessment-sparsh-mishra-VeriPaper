from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "version" in payload


def test_readiness_endpoint() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "checks" in payload


def test_analyze_text_file_success() -> None:
    paper = ("In this paper, we propose a robust method for evaluating document "
             "quality through multi-factor analysis. Our approach combines semantic "
             "embeddings with statistical feature extraction to produce a credibility "
             "score for research papers. We evaluate the proposed framework on a "
             "diverse set of academic documents and demonstrate consistent "
             "performance across multiple domains. The results indicate that "
             "combining textual and structural features improves detection "
             "accuracy significantly. We report an overall accuracy of 85 percent "
             "on held-out evaluation data.\n\nReference: DOI 10.1000/xyz123\n" "p < 0.04")
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.txt", paper, "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.txt"
    assert 0 <= payload["overall_research_credibility"] <= 100
    assert payload["report_path"].startswith("/files/")


def test_analyze_rejects_unsupported_extension() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.exe", b"abc", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_analyze_rejects_empty_file() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
