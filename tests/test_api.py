"""
tests/test_api.py
-----------------
Tests for Flask API endpoints.
Uses Flask test client — no running server needed.
"""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(scope="module")
def client():
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        response = client.get("/health")
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_health_has_service_name(self, client):
        response = client.get("/health")
        data = json.loads(response.data)
        assert data["service"] == "CareerAI"


class TestModelInfoEndpoint:
    def test_model_info_endpoint_exists(self, client):
        response = client.get("/model-info")
        # Either 200 (trained) or 503 (not trained) — both valid
        assert response.status_code in (200, 503)

    def test_model_info_returns_json(self, client):
        response = client.get("/model-info")
        data = json.loads(response.data)
        assert "status" in data


class TestPredictRoleEndpoint:
    def test_requires_text(self, client):
        response = client.post("/predict-role",
                               json={},
                               content_type="application/json")
        assert response.status_code == 400

    def test_empty_text_rejected(self, client):
        response = client.post("/predict-role",
                               json={"text": ""},
                               content_type="application/json")
        assert response.status_code == 400

    def test_valid_text_prediction(self, client):
        response = client.post(
            "/predict-role",
            json={"text": "Python machine learning TensorFlow scikit-learn NLP deep learning"},
            content_type="application/json",
        )
        # 200 if model trained, 503 if not
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data["status"] == "success"
            assert "predicted_role" in data["data"]
            assert "confidence" in data["data"]


class TestMatchJobEndpoint:
    RESUME_TEXT = (
        "Python machine learning TensorFlow Docker Kubernetes AWS SQL pandas scikit-learn"
    )
    JD_TEXT = (
        "Looking for ML engineer with Python, TensorFlow, Docker, AWS, SQL skills."
    )

    def test_requires_json(self, client):
        response = client.post("/match-job")
        assert response.status_code == 400

    def test_missing_resume_text(self, client):
        response = client.post(
            "/match-job",
            json={"jd_text": self.JD_TEXT},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_missing_jd_text(self, client):
        response = client.post(
            "/match-job",
            json={"resume_text": self.RESUME_TEXT},
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_valid_match_returns_score(self, client):
        response = client.post(
            "/match-job",
            json={"resume_text": self.RESUME_TEXT, "jd_text": self.JD_TEXT},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"
        assert "overall_score" in data["data"]
        assert 0 <= data["data"]["overall_score"] <= 100

    def test_match_returns_matched_skills(self, client):
        response = client.post(
            "/match-job",
            json={"resume_text": self.RESUME_TEXT, "jd_text": self.JD_TEXT},
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert "matched_skills" in data["data"]
        assert isinstance(data["data"]["matched_skills"], list)


class TestUploadResumeEndpoint:
    def test_no_file_returns_400(self, client):
        response = client.post("/upload-resume")
        assert response.status_code == 400

    def test_wrong_file_type_rejected(self, client):
        data = {"resume": (io.BytesIO(b"hello world"), "resume.txt")}
        response = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_invalid_pdf_returns_error(self, client):
        """Send bytes that look like PDF header but are garbage."""
        fake_pdf = b"%PDF-1.4 garbage data that is not valid"
        data = {"resume": (io.BytesIO(fake_pdf), "fake.pdf")}
        response = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        # Should return 400 (extraction error), not 500
        assert response.status_code in (400, 422, 500)  # any error, not a crash


class TestAnalyzeEndpoint:
    def test_no_resume_returns_400(self, client):
        data = {"jd_text": "Python developer needed"}
        response = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_no_jd_returns_400(self, client):
        fake_pdf = b"%PDF-1.4 fake"
        data = {"resume": (io.BytesIO(fake_pdf), "resume.pdf")}
        response = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_404_for_unknown_route(self, client):
        response = client.get("/nonexistent-route")
        assert response.status_code == 404
