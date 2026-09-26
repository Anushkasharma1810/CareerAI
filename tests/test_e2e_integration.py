"""
tests/test_e2e_integration.py
------------------------------
End-to-end integration tests for CareerAI.

These tests exercise the COMPLETE application flow using the Flask test client:

  PDF bytes -> /upload-resume  -> text extracted, skills returned
  text      -> /predict-role   -> ML model loaded, prediction returned
  texts     -> /match-job      -> matching engine, score returned
  PDF+JD    -> /analyze        -> full pipeline, all sections populated

No running server needed — Flask test client is used throughout.
No hard-coded assertions on specific score values; all checks are
structural / range / monotonicity to prove live computation.

Key E2E assertions:
  - PDF text is actually extracted (not mocked)
  - Resume text reaches the ML model (predict_role called with real text)
  - Trained model returns a known category (not a placeholder)
  - Match score is in [0,100] and differs for different inputs
  - Matched skills come from the intersection of resume & JD skills
  - Missing skills are in JD but not resume
  - Recommendations are non-empty strings driven by skill gaps
  - No hard-coded result: same endpoint returns different values for
    different inputs
"""

import io
import json
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_pdf_bytes(text: str) -> bytes:
    """
    Create a minimal real PDF containing `text` using PyMuPDF.
    Returns the raw bytes of the PDF file.
    """
    import fitz  # PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    # insert_text wraps long lines; split into chunks of 80 chars for readability
    y = 72
    for i in range(0, len(text), 100):
        page.insert_text((50, y), text[i:i+100])
        y += 14
        if y > 700:
            page = doc.new_page()
            y = 72
    tmp = tempfile.mktemp(suffix=".pdf")
    doc.save(tmp)
    doc.close()
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data


# ── realistic test data ───────────────────────────────────────────────────────

ML_RESUME_TEXT = (
    "Jane Doe  jane@example.com  github.com/janedoe\n\n"
    "SUMMARY\n"
    "Senior Machine Learning Engineer with 5 years of experience building and "
    "deploying production ML systems. Passionate about NLP, deep learning, and MLOps.\n\n"
    "SKILLS\n"
    "Python, TensorFlow, PyTorch, scikit-learn, Keras, NLP, deep learning, "
    "computer vision, Docker, Kubernetes, AWS, SQL, pandas, NumPy, Git, CI/CD, "
    "REST API, MLflow, feature engineering, model deployment, MLOps, "
    "machine learning, Hugging Face, transformers, XGBoost\n\n"
    "EXPERIENCE\n"
    "ML Engineer @ TechCorp (2020-2025)\n"
    "- Built NLP pipeline with Hugging Face transformers achieving 94% accuracy.\n"
    "- Deployed TensorFlow models via Docker/Kubernetes on AWS EKS.\n"
    "- Reduced inference latency 40% using model quantization and ONNX.\n"
    "- Implemented CI/CD pipeline for model retraining with MLflow tracking.\n\n"
    "EDUCATION\n"
    "M.S. Computer Science — Stanford University\n"
)

ML_JD_TEXT = (
    "We are looking for a Machine Learning Engineer.\n\n"
    "Required Skills:\n"
    "Python, TensorFlow, scikit-learn, NLP, Docker, AWS, SQL, pandas, deep learning\n\n"
    "Preferred Skills:\n"
    "PyTorch, Kubernetes, computer vision, MLOps, REST API, CI/CD, MLflow\n\n"
    "Responsibilities:\n"
    "- Build and deploy ML models at scale.\n"
    "- Collaborate with data engineers on feature pipelines.\n"
    "- Monitor model performance and trigger retraining.\n"
)

WEBDEV_RESUME_TEXT = (
    "Bob Smith  bob@example.com\n\n"
    "SKILLS\n"
    "React, TypeScript, JavaScript, Node.js, HTML, CSS, PostgreSQL, "
    "Docker, Git, REST API, GraphQL, CI/CD, Webpack, Jest\n\n"
    "EXPERIENCE\n"
    "Frontend Engineer @ WebCo (2019-2025)\n"
    "- Built responsive React SPAs with TypeScript.\n"
    "- Integrated REST API and GraphQL backends.\n"
    "- Automated deployments with CI/CD pipelines.\n"
)

WEBDEV_JD_TEXT = (
    "Full Stack Web Developer\n\n"
    "Required: React, TypeScript, Node.js, PostgreSQL, Docker, Git, CSS, HTML\n"
    "Preferred: Kubernetes, GraphQL, AWS, CI/CD, Jest\n"
)

KNOWN_ROLES = {
    "AI/ML Engineer", "Data Analyst", "DevOps Engineer",
    "Software Engineer", "Web Developer",
}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def ml_resume_pdf():
    return _make_pdf_bytes(ML_RESUME_TEXT)


@pytest.fixture(scope="module")
def webdev_resume_pdf():
    return _make_pdf_bytes(WEBDEV_RESUME_TEXT)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Health + Model Load
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackendHealth:
    def test_health_endpoint_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_is_loaded(self, client):
        """ML model must be loaded at startup — not 'not_trained'."""
        r = client.get("/health")
        d = json.loads(r.data)
        assert d["model_status"] == "loaded", (
            f"Expected model_status='loaded', got '{d['model_status']}'. "
            "Run `py src/training/train.py` to train the model first."
        )

    def test_model_info_returns_real_metrics(self, client):
        """Model info endpoint must return real accuracy/F1, not zeros or None."""
        r = client.get("/model-info")
        assert r.status_code == 200
        d = json.loads(r.data)
        info = d["data"]
        assert info["best_model"] in ("Logistic Regression", "Random Forest")
        # Both models must have non-trivial accuracy (>0.5)
        for model_key, metrics in info["models"].items():
            acc = metrics.get("accuracy", 0)
            assert acc > 0.5, f"{model_key} accuracy={acc} looks wrong"
            f1 = metrics.get("f1_weighted", 0)
            assert f1 > 0.5, f"{model_key} f1_weighted={f1} looks wrong"


# ═══════════════════════════════════════════════════════════════════════════════
# B. PDF Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestPDFExtraction:
    def test_upload_real_pdf_extracts_text(self, client, ml_resume_pdf):
        """A real PDF must return extracted text, not an error."""
        data = {"resume": (io.BytesIO(ml_resume_pdf), "resume.pdf")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["status"] == "success"
        assert d["data"]["char_count"] > 50, "Expected substantial text extracted"

    def test_extracted_text_contains_resume_content(self, client, ml_resume_pdf):
        """Extracted text must contain identifiable resume tokens."""
        data = {"resume": (io.BytesIO(ml_resume_pdf), "resume.pdf")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        preview = d["data"]["text_preview"].lower()
        # At least one key skill should survive extraction + preview
        assert any(kw in preview for kw in ["python", "machine", "learning", "tensorflow", "docker"])

    def test_extraction_method_reported(self, client, ml_resume_pdf):
        """Response must name the extraction library used (PyMuPDF or pdfplumber)."""
        data = {"resume": (io.BytesIO(ml_resume_pdf), "resume.pdf")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        assert d["data"]["extraction_method"] in ("PyMuPDF", "pdfplumber")

    def test_skills_extracted_from_real_pdf(self, client, ml_resume_pdf):
        """Skills list must be non-empty for an ML resume."""
        data = {"resume": (io.BytesIO(ml_resume_pdf), "resume.pdf")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        assert len(d["data"]["skills"]) > 0, "Expected skills extracted from ML resume"

    def test_non_pdf_rejected(self, client):
        data = {"resume": (io.BytesIO(b"hello world"), "resume.txt")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        assert r.status_code == 400

    def test_corrupt_pdf_returns_error_not_crash(self, client):
        data = {"resume": (io.BytesIO(b"%PDF-1.4 garbage bytes"), "fake.pdf")}
        r = client.post("/upload-resume", data=data, content_type="multipart/form-data")
        assert r.status_code in (400, 422, 500)  # any error, not a 200


# ═══════════════════════════════════════════════════════════════════════════════
# C. ML Role Prediction
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLPrediction:
    def test_predict_returns_known_role(self, client):
        """Prediction must return one of the 5 trained categories."""
        r = client.post("/predict-role",
                        json={"text": ML_RESUME_TEXT},
                        content_type="application/json")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["data"]["predicted_role"] in KNOWN_ROLES, (
            f"Predicted role '{d['data']['predicted_role']}' not in known categories"
        )

    def test_ml_resume_predicts_ml_role(self, client):
        """A clear ML resume should predict AI/ML Engineer."""
        r = client.post("/predict-role",
                        json={"text": ML_RESUME_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)
        assert d["data"]["predicted_role"] == "AI/ML Engineer", (
            f"Expected AI/ML Engineer, got {d['data']['predicted_role']} "
            f"(confidence={d['data']['confidence']})"
        )

    def test_webdev_resume_predicts_webdev_role(self, client):
        """A clear web dev resume should predict Web Developer."""
        r = client.post("/predict-role",
                        json={"text": WEBDEV_RESUME_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)
        assert d["data"]["predicted_role"] == "Web Developer", (
            f"Expected Web Developer, got {d['data']['predicted_role']} "
            f"(confidence={d['data']['confidence']})"
        )

    def test_confidence_is_valid_probability(self, client):
        r = client.post("/predict-role",
                        json={"text": ML_RESUME_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)
        conf = d["data"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_all_probabilities_sum_to_one(self, client):
        r = client.post("/predict-role",
                        json={"text": ML_RESUME_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)
        probs = d["data"]["all_probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_different_resumes_give_different_predictions(self, client):
        """Proves predictions are computed from input, not hard-coded."""
        r_ml = client.post("/predict-role",
                           json={"text": ML_RESUME_TEXT},
                           content_type="application/json")
        r_web = client.post("/predict-role",
                            json={"text": WEBDEV_RESUME_TEXT},
                            content_type="application/json")
        ml_role  = json.loads(r_ml.data)["data"]["predicted_role"]
        web_role = json.loads(r_web.data)["data"]["predicted_role"]
        assert ml_role != web_role, (
            "ML resume and web dev resume returned the same prediction "
            "— this suggests hard-coded output."
        )

    def test_prediction_is_deterministic(self, client):
        """Same text must always return same prediction (no randomness)."""
        r1 = client.post("/predict-role", json={"text": ML_RESUME_TEXT},
                         content_type="application/json")
        r2 = client.post("/predict-role", json={"text": ML_RESUME_TEXT},
                         content_type="application/json")
        d1 = json.loads(r1.data)["data"]
        d2 = json.loads(r2.data)["data"]
        assert d1["predicted_role"] == d2["predicted_role"]
        assert d1["confidence"] == d2["confidence"]

    def test_model_disclaimer_present(self, client):
        """Response must include the confidence disclaimer."""
        r = client.post("/predict-role", json={"text": ML_RESUME_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        assert "note" in d
        assert "NOT" in d["note"] or "not" in d["note"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# D. Resume-JD Matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatchingEndpoint:
    def test_match_returns_score_in_range(self, client):
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                        content_type="application/json")
        assert r.status_code == 200
        d = json.loads(r.data)["data"]
        assert 0 <= d["overall_score"] <= 100

    def test_strong_match_scores_above_60(self, client):
        """ML resume vs ML JD — strong match should score > 60."""
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        assert d["overall_score"] >= 60, (
            f"Expected strong match score >=60, got {d['overall_score']}"
        )

    def test_cross_domain_scores_below_strong(self, client):
        """ML resume vs web JD should score lower than ML resume vs ML JD."""
        r_good = client.post("/match-job",
                             json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                             content_type="application/json")
        r_cross = client.post("/match-job",
                              json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                              content_type="application/json")
        good_score  = json.loads(r_good.data)["data"]["overall_score"]
        cross_score = json.loads(r_cross.data)["data"]["overall_score"]
        assert good_score > cross_score, (
            f"Same-domain ({good_score}) should beat cross-domain ({cross_score})"
        )

    def test_matched_skills_non_empty_for_good_match(self, client):
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        assert len(d["matched_skills"]) > 0

    def test_matched_skills_are_real_canonical_names(self, client):
        """Matched skills must be recognisable canonical skill names."""
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        # At least one of these must appear — they're clearly in both texts
        expected_in_matched = {"Python", "TensorFlow", "Docker", "SQL", "pandas", "NLP"}
        actual = set(d["matched_skills"])
        overlap = expected_in_matched & actual
        assert len(overlap) >= 3, (
            f"Expected >=3 canonical skills in matched, got: {actual}"
        )

    def test_missing_skills_are_in_jd_not_resume(self, client):
        """
        Missing skills must come from the JD.
        Test: web dev JD vs ML resume — web-specific skills should appear as missing.
        """
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        # React, TypeScript, Node.js, PostgreSQL, HTML, CSS are in WEBDEV_JD_TEXT
        # but not in ML_RESUME_TEXT — at least some should appear as missing
        web_only = {"React", "TypeScript", "Node.js", "PostgreSQL", "HTML", "CSS"}
        missing = set(d["missing_skills"])
        overlap = web_only & missing
        assert len(overlap) >= 2, (
            f"Expected web-only skills in missing; got missing={missing}"
        )

    def test_score_is_deterministic(self, client):
        r1 = client.post("/match-job",
                         json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                         content_type="application/json")
        r2 = client.post("/match-job",
                         json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                         content_type="application/json")
        d1 = json.loads(r1.data)["data"]
        d2 = json.loads(r2.data)["data"]
        assert d1["overall_score"] == d2["overall_score"]
        assert d1["matched_skills"] == d2["matched_skills"]

    def test_different_jds_give_different_scores(self, client):
        """Proves matching is computed from input — not returning a fixed value."""
        r1 = client.post("/match-job",
                         json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                         content_type="application/json")
        r2 = client.post("/match-job",
                         json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                         content_type="application/json")
        s1 = json.loads(r1.data)["data"]["overall_score"]
        s2 = json.loads(r2.data)["data"]["overall_score"]
        assert s1 != s2, "Same resume vs two very different JDs returned same score — hard-coded?"


# ═══════════════════════════════════════════════════════════════════════════════
# E. Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommendations:
    def test_recommendations_present_for_partial_match(self, client):
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        assert len(d["recommendations"]) > 0

    def test_recommendations_are_strings(self, client):
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        for rec in d["recommendations"]:
            assert isinstance(rec, str) and len(rec) > 5

    def test_recommendations_reference_missing_skills(self, client):
        """
        When there are missing skills, at least one recommendation must
        reference one of those skills by name.
        """
        r = client.post("/match-job",
                        json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                        content_type="application/json")
        d = json.loads(r.data)["data"]
        missing = set(d["missing_skills"])
        recs_text = " ".join(d["recommendations"]).lower()
        # At least one missing skill name should appear in recommendations text
        found = any(skill.lower() in recs_text for skill in missing)
        assert found, (
            f"None of the missing skills {missing} appear in recommendations: "
            f"{d['recommendations']}"
        )

    def test_full_match_recommendation_differs_from_partial(self, client):
        """
        Perfect-match recommendations differ from partial-match ones,
        proving they're generated dynamically.
        """
        r_partial = client.post("/match-job",
                                json={"resume_text": ML_RESUME_TEXT, "jd_text": WEBDEV_JD_TEXT},
                                content_type="application/json")
        r_full = client.post("/match-job",
                             json={"resume_text": ML_RESUME_TEXT, "jd_text": ML_JD_TEXT},
                             content_type="application/json")
        partial_recs = json.loads(r_partial.data)["data"]["recommendations"]
        full_recs    = json.loads(r_full.data)["data"]["recommendations"]
        assert partial_recs != full_recs, (
            "Partial and full matches returned identical recommendations — hard-coded?"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# F. Full /analyze pipeline (PDF + JD -> all sections)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzePipeline:
    def test_full_analyze_returns_all_sections(self, client, ml_resume_pdf):
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["status"] == "success"
        for section in ("resume", "job_description", "role_prediction", "match_analysis"):
            assert section in d, f"Missing section: {section}"

    def test_analyze_pdf_text_extracted(self, client, ml_resume_pdf):
        """PDF text must actually be extracted — char_count > 0."""
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        assert d["resume"]["char_count"] > 50

    def test_analyze_role_prediction_populated(self, client, ml_resume_pdf):
        """role_prediction section must contain a real predicted_role."""
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        pred = d["role_prediction"]
        assert "predicted_role" in pred
        assert pred["predicted_role"] in KNOWN_ROLES

    def test_analyze_match_score_populated(self, client, ml_resume_pdf):
        """match_analysis must contain a valid overall_score."""
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        ma = d["match_analysis"]
        assert "overall_score" in ma
        assert 0 <= ma["overall_score"] <= 100

    def test_analyze_matched_skills_populated(self, client, ml_resume_pdf):
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        ma = d["match_analysis"]
        assert "matched_skills" in ma
        assert len(ma["matched_skills"]) > 0

    def test_analyze_recommendations_populated(self, client, ml_resume_pdf):
        """Recommendations list must be non-empty."""
        data = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        d = json.loads(r.data)
        recs = d["match_analysis"].get("recommendations", [])
        assert len(recs) > 0

    def test_analyze_different_jds_give_different_scores(self, client, ml_resume_pdf, webdev_resume_pdf):
        """
        Same PDF resume submitted against two very different JDs must produce
        different match scores — proves no hard-coded result.
        """
        data_ml = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        data_web = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": WEBDEV_JD_TEXT,
        }
        r_ml  = client.post("/analyze", data=data_ml,  content_type="multipart/form-data")
        r_web = client.post("/analyze", data=data_web, content_type="multipart/form-data")
        s_ml  = json.loads(r_ml.data)["match_analysis"]["overall_score"]
        s_web = json.loads(r_web.data)["match_analysis"]["overall_score"]
        assert s_ml != s_web, (
            "Same PDF against two different JDs returned identical score — "
            "this suggests hard-coded output."
        )

    def test_analyze_different_resumes_give_different_roles(self, client, ml_resume_pdf, webdev_resume_pdf):
        """
        Two different PDF resumes must predict different job roles — proves
        ML model is invoked with actual resume text.
        """
        data_ml = {
            "resume": (io.BytesIO(ml_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        data_web = {
            "resume": (io.BytesIO(webdev_resume_pdf), "resume.pdf"),
            "jd_text": ML_JD_TEXT,
        }
        r_ml  = client.post("/analyze", data=data_ml,  content_type="multipart/form-data")
        r_web = client.post("/analyze", data=data_web, content_type="multipart/form-data")
        role_ml  = json.loads(r_ml.data)["role_prediction"]["predicted_role"]
        role_web = json.loads(r_web.data)["role_prediction"]["predicted_role"]
        assert role_ml != role_web, (
            f"ML resume ({role_ml}) and web dev resume ({role_web}) got the "
            "same predicted role — this suggests hard-coded output."
        )

    def test_analyze_no_resume_returns_400(self, client):
        data = {"jd_text": ML_JD_TEXT}
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert r.status_code == 400

    def test_analyze_no_jd_returns_400(self, client):
        data = {"resume": (io.BytesIO(b"%PDF-1.4 stub"), "r.pdf")}
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
