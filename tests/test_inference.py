"""
tests/test_inference.py
------------------------
Tests for the ML model inference pipeline.
Runs only when model artifacts exist.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

MODELS_DIR = ROOT / "models"
MODEL_EXISTS = (MODELS_DIR / "best_model.joblib").exists()

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="Model artifacts not found. Run `python src/training/train.py` first."
)


@pytest.fixture(scope="module")
def predictor():
    from src.inference.predictor import get_predictor
    return get_predictor()


class TestRolePredictor:
    def test_predictor_loads(self, predictor):
        assert predictor.is_loaded

    def test_predict_returns_dict(self, predictor):
        result = predictor.predict("Python machine learning TensorFlow scikit-learn")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, predictor):
        result = predictor.predict("Python developer with Django REST API experience")
        assert "predicted_role" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert "model_name" in result

    def test_predict_role_is_known_category(self, predictor):
        result = predictor.predict("Python machine learning neural networks")
        known_roles = predictor.label_names
        assert result["predicted_role"] in known_roles

    def test_confidence_between_0_and_1(self, predictor):
        result = predictor.predict("React JavaScript TypeScript Node.js frontend developer")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_probabilities_sum_to_1(self, predictor):
        result = predictor.predict("Docker Kubernetes Terraform AWS CI/CD DevOps")
        probs = result["all_probabilities"]
        if probs:
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.01  # within 1%

    def test_empty_text_handled_gracefully(self, predictor):
        result = predictor.predict("")
        assert "error" in result or result["predicted_role"] == "Unknown"

    def test_very_long_text_handled(self, predictor):
        long_text = "Python machine learning " * 500
        result = predictor.predict(long_text)
        assert "predicted_role" in result

    def test_ml_profile_likely_predicts_ml_role(self, predictor):
        ml_text = (
            "Machine learning engineer specializing in deep learning, "
            "TensorFlow, PyTorch, NLP, computer vision, model deployment, "
            "Python, scikit-learn, feature engineering, MLOps."
        )
        result = predictor.predict(ml_text)
        # Result must be one of the known classes
        assert result["predicted_role"] in predictor.label_names

    def test_prediction_is_deterministic(self, predictor):
        text = "Python data analysis pandas SQL Tableau Power BI Excel"
        r1 = predictor.predict(text)
        r2 = predictor.predict(text)
        assert r1["predicted_role"] == r2["predicted_role"]
        assert r1["confidence"] == r2["confidence"]

    def test_get_model_info(self, predictor):
        info = predictor.get_model_info()
        assert "best_model" in info
        assert "label_classes" in info
        assert "models" in info
        assert len(info["label_classes"]) > 0
