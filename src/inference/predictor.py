"""
predictor.py
------------
Loads the saved ML model artifacts once at startup and provides real-time
inference without retraining.

Usage:
    predictor = RolePredictor()
    result = predictor.predict("Python machine learning TensorFlow data science...")
    print(result)
    # {'predicted_role': 'AI/ML Engineer', 'confidence': 0.87, 'all_probabilities': {...}}
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import preprocess_for_model

MODELS_DIR = ROOT / "models"


class ModelNotTrainedError(Exception):
    """Raised when model artifacts are not found on disk."""


class RolePredictor:
    """
    Singleton-style predictor that loads artifacts once.
    Thread-safe for read-only inference.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        """Load all model artifacts from disk. Call once at app startup."""
        if self._loaded:
            return self

        required = [
            "best_model.joblib",
            "tfidf_vectorizer.joblib",
            "label_encoder.joblib",
            "training_metadata.json",
        ]
        missing = [f for f in required if not (MODELS_DIR / f).exists()]
        if missing:
            raise ModelNotTrainedError(
                f"Model artifacts not found: {missing}. "
                "Run `python src/training/train.py` first."
            )

        self.model = joblib.load(MODELS_DIR / "best_model.joblib")
        self.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
        self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

        with open(MODELS_DIR / "training_metadata.json") as f:
            self.metadata = json.load(f)

        self.label_names = self.label_encoder.classes_.tolist()
        self._loaded = True
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        """
        Predict job role from raw text.

        Parameters
        ----------
        text : str  Raw resume or profile text (will be preprocessed internally).

        Returns
        -------
        dict with keys:
            predicted_role  – str
            confidence      – float (0-1)  model softmax probability of top class
            all_probabilities – {role: probability}
            model_name      – str
            note            – disclaimer about confidence vs employment probability
        """
        if not self._loaded:
            self.load()

        if not text or not text.strip():
            return {
                "predicted_role": "Unknown",
                "confidence": 0.0,
                "all_probabilities": {},
                "error": "Empty text provided",
            }

        # Preprocess
        processed = preprocess_for_model(text)

        # Vectorize
        X = self.vectorizer.transform([processed])

        # Predict
        label_idx = self.model.predict(X)[0]
        predicted_role = self.label_encoder.inverse_transform([label_idx])[0]

        # Probabilities (if available)
        all_probs = {}
        confidence = None

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
            all_probs = {
                self.label_names[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            }
        else:
            # decision_function fallback for SVMs etc.
            confidence = 1.0  # no calibrated probability available
            all_probs = {predicted_role: 1.0}

        return {
            "predicted_role": predicted_role,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "model_name": self.metadata.get("best_model", "Unknown"),
            "note": (
                "Model confidence is the classifier's probability for this prediction. "
                "It does NOT represent employment probability."
            ),
        }

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model."""
        if not self._loaded:
            self.load()
        return {
            "best_model": self.metadata.get("best_model"),
            "version": self.metadata.get("version"),
            "selection_criterion": self.metadata.get(
                "selection_criterion",
                "mean CV F1 (weighted) on training set"
            ),
            "selection_reason": self.metadata.get("selection_reason", ""),
            "cv_folds": self.metadata.get("cv_folds", 5),
            "label_classes": self.label_names,
            "tfidf_features": self.metadata.get("tfidf_config", {}).get("max_features"),
            "dataset": self.metadata.get("dataset", {}),
            "models": {
                name: {
                    # CV metrics (used for selection)
                    "cv_f1_mean":         info["metrics"].get("cv_f1_mean"),
                    "cv_f1_std":          info["metrics"].get("cv_f1_std"),
                    "cv_fold_scores":     info["metrics"].get("cv_fold_scores"),
                    # Test-set metrics (reported after selection)
                    "accuracy":           info["metrics"]["accuracy"],
                    "f1_weighted":        info["metrics"]["f1_weighted"],
                    "precision_weighted": info["metrics"]["precision_weighted"],
                    "recall_weighted":    info["metrics"]["recall_weighted"],
                    "confusion_matrix":   info["metrics"].get("confusion_matrix"),
                    "classification_report": info["metrics"].get("classification_report"),
                }
                for name, info in self.metadata.get("models", {}).items()
            },
        }


# Module-level singleton
_predictor = RolePredictor()


def get_predictor() -> RolePredictor:
    """Get the loaded predictor singleton."""
    if not _predictor.is_loaded:
        _predictor.load()
    return _predictor


# ── standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing RolePredictor...")
    predictor = get_predictor()

    test_cases = [
        (
            "Python machine learning deep learning TensorFlow PyTorch NLP transformers "
            "computer vision scikit-learn model deployment MLOps feature engineering",
            "AI/ML Engineer",
        ),
        (
            "React JavaScript TypeScript HTML CSS Node.js REST API frontend web development "
            "responsive design Webpack Git Figma",
            "Web Developer",
        ),
        (
            "SQL data analysis Tableau Power BI Excel business intelligence "
            "KPI dashboard reporting stakeholders pivot tables",
            "Data Analyst",
        ),
        (
            "Docker Kubernetes Terraform AWS CI/CD Jenkins infrastructure as code "
            "monitoring Prometheus Grafana SRE Ansible",
            "DevOps Engineer",
        ),
        (
            "Java Spring Boot microservices REST API PostgreSQL Git Agile "
            "unit testing JUnit Maven software design patterns",
            "Software Engineer",
        ),
    ]

    print(f"\n{'Text snippet':<55} {'Expected':<25} {'Predicted':<25} {'Conf':>6}")
    print("-" * 115)
    correct = 0
    for text, expected in test_cases:
        result = predictor.predict(text)
        predicted = result["predicted_role"]
        conf = result["confidence"]
        match = "✓" if predicted == expected else "✗"
        snippet = text[:52] + "..."
        print(f"  {snippet:<55} {expected:<25} {predicted:<25} {conf:>6.2%}  {match}")
        if predicted == expected:
            correct += 1

    print(f"\n  Result: {correct}/{len(test_cases)} correct on quick smoke test")
    print("\n  Model info:")
    info = predictor.get_model_info()
    print(f"    Best model : {info['best_model']}")
    print(f"    Classes    : {info['label_classes']}")
    for mname, mmetrics in info["models"].items():
        print(f"    {mname}: Acc={mmetrics['accuracy']:.4f}  F1={mmetrics['f1_weighted']:.4f}")
