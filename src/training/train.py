"""
train.py
--------
Full training pipeline:
  Dataset → Preprocessing → TF-IDF → CV-based model selection → Final evaluation

Model selection methodology (correct procedure):
  1. Stratified 80/20 train/test split — test set is locked away.
  2. For each candidate model, run Stratified 5-Fold CV on the TRAINING SET only.
  3. Select the model with the highest mean CV F1 (weighted).
     The test set plays NO role in this decision.
  4. Re-fit the selected model on the full training set.
  5. Evaluate ONCE on the held-out test set — this is the reported final metric.

Run:
    python src/training/train.py

Outputs (saved to models/):
    tfidf_vectorizer.joblib      – fitted TF-IDF vectorizer (train set only)
    label_encoder.joblib         – fitted LabelEncoder
    logistic_regression.joblib   – LR fitted on full training set
    random_forest.joblib         – RF fitted on full training set
    best_model.joblib            – winner (selected by CV, not test-set)
    training_metadata.json       – metrics, selection rationale, dataset info
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.dataset_builder import build_dataset, ATLAS_CSV
from src.preprocessing.text_cleaner import preprocess_for_model

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

# ── hyperparameters ───────────────────────────────────────────────────────────
TFIDF_CONFIG = {
    "max_features": 8000,
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.95,
    "sublinear_tf": True,
}

LR_CONFIG = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}

RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

TEST_SIZE = 0.20
CV_FOLDS = 5
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_data() -> tuple[list[str], list[str]]:
    """Load dataset and apply text preprocessing."""
    print("\n[1/7] Loading and preprocessing dataset...")
    df = build_dataset()

    print("  Applying text preprocessing (this may take a moment)...")
    df["processed_text"] = df["text"].apply(preprocess_for_model)

    mask = df["processed_text"].str.strip().str.len() > 5
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  Dropped {dropped} rows with empty text after preprocessing")
    df = df[mask].reset_index(drop=True)

    texts = df["processed_text"].tolist()
    labels = df["label"].tolist()
    print(f"  Ready: {len(texts)} samples across {len(set(labels))} classes")
    return texts, labels


def build_features(
    X_train: list[str], X_test: list[str]
) -> tuple[object, object, object]:
    """
    Fit TF-IDF on training data only, transform both splits.
    The vectorizer is never exposed to test-set content.
    """
    print("\n[2/7] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"  Vocabulary (train-fit only): {X_train_tfidf.shape[1]} features")
    return vectorizer, X_train_tfidf, X_test_tfidf


def cv_score_model(
    name: str,
    model,
    X_train_tfidf,
    y_train: np.ndarray,
) -> dict:
    """
    Run Stratified K-Fold CV on the training set ONLY.
    Test set is never touched here.
    Returns CV mean and std — used for model selection.
    """
    print(f"\n  CV evaluation: {name}...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.time()
    cv_scores = cross_val_score(
        model, X_train_tfidf, y_train,
        cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    elapsed = time.time() - t0

    print(f"    Fold scores  : {[round(s, 4) for s in cv_scores]}")
    print(f"    CV F1 mean   : {cv_scores.mean():.4f}")
    print(f"    CV F1 std    : {cv_scores.std():.4f}")
    print(f"    CV time      : {elapsed:.2f}s")

    return {
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std":  round(float(cv_scores.std()),  4),
        "cv_fold_scores": [round(float(s), 4) for s in cv_scores],
        "cv_time_seconds": round(elapsed, 2),
    }


def select_best_model_by_cv(
    lr_cv: dict, rf_cv: dict
) -> tuple[str, str]:
    """
    Select the model with the higher mean CV F1 on the training set.
    The test set is NOT consulted here.

    Returns (winner_name, reason_string).
    """
    print(f"\n[4/7] Model selection by CV F1 (training set only)...")
    lr_mean = lr_cv["cv_f1_mean"]
    rf_mean = rf_cv["cv_f1_mean"]

    if lr_mean >= rf_mean:
        winner = "Logistic Regression"
        reason = (
            f"Logistic Regression CV F1 ({lr_mean:.4f}) >= "
            f"Random Forest CV F1 ({rf_mean:.4f})"
        )
    else:
        winner = "Random Forest"
        reason = (
            f"Random Forest CV F1 ({rf_mean:.4f}) > "
            f"Logistic Regression CV F1 ({lr_mean:.4f})"
        )

    print(f"  LR  CV F1 mean: {lr_mean:.4f} ± {lr_cv['cv_f1_std']:.4f}")
    print(f"  RF  CV F1 mean: {rf_mean:.4f} ± {rf_cv['cv_f1_std']:.4f}")
    print(f"  Selected       : {winner}")
    print(f"  Reason         : {reason}")
    print(f"  [NOTE] Test set was NOT used in this decision.")
    return winner, reason


def fit_and_evaluate_final(
    name: str,
    model,
    X_train_tfidf,
    X_test_tfidf,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> tuple[object, dict]:
    """
    Fit the selected model on the full training set, then evaluate ONCE
    on the held-out test set.  This is the only place the test set is used.
    """
    print(f"\n[5/7] Fitting {name} on full training set...")
    t0 = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    print(f"  Evaluating on held-out test set (first and only time)...")
    y_pred = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=label_names,
        output_dict=True, zero_division=0
    )

    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test F1 (wtd) : {f1:.4f}")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall   : {rec:.4f}")

    test_metrics = {
        "accuracy":             round(float(acc),  4),
        "precision_weighted":   round(float(prec), 4),
        "recall_weighted":      round(float(rec),  4),
        "f1_weighted":          round(float(f1),   4),
        "confusion_matrix":     cm,
        "classification_report": report,
        "train_time_seconds":   round(train_time, 2),
    }
    return model, test_metrics


def fit_secondary_model(
    name: str,
    model,
    X_train_tfidf,
    X_test_tfidf,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
) -> tuple[object, dict]:
    """
    Fit and evaluate the non-selected model (for the comparison table).
    Both models are shown in the evaluation dashboard.
    """
    print(f"\n  Fitting secondary model ({name}) for comparison table...")
    t0 = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test_tfidf)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=label_names,
        output_dict=True, zero_division=0
    )
    print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}")
    return model, {
        "accuracy":             round(float(acc),  4),
        "precision_weighted":   round(float(prec), 4),
        "recall_weighted":      round(float(rec),  4),
        "f1_weighted":          round(float(f1),   4),
        "confusion_matrix":     cm,
        "classification_report": report,
        "train_time_seconds":   round(train_time, 2),
    }


def save_artifacts(
    vectorizer,
    label_encoder,
    lr_model,
    rf_model,
    best_name: str,
    best_model,
    lr_cv: dict,
    rf_cv: dict,
    lr_test: dict,
    rf_test: dict,
    selection_reason: str,
    label_names: list[str],
    dataset_info: dict,
) -> dict:
    """Serialize all model artifacts and training metadata to disk."""
    print("\n[7/7] Saving artifacts...")

    joblib.dump(vectorizer,  MODELS_DIR / "tfidf_vectorizer.joblib")
    print("  Saved tfidf_vectorizer.joblib")

    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print("  Saved label_encoder.joblib")

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    print("  Saved logistic_regression.joblib")

    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib")
    print("  Saved random_forest.joblib")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"  Saved best_model.joblib ({best_name})")

    # Merge CV + test metrics for each model
    def merge_metrics(cv: dict, test: dict, name: str) -> dict:
        return {
            "name": name,
            # CV metrics (used for selection)
            "cv_f1_mean":      cv["cv_f1_mean"],
            "cv_f1_std":       cv["cv_f1_std"],
            "cv_fold_scores":  cv["cv_fold_scores"],
            # Test-set metrics (reported after selection)
            "accuracy":             test["accuracy"],
            "precision_weighted":   test["precision_weighted"],
            "recall_weighted":      test["recall_weighted"],
            "f1_weighted":          test["f1_weighted"],
            "confusion_matrix":     test["confusion_matrix"],
            "classification_report": test["classification_report"],
            "train_time_seconds":   test["train_time_seconds"],
        }

    metadata = {
        "version": "1.1.0",
        "best_model": best_name,
        "selection_criterion": "mean CV F1 (weighted) on training set — test set not used for selection",
        "selection_reason": selection_reason,
        "label_classes": label_names,
        "tfidf_config": TFIDF_CONFIG,
        "cv_folds": CV_FOLDS,
        "models": {
            "logistic_regression": {
                "config": LR_CONFIG,
                "metrics": merge_metrics(lr_cv, lr_test, "Logistic Regression"),
            },
            "random_forest": {
                "config": RF_CONFIG,
                "metrics": merge_metrics(rf_cv, rf_test, "Random Forest"),
            },
        },
        "dataset": dataset_info,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }

    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved training_metadata.json")

    eval_path = EVAL_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved evaluation/evaluation_results.json")

    return metadata


def run_training_pipeline():
    """
    End-to-end training pipeline with correct model-selection methodology:

      Data split → TF-IDF (train only) → CV on train → select by CV →
      fit winner on full train → evaluate ONCE on test
    """
    print("=" * 60)
    print("  CareerAI — ML Training Pipeline  v1.1")
    print("=" * 60)
    print("  Selection method: CV F1 on training set (test set locked)")

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    texts, labels = load_and_preprocess_data()

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    label_names = list(label_encoder.classes_)
    print(f"\n  Classes: {label_names}")

    # ── 3. Stratified train / test split ─────────────────────────────────────
    #       Test set is set aside and NOT touched until step 6.
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"\n[3/7] Split: {len(X_train)} train / {len(X_test)} test")
    print("       Test set is now locked — will NOT be used until final evaluation.")

    # ── 4. TF-IDF — fit on training set only ─────────────────────────────────
    vectorizer, X_train_tfidf, X_test_tfidf = build_features(X_train, X_test)

    # ── 5. CV-based model selection (training set only) ───────────────────────
    print("\n[3b/7] Cross-validation on training set for model selection...")
    print(f"       Strategy: Stratified {CV_FOLDS}-Fold CV, scoring=f1_weighted")

    lr_model = LogisticRegression(**LR_CONFIG)
    rf_model = RandomForestClassifier(**RF_CONFIG)

    lr_cv = cv_score_model("Logistic Regression", lr_model, X_train_tfidf, y_train)
    rf_cv = cv_score_model("Random Forest",        rf_model, X_train_tfidf, y_train)

    # ── 6. Select winner by CV F1 — test set NOT consulted ───────────────────
    best_name, selection_reason = select_best_model_by_cv(lr_cv, rf_cv)

    # ── 7. Fit winner on full training set, evaluate ONCE on test ─────────────
    #       This is the only point the test set is touched.
    if best_name == "Logistic Regression":
        best_model, best_test = fit_and_evaluate_final(
            "Logistic Regression", lr_model,
            X_train_tfidf, X_test_tfidf, y_train, y_test, label_names
        )
        # Fit secondary for comparison table
        _, secondary_test = fit_secondary_model(
            "Random Forest", rf_model,
            X_train_tfidf, X_test_tfidf, y_train, y_test, label_names
        )
        lr_test = best_test
        rf_test = secondary_test
    else:
        best_model, best_test = fit_and_evaluate_final(
            "Random Forest", rf_model,
            X_train_tfidf, X_test_tfidf, y_train, y_test, label_names
        )
        _, secondary_test = fit_secondary_model(
            "Logistic Regression", lr_model,
            X_train_tfidf, X_test_tfidf, y_train, y_test, label_names
        )
        rf_test = best_test
        lr_test = secondary_test

    # ── 8. Dataset provenance ─────────────────────────────────────────────────
    if ATLAS_CSV.exists():
        ds_source = "ahmedheakl/resume-atlas (HuggingFace, Apache 2.0)"
        ds_note   = "Real resume texts. Non-tech categories excluded. Balanced to 500/class."
    else:
        ds_source = "synthetic (fallback)"
        ds_note   = "Synthetic data — metrics do not represent real-world performance."

    dataset_info = {
        "total_samples":  len(texts),
        "train_samples":  len(X_train),
        "test_samples":   len(X_test),
        "num_classes":    len(label_names),
        "classes":        label_names,
        "source":         ds_source,
        "note":           ds_note,
    }

    # ── 9. Save everything ────────────────────────────────────────────────────
    metadata = save_artifacts(
        vectorizer, label_encoder,
        lr_model, rf_model,
        best_name, best_model,
        lr_cv, rf_cv,
        lr_test, rf_test,
        selection_reason,
        label_names, dataset_info,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Selection method : CV F1 on training set only")
    print(f"  Best model       : {best_name}")
    print(f"  Selection reason : {selection_reason}")
    print()
    print("  CV results (training set — used for selection):")
    print(f"    LR  CV F1: {lr_cv['cv_f1_mean']:.4f} ± {lr_cv['cv_f1_std']:.4f}")
    print(f"    RF  CV F1: {rf_cv['cv_f1_mean']:.4f} ± {rf_cv['cv_f1_std']:.4f}")
    print()
    print("  Final test-set metrics (held-out, evaluated ONCE after selection):")
    best_test_m = lr_test if best_name == "Logistic Regression" else rf_test
    print(f"    Accuracy  : {best_test_m['accuracy']:.4f}")
    print(f"    Precision : {best_test_m['precision_weighted']:.4f}")
    print(f"    Recall    : {best_test_m['recall_weighted']:.4f}")
    print(f"    F1        : {best_test_m['f1_weighted']:.4f}")
    print()
    print(f"  Artifacts saved to: {MODELS_DIR}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
