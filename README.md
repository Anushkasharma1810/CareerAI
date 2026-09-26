# CareerAI — Intelligent Resume & Job Intelligence System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![React](https://img.shields.io/badge/React-18-blue)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-green)](https://flask.palletsprojects.com)
[![No OpenAI](https://img.shields.io/badge/OpenAI-NOT%20USED-red)](.)

---

## Problem Statement

Job seekers struggle to understand how well their resume matches a specific job description.
They don't know which skills are missing, how to improve their profile, or what job category they fall into.

## Motivation

CareerAI provides objective, explainable resume analysis powered by a real trained ML model — no paid API keys, no hallucinated results.

---

## Features

- **PDF Resume Upload** — drag-and-drop, up to 10 MB
- **Job Description Input** — paste text or upload PDF/TXT
- **ML Role Prediction** — custom TF-IDF + Logistic Regression model predicts your job category
- **Match Score** — weighted combination of skill overlap, TF-IDF cosine similarity, and keyword density
- **Skill Gap Analysis** — matched skills, missing skills, all resume skills
- **Actionable Recommendations** — driven by detected skill gaps, not hard-coded
- **Model Evaluation Dashboard** — real accuracy, F1, confusion matrix, model comparison
- **No OpenAI/Gemini** — fully offline, runs on local hardware

---

## Architecture

```
User → React Frontend
          │
          ↓ HTTP (multipart/form-data)
       Flask Backend (/analyze)
          │
          ├── PDF Extractor (PyMuPDF → pdfplumber fallback)
          ├── JD Processor (skill extraction + title detection)
          ├── Role Predictor (TF-IDF → Logistic Regression)
          └── Match Engine (skill overlap + cosine sim + keyword density)
          │
          ↓ JSON response
       React Frontend → Results Dashboard
```

---

## Dataset

**Primary (active):** resume-atlas (HuggingFace, Apache 2.0)
- Source: https://huggingface.co/datasets/ahmedheakl/resume-atlas
- License: Apache 2.0 (free for research and commercial use)
- Raw size: 13,389 resume texts across 43 categories
- After mapping to tech roles + dedup: **4,295 samples** across 5 categories
- Training uses 500 samples/class (2,500 total) for balanced training
- File: `data/raw/updated_resume_dataset.csv` (auto-downloaded by `_build_real_dataset.py`)

**Fallback:** Synthetic dataset — used only if real CSV is absent. Clearly labelled. Metrics on synthetic data must NOT be cited as real-world performance.

---

## Data Preprocessing

1. Unicode normalization (NFKD → ASCII)
2. Lowercase
3. Remove URLs, emails, phone numbers, bullet characters
4. Remove punctuation (keep hyphens for "full-stack")
5. Remove isolated numbers
6. Tokenize with NLTK `word_tokenize`
7. Lemmatize with `WordNetLemmatizer`
8. Remove English stop words (preserve "not", "no", "nor")

---

## ML Algorithm

**TF-IDF Vectorizer**
- 8,000 features, unigrams + bigrams
- Sublinear TF scaling
- Min DF = 1, Max DF = 95%

**Models compared:**
1. Logistic Regression (lbfgs solver, C=1.0, max_iter=1000)
2. Random Forest (200 trees, n_jobs=-1)

**Selection criterion:** Mean CV F1 (weighted) on **training set only** — test set is not used for selection.

---

## Training Process

```bash
python src/training/train.py
```

Correct model-selection pipeline (`src/training/train.py`):
1. Load and preprocess dataset
2. Encode labels with `LabelEncoder`
3. Stratified 80/20 train/test split — **test set locked away**
4. Fit TF-IDF vectorizer on **training set only**
5. Run Stratified 5-Fold CV on training set for both models
6. **Select winner by mean CV F1 — test set NOT consulted**
7. Fit winner on full training set
8. Evaluate **once** on held-out test set — this is the final reported metric
9. Fit secondary model for comparison table only
10. Serialize all artifacts to `models/`

---

## Evaluation Methodology

- **Train/test split:** Stratified 80/20, `random_state=42`
- **Model selection:** Stratified 5-Fold CV on training set only (`scoring=f1_weighted`)
- **Test isolation:** Test set is touched exactly once — after model selection is complete
- **Reported metrics:** Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted) on held-out test set
- **Confusion matrix:** Full matrix for selected model on test set
- **No test-set leakage:** Selection criterion uses only CV scores from training data

---

## Matching Methodology

```
Match Score = 50% × Skill Overlap
            + 35% × TF-IDF Cosine Similarity
            + 15% × Keyword Density
```

- **Skill Overlap:** `|matched_skills| / |jd_skills|`
- **Cosine Similarity:** TF-IDF bi-gram vectors of preprocessed texts
- **Keyword Density:** Fraction of top-50 JD TF-IDF terms found in resume
- Score is **NOT** hiring probability

---

## Backend Architecture

| File | Purpose |
|------|---------|
| `backend/app.py` | Flask app factory, CORS, startup model loading |
| `backend/routes/api.py` | All REST endpoints |
| `backend/services/analyzer.py` | Orchestrates full pipeline |
| `backend/services/pdf_extractor.py` | PyMuPDF + pdfplumber extraction |
| `backend/services/jd_processor.py` | JD parsing, required/preferred split |
| `src/inference/predictor.py` | Singleton model loader + inference |
| `src/matching/matcher.py` | Match score computation |
| `src/skills/skill_extractor.py` | Regex-based skill extraction |
| `src/skills/skill_vocabulary.py` | 150+ skill canonical + aliases |
| `src/preprocessing/text_cleaner.py` | Text cleaning pipeline |
| `src/training/train.py` | Full training pipeline |
| `src/preprocessing/dataset_builder.py` | Dataset builder |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/model-info` | Model metadata + evaluation metrics |
| POST | `/upload-resume` | Extract text from PDF resume |
| POST | `/predict-role` | Predict job role from text |
| POST | `/match-job` | Compute resume-JD match (JSON body) |
| POST | `/analyze` | Full pipeline (multipart/form-data) |

---

## Project Structure

```
CareerAI/
├── backend/
│   ├── app.py
│   ├── routes/api.py
│   └── services/
│       ├── analyzer.py
│       ├── pdf_extractor.py
│       └── jd_processor.py
├── frontend/src/
│   ├── App.js      (all React components)
│   └── App.css     (all styles)
├── data/
│   ├── raw/        (place UpdatedResumeDataSet.csv here)
│   └── processed/  (dataset.csv generated here)
├── models/         (trained artifacts)
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   ├── matching/
│   └── skills/
├── tests/
├── evaluation/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / navigate to project
cd CareerAI

# 2. Install Python dependencies
python -m pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..
```

---

## How to Train the Model

```bash
# Step 1: Download the real dataset (one-time, ~23 MB parquet from HuggingFace)
python _build_real_dataset.py
# Writes data/raw/updated_resume_dataset.csv  (Apache 2.0 license)

# Step 2: Train
python src/training/train.py
```

Artifacts saved to `models/`:
- `best_model.joblib`
- `tfidf_vectorizer.joblib`
- `label_encoder.joblib`
- `logistic_regression.joblib`
- `random_forest.joblib`
- `training_metadata.json`

---

## How to Run the Application

### 1. Start Backend

```bash
python backend/app.py
# Backend runs at http://localhost:5000
```

### 2. Start Frontend

```bash
cd frontend
npm start
# Frontend runs at http://localhost:3000
```

---

## Example Workflow

1. Navigate to http://localhost:3000
2. Drop a PDF resume onto the upload area
3. Paste a job description into the text area
4. Click **Analyze Resume**
5. View:
   - Overall match score (0-100) with breakdown
   - Predicted job role with per-class probabilities
   - Matched and missing skills
   - Actionable recommendations
6. Click **Model Evaluation** tab to see real metrics and confusion matrix

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: preprocessing, skill extraction, matching engine, API endpoints, model inference.

---

## Limitations

1. Synthetic dataset by default — download Kaggle CSV for real-world performance
2. Score of 100% on synthetic data is expected (all templates are distinct) and does NOT represent real-world accuracy
3. Skill extraction uses regex, not semantic matching — skill aliases must be manually added
4. PDF extraction fails on scanned/image-based PDFs
5. Matching score reflects textual overlap only — NOT hiring probability

---

## Future Improvements

1. Add Kaggle CC0 dataset or LinkedIn job postings dataset
2. Add SVM and Gradient Boosting to model comparison
3. Add local LLM (Mistral/LLaMA via Ollama) for richer recommendations
4. Add RAG with ChromaDB over career development documents
5. OCR support for scanned resumes
6. User account system to track improvement over time
7. Automated skill vocabulary expansion from corpus
8. Confidence calibration (Platt scaling or isotonic regression)

---

## Ethical Notes

- No personal data is stored — resume text is processed in memory and discarded
- Score is NOT a hiring probability and is clearly labelled as such
- Model confidence is the softmax probability of the classifier, not an employment prediction
- Synthetic training data is transparently disclosed
