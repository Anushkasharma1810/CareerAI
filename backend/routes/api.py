"""
routes/api.py
-------------
Flask API routes for CareerAI.

Endpoints:
  GET  /health
  GET  /model-info
  POST /upload-resume
  POST /predict-role
  POST /match-job
  POST /analyze
"""

import logging
import sys
from pathlib import Path

from flask import Blueprint, jsonify, request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import get_predictor, ModelNotTrainedError
from src.matching.matcher import compute_match_score
from src.skills.skill_extractor import extract_skills
from backend.services.pdf_extractor import extract_text_from_bytes, PDFExtractionError
from backend.services.jd_processor import process_job_description
from backend.services.analyzer import analyze_resume

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_RESUME_SIZE = 10 * 1024 * 1024  # 10 MB


def _bad_request(message: str, code: int = 400) -> tuple:
    return jsonify({"status": "error", "error": message}), code


# ── GET /health ────────────────────────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    """Service health check."""
    try:
        predictor = get_predictor()
        model_status = "loaded" if predictor.is_loaded else "not_loaded"
    except ModelNotTrainedError:
        model_status = "not_trained"
    except Exception:
        model_status = "error"

    return jsonify({
        "status": "ok",
        "model_status": model_status,
        "service": "CareerAI",
        "version": "1.0.0",
    }), 200


# ── GET /model-info ────────────────────────────────────────────────────────────
@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """Return trained model metadata and evaluation metrics."""
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        return jsonify({"status": "success", "data": info}), 200
    except ModelNotTrainedError as e:
        return _bad_request(str(e), 503)
    except Exception as e:
        logger.exception("Error in /model-info")
        return _bad_request(f"Error loading model info: {str(e)}", 500)


# ── POST /upload-resume ────────────────────────────────────────────────────────
@api_bp.route("/upload-resume", methods=["POST"])
def upload_resume():
    """
    Upload a PDF resume and return extracted text + skills.
    Form data: resume (file)
    """
    if "resume" not in request.files:
        return _bad_request("No 'resume' file in request.")

    f = request.files["resume"]
    if not f.filename:
        return _bad_request("No file selected.")

    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _bad_request(f"Only PDF files are accepted. Got '{suffix}'.")

    file_bytes = f.read()

    if len(file_bytes) > MAX_RESUME_SIZE:
        return _bad_request(
            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Max: 10 MB."
        )

    try:
        result = extract_text_from_bytes(file_bytes, f.filename)
    except PDFExtractionError as e:
        return _bad_request(str(e))
    except Exception as e:
        logger.exception("Unexpected error in /upload-resume")
        return _bad_request(f"Error processing file: {str(e)}", 500)

    skills = extract_skills(result["text"])

    return jsonify({
        "status": "success",
        "data": {
            "page_count": result["page_count"],
            "char_count": result["char_count"],
            "extraction_method": result["method"],
            "skills": skills,
            "text_preview": result["text"][:500],
            "warnings": result["warnings"],
        },
    }), 200


# ── POST /predict-role ─────────────────────────────────────────────────────────
@api_bp.route("/predict-role", methods=["POST"])
def predict_role():
    """
    Predict job role from resume text.
    JSON body: {"text": "resume text here"}
    OR Form data with uploaded resume PDF.
    """
    # Accept JSON or form data
    text = None
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
    else:
        text = request.form.get("text", "")
        if not text and "resume" in request.files:
            f = request.files["resume"]
            file_bytes = f.read()
            try:
                extraction = extract_text_from_bytes(file_bytes, f.filename)
                text = extraction["text"]
            except PDFExtractionError as e:
                return _bad_request(str(e))

    if not text or not text.strip():
        return _bad_request("No text provided. Send {'text': '...'} or upload a resume PDF.")

    try:
        predictor = get_predictor()
        prediction = predictor.predict(text)
        return jsonify({"status": "success", "data": prediction}), 200
    except ModelNotTrainedError as e:
        return _bad_request(str(e), 503)
    except Exception as e:
        logger.exception("Error in /predict-role")
        return _bad_request(f"Prediction error: {str(e)}", 500)


# ── POST /match-job ────────────────────────────────────────────────────────────
@api_bp.route("/match-job", methods=["POST"])
def match_job():
    """
    Compute resume-JD match score.
    JSON body:
      {"resume_text": "...", "jd_text": "..."}
    """
    if not request.is_json:
        return _bad_request("Request must be JSON.")

    data = request.get_json(silent=True) or {}
    resume_text = data.get("resume_text", "").strip()
    jd_text = data.get("jd_text", "").strip()

    if not resume_text:
        return _bad_request("Missing 'resume_text' in request body.")
    if not jd_text:
        return _bad_request("Missing 'jd_text' in request body.")

    try:
        match_result = compute_match_score(resume_text, jd_text)
        return jsonify({"status": "success", "data": match_result}), 200
    except Exception as e:
        logger.exception("Error in /match-job")
        return _bad_request(f"Matching error: {str(e)}", 500)


# ── POST /analyze ──────────────────────────────────────────────────────────────
@api_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Full analysis pipeline in one call.
    Form data:
      resume      (file, required)  PDF resume
      jd_text     (str, optional)   pasted job description
      jd_file     (file, optional)  uploaded JD file
    """
    if "resume" not in request.files:
        return _bad_request("No 'resume' PDF file in request.")

    resume_file = request.files["resume"]
    if not resume_file.filename:
        return _bad_request("No resume file selected.")

    suffix = Path(resume_file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return _bad_request(f"Only PDF resumes are accepted. Got '{suffix}'.")

    resume_bytes = resume_file.read()

    # Job description
    jd_text = request.form.get("jd_text", "").strip() or None
    jd_file_bytes = None
    jd_filename = None

    if "jd_file" in request.files:
        jd_file = request.files["jd_file"]
        if jd_file.filename:
            jd_file_bytes = jd_file.read()
            jd_filename = jd_file.filename

    if not jd_text and not jd_file_bytes:
        return _bad_request("Provide a job description via 'jd_text' field or 'jd_file' upload.")

    try:
        analysis = analyze_resume(
            resume_bytes=resume_bytes,
            resume_filename=resume_file.filename,
            jd_text=jd_text,
            jd_file_bytes=jd_file_bytes,
            jd_filename=jd_filename,
        )

        if analysis.get("status") == "error":
            return jsonify(analysis), 422

        return jsonify(analysis), 200

    except Exception as e:
        logger.exception("Unexpected error in /analyze")
        return _bad_request(f"Analysis failed: {str(e)}", 500)
