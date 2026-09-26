"""
analyzer.py
-----------
Orchestrates the full resume analysis pipeline:
  1. Extract text from resume PDF
  2. Process job description
  3. Predict job role via ML model
  4. Compute match score
  5. Assemble final report
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import get_predictor
from src.matching.matcher import compute_match_score
from src.skills.skill_extractor import extract_skills

from backend.services.pdf_extractor import extract_text_from_bytes, PDFExtractionError
from backend.services.jd_processor import process_job_description, process_jd_file


def analyze_resume(
    resume_bytes: bytes,
    resume_filename: str,
    jd_text: str = None,
    jd_file_bytes: bytes = None,
    jd_filename: str = None,
) -> dict:
    """
    Full analysis pipeline.

    Parameters
    ----------
    resume_bytes    : bytes  PDF file content
    resume_filename : str    original filename
    jd_text         : str    pasted job description (optional)
    jd_file_bytes   : bytes  uploaded JD file (optional)
    jd_filename     : str    JD filename if uploaded

    Returns
    -------
    Structured analysis dict.
    """
    result = {
        "status": "success",
        "resume": {},
        "job_description": {},
        "role_prediction": {},
        "match_analysis": {},
    }

    # ── Step 1: Extract resume text ────────────────────────────────────────
    try:
        pdf_result = extract_text_from_bytes(resume_bytes, resume_filename)
    except PDFExtractionError as e:
        return {"status": "error", "error": str(e), "stage": "resume_extraction"}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error reading PDF: {str(e)}", "stage": "resume_extraction"}

    resume_text = pdf_result["text"]
    resume_skills = extract_skills(resume_text)

    result["resume"] = {
        "page_count": pdf_result["page_count"],
        "char_count": pdf_result["char_count"],
        "extraction_method": pdf_result["method"],
        "warnings": pdf_result["warnings"],
        "skills": resume_skills,
        "text_preview": resume_text[:500] + ("..." if len(resume_text) > 500 else ""),
    }

    # ── Step 2: Process job description ───────────────────────────────────
    if jd_file_bytes and jd_filename:
        jd_result = process_jd_file(jd_file_bytes, jd_filename)
    elif jd_text:
        jd_result = process_job_description(jd_text)
    else:
        return {"status": "error", "error": "No job description provided.", "stage": "jd_processing"}

    if "error" in jd_result and not jd_result.get("cleaned_text"):
        return {"status": "error", "error": jd_result["error"], "stage": "jd_processing"}

    result["job_description"] = {
        "job_title": jd_result.get("job_title"),
        "all_skills": jd_result.get("all_skills", []),
        "required_skills": jd_result.get("required_skills", []),
        "preferred_skills": jd_result.get("preferred_skills", []),
        "word_count": jd_result.get("word_count", 0),
    }

    # ── Step 3: ML role prediction ─────────────────────────────────────────
    try:
        predictor = get_predictor()
        prediction = predictor.predict(resume_text)
        result["role_prediction"] = prediction
    except Exception as e:
        result["role_prediction"] = {
            "predicted_role": "Unknown",
            "confidence": 0.0,
            "error": str(e),
        }

    # ── Step 4: Match analysis ─────────────────────────────────────────────
    jd_original = jd_result.get("original_text") or jd_text or ""
    match = compute_match_score(resume_text, jd_original)
    result["match_analysis"] = match

    return result
