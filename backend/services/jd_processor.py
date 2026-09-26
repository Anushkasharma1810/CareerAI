"""
jd_processor.py
---------------
Job Description processing service.

Accepts:
  1. Pasted text
  2. Uploaded text file
  3. Uploaded PDF

Extracts:
  - required/preferred skills
  - technical keywords
  - job title/role (heuristic)
  - cleaned text for downstream matching
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import clean_text
from src.skills.skill_extractor import extract_skills

# Known job title patterns
_TITLE_PATTERNS = [
    r"(?:job\s+title|position|role)[:\s]+([A-Za-z][A-Za-z /\-&]{3,50})",
    r"(?:we are looking for|seeking)[:\s]+(?:a\s+|an\s+)?([A-Za-z][A-Za-z /\-&]{3,50})",
    r"(?:hiring)[:\s]+(?:a\s+|an\s+)?([A-Za-z][A-Za-z /\-&]{3,50})",
]

# Keywords that indicate required vs preferred
_REQUIRED_MARKERS = [
    "required", "must have", "must-have", "mandatory", "essential",
    "need", "you will", "you must", "responsibilities", "key skills",
]
_PREFERRED_MARKERS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "desirable",
    "plus", "advantage", "ideally", "optionally",
]


def _extract_job_title(text: str) -> str | None:
    """Heuristic extraction of job title from JD text."""
    for pattern in _TITLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip().rstrip(".,;:")
            if 4 <= len(title) <= 80:
                return title
    return None


def _split_required_preferred(text: str) -> tuple[str, str]:
    """
    Very rough split of JD into required/preferred sections.
    Returns (required_text, preferred_text).
    """
    lower = text.lower()
    preferred_markers = ["preferred", "nice to have", "bonus", "desirable", "plus if"]

    best_split = len(text)
    for marker in preferred_markers:
        idx = lower.find(marker)
        if idx != -1 and idx < best_split:
            best_split = idx

    if best_split < len(text) * 0.8:  # avoid splitting at end
        return text[:best_split], text[best_split:]
    return text, ""


def process_job_description(text: str) -> dict:
    """
    Process a job description text string.

    Parameters
    ----------
    text : str  Raw job description text

    Returns
    -------
    dict:
        cleaned_text       str
        job_title          str | None
        all_skills         list[str]
        required_skills    list[str]
        preferred_skills   list[str]
        char_count         int
        word_count         int
    """
    if not text or not text.strip():
        return {
            "error": "Empty job description provided.",
            "cleaned_text": "",
            "all_skills": [],
        }

    if len(text) < 30:
        return {
            "error": "Job description is too short. Please provide a complete JD.",
            "cleaned_text": "",
            "all_skills": [],
        }

    # Extract title before cleaning (needs original casing)
    job_title = _extract_job_title(text)

    # Split into required/preferred sections
    required_text, preferred_text = _split_required_preferred(text)

    # Extract skills from each section
    required_skills = extract_skills(required_text)
    preferred_skills = [s for s in extract_skills(preferred_text) if s not in required_skills]
    all_skills = extract_skills(text)  # full dedup pass

    # Clean for downstream use
    cleaned = clean_text(text)

    return {
        "cleaned_text": cleaned,
        "original_text": text,
        "job_title": job_title,
        "all_skills": all_skills,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def process_jd_file(file_bytes: bytes, filename: str) -> dict:
    """
    Process an uploaded JD file (PDF or plain text).
    """
    fname = filename.lower()

    if fname.endswith(".pdf"):
        # Import here to avoid circular dependency
        from backend.services.pdf_extractor import extract_text_from_bytes
        result = extract_text_from_bytes(file_bytes, filename)
        text = result["text"]
    elif fname.endswith((".txt", ".md", ".rst")):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="replace")
    else:
        return {
            "error": f"Unsupported JD file type '{filename}'. Upload PDF or .txt file."
        }

    return process_job_description(text)
