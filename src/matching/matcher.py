"""
matcher.py
----------
Resume ↔ Job Description matching engine.

Scoring methodology (v1.1):
  1. Skill overlap score  (60% weight)
     = matched_skills / total_jd_skills   (recall-style coverage)
     Most reliable signal: directly measures required-skill coverage.

  2. TF-IDF cosine similarity  (25% weight)
     Measures semantic overlap beyond exact skill names.
     Weight reduced from 35% to 25% because TF-IDF cosine is structurally
     lower when one document (JD) is much shorter than the other (resume);
     giving it 35% systematically penalises strong matches.

  3. Keyword density score  (15% weight)
     Fraction of top-50 TF-IDF JD terms that appear in the resume.

Final score = weighted sum, clamped to [0, 100].
Score reflects text similarity only — NOT hiring probability.

Rationale for v1.1 weight change:
  - Skill overlap is the most explainable and reliable signal.
  - Cosine between a 200-word JD and an 800-word resume is structurally
    low (~0.30-0.40) even for an excellent match, because TF-IDF weights
    are relative to document length; it should not dominate the score.
  - Keyword density is a useful but coarser signal; kept at 15%.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.text_cleaner import clean_text, preprocess_for_matching
from src.skills.skill_extractor import extract_skills_set

# ── scoring weights (v1.1) ────────────────────────────────────────────────────
SKILL_WEIGHT   = 0.60   # increased from 0.50 — most reliable signal
COSINE_WEIGHT  = 0.25   # reduced from 0.35 — structurally low for short JDs
KEYWORD_WEIGHT = 0.15   # unchanged

assert abs(SKILL_WEIGHT + COSINE_WEIGHT + KEYWORD_WEIGHT - 1.0) < 1e-9, \
    "Weights must sum to 1.0"


def _cosine_sim(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two preprocessed documents."""
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def _keyword_density(resume_text: str, jd_text: str) -> float:
    """
    Fraction of significant JD words (top-50 TF-IDF terms) that appear in the resume.
    """
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=50)
        vectorizer.fit_transform([jd_text])
        jd_keywords = set(vectorizer.vocabulary_.keys())
        if not jd_keywords:
            return 0.0
        resume_words = set(resume_text.lower().split())
        matched_count = len(jd_keywords & resume_words)
        return matched_count / len(jd_keywords)
    except Exception:
        return 0.0


def compute_match_score(
    resume_text: str,
    jd_text: str,
) -> dict:
    """
    Full matching analysis between resume and job description.

    Parameters
    ----------
    resume_text : str  Raw extracted resume text
    jd_text     : str  Raw job description text

    Returns
    -------
    dict containing:
        overall_score         int 0-100
        skill_score           float 0-100
        cosine_score          float 0-100
        keyword_score         float 0-100
        matched_skills        list[str]
        missing_skills        list[str]
        resume_skills         list[str]
        jd_skills             list[str]
        keywords_matched      list[str]
        recommendations       list[str]
        strengths             list[str]
        methodology           str  scoring explanation
        debug                 dict  diagnostic breakdown
    """
    if not resume_text or not resume_text.strip() or not jd_text or not jd_text.strip():
        return {
            "overall_score": 0,
            "error": "Empty resume or job description provided.",
        }

    # ── 1. Skill extraction ────────────────────────────────────────────────
    resume_skills = extract_skills_set(resume_text)
    jd_skills     = extract_skills_set(jd_text)

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    # Skill overlap: recall-style — how much of the JD is covered by the resume
    if jd_skills:
        skill_score = len(matched_skills) / len(jd_skills)
    else:
        skill_score = 0.5  # no skills in JD → neutral

    # ── 2. TF-IDF cosine similarity ────────────────────────────────────────
    resume_processed = preprocess_for_matching(resume_text)
    jd_processed     = preprocess_for_matching(jd_text)
    cosine_score     = _cosine_sim(resume_processed, jd_processed)

    # ── 3. Keyword density ─────────────────────────────────────────────────
    keyword_score = _keyword_density(resume_processed, jd_processed)

    # ── Weighted overall score ─────────────────────────────────────────────
    raw_score = (
        skill_score   * SKILL_WEIGHT
        + cosine_score  * COSINE_WEIGHT
        + keyword_score * KEYWORD_WEIGHT
    )
    overall_score = int(min(100, max(0, round(raw_score * 100))))

    # ── Keywords matched (top JD terms found in resume) ────────────────────
    try:
        vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=50)
        vec.fit_transform([jd_processed])
        jd_top_words    = set(vec.vocabulary_.keys())
        resume_word_set = set(resume_processed.lower().split())
        keywords_matched = sorted(jd_top_words & resume_word_set)[:20]
    except Exception:
        keywords_matched = []

    # ── Strengths ──────────────────────────────────────────────────────────
    strengths = matched_skills[:10]

    # ── Recommendations ────────────────────────────────────────────────────
    recommendations = _generate_recommendations(missing_skills, matched_skills, overall_score)

    # ── Debug / diagnostic breakdown ───────────────────────────────────────
    debug = {
        "skill_overlap_raw":   round(skill_score, 4),
        "cosine_raw":          round(cosine_score, 4),
        "keyword_density_raw": round(keyword_score, 4),
        "skill_contribution":  round(skill_score   * SKILL_WEIGHT   * 100, 2),
        "cosine_contribution": round(cosine_score  * COSINE_WEIGHT  * 100, 2),
        "keyword_contribution":round(keyword_score * KEYWORD_WEIGHT * 100, 2),
        "raw_weighted_sum":    round(raw_score * 100, 4),
        "jd_skills_count":     len(jd_skills),
        "resume_skills_count": len(resume_skills),
        "matched_skills_count":len(matched_skills),
        "missing_skills_count":len(missing_skills),
        "weights": {
            "skill":   SKILL_WEIGHT,
            "cosine":  COSINE_WEIGHT,
            "keyword": KEYWORD_WEIGHT,
        },
    }

    return {
        "overall_score":  overall_score,
        "skill_score":    round(skill_score   * 100, 1),
        "cosine_score":   round(cosine_score  * 100, 1),
        "keyword_score":  round(keyword_score * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills[:20],
        "resume_skills":  sorted(resume_skills),
        "jd_skills":      sorted(jd_skills),
        "keywords_matched": keywords_matched,
        "strengths":      strengths,
        "recommendations": recommendations,
        "methodology": (
            f"Score = {int(SKILL_WEIGHT*100)}% skill overlap "
            f"+ {int(COSINE_WEIGHT*100)}% TF-IDF cosine similarity "
            f"+ {int(KEYWORD_WEIGHT*100)}% keyword density. "
            "Reflects text similarity only — NOT hiring probability."
        ),
        "debug": debug,
    }


# ── Recommendation engine ─────────────────────────────────────────────────────
_SKILL_RESOURCES: dict[str, str] = {
    "Docker": "Learn Docker via docker.com/get-started",
    "Kubernetes": "Study Kubernetes fundamentals at kubernetes.io/docs/tutorials/",
    "AWS": "Complete AWS Cloud Practitioner on aws.training",
    "GCP": "Complete Google Cloud Fundamentals on cloud.google.com/training",
    "Azure": "Complete AZ-900 on Microsoft Learn",
    "TensorFlow": "Complete TensorFlow Developer Certificate course on Coursera",
    "PyTorch": "Work through pytorch.org/tutorials",
    "Machine Learning": "Andrew Ng's ML Specialization on Coursera",
    "Deep Learning": "Deep Learning Specialization by deeplearning.ai",
    "NLP": "HuggingFace NLP Course at huggingface.co/learn",
    "Transformers": "Study Attention is All You Need paper + HuggingFace course",
    "SQL": "Practice SQL on Mode Analytics or SQLZoo",
    "PostgreSQL": "PostgreSQL Tutorial at postgresqltutorial.com",
    "MongoDB": "MongoDB University free courses at learn.mongodb.com",
    "React": "Official React documentation at react.dev",
    "TypeScript": "TypeScript Handbook at typescriptlang.org/docs",
    "Node.js": "Node.js Getting Started guide at nodejs.org/en/learn",
    "Terraform": "HashiCorp Learn at developer.hashicorp.com/terraform",
    "CI/CD": "GitHub Actions docs + Jenkins tutorials",
    "Cybersecurity": "TryHackMe or HackTheBox platforms for hands-on practice",
    "pandas": "pandas documentation + Kaggle pandas course",
    "Tableau": "Tableau Public training resources",
    "Power BI": "Microsoft Learn Power BI modules",
    "Git": "Official Git documentation at git-scm.com/doc",
    "Linux": "Linux Foundation free intro courses at training.linuxfoundation.org",
    "Agile": "Scrum Guide at scrumguides.org",
    "Statistical Analysis": "Statistics with Python Specialization on Coursera",
    "Data Visualization": "Tableau, Matplotlib, or D3.js learning paths",
    "REST API": "REST API design guide at restfulapi.net",
    "LightGBM": "LightGBM documentation at lightgbm.readthedocs.io",
    "Spark": "Apache Spark tutorials at spark.apache.org/docs/latest/",
}


def _generate_recommendations(
    missing_skills: list[str],
    matched_skills: list[str],
    overall_score: int,
) -> list[str]:
    """
    Generate actionable recommendations based on skill gaps.
    Entirely driven by gap analysis — not hard-coded for specific resumes.
    """
    recs = []

    for skill in missing_skills[:8]:
        resource = _SKILL_RESOURCES.get(skill)
        if resource:
            recs.append(f"Add {skill}: {resource}")
        else:
            recs.append(
                f"Gain experience with {skill} and add it to your resume with specific projects/examples"
            )

    if overall_score < 40:
        recs.append(
            "Your resume has low overlap with this job. Consider tailoring keywords and "
            "skills to match the job description more closely."
        )
    elif overall_score < 65:
        recs.append(
            "Moderate match. Add a 'Skills' section that mirrors the exact terms used in the job description."
        )
    elif overall_score >= 80:
        recs.append(
            "Strong match. Focus on quantifying your achievements (e.g., 'Improved X by Y%') "
            "to stand out further."
        )

    if matched_skills:
        recs.append(
            f"Highlight your matched skills ({', '.join(matched_skills[:4])}) "
            "prominently in your resume summary/objective."
        )

    if not missing_skills:
        recs.append(
            "Excellent skill coverage! Ensure your resume demonstrates depth in each area "
            "with specific project outcomes."
        )

    return recs[:10]
