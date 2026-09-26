"""
scripts/diagnose_matching.py
-----------------------------
Quick diagnostic to validate matching engine behavior and show
before/after score breakdown for the v1.1 weight changes.

Run:
    python scripts/diagnose_matching.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching.matcher import compute_match_score

# ── Test cases ────────────────────────────────────────────────────────────────

CASES = {
    "Strong ML Resume vs ML JD": {
        "resume": """
            Senior Machine Learning Engineer with 6 years of experience.
            Proficient in Python, TensorFlow, PyTorch, scikit-learn, NLP, deep learning,
            computer vision, Docker, Kubernetes, AWS, SQL, pandas, NumPy, Git, CI/CD,
            REST API, MLOps. Built production ML pipelines on AWS using Docker containers.
            Experience with RESTful APIs, machine learning model deployment, and MLflow.
        """,
        "jd": """
            We are looking for a Machine Learning Engineer.
            Required: Python, TensorFlow, scikit-learn, NLP, Docker, AWS, SQL, pandas, REST API.
            Preferred: PyTorch, Kubernetes, computer vision, MLOps, deep learning.
        """,
    },
    "Weak Sales Resume vs Tech JD": {
        "resume": """
            Sales manager with 10 years of experience in retail management.
            Skills: communication, leadership, Excel, PowerPoint, customer service, negotiation.
        """,
        "jd": """
            Full Stack Web Developer position.
            Required: React, TypeScript, Node.js, PostgreSQL, Docker, Git, CSS, HTML.
            Preferred: Kubernetes, GraphQL, AWS, CI/CD.
        """,
    },
    "RESTful alias test (resume has RESTful, JD has REST API)": {
        "resume": "Python developer with experience in RESTful services and APIs.",
        "jd": "Looking for developer with REST API experience.",
    },
    "sklearn alias test (resume: sklearn, JD: scikit-learn)": {
        "resume": "Data scientist proficient in sklearn and pandas.",
        "jd": "Required: scikit-learn, Python, pandas.",
    },
    "ML abbreviation test (resume: ML, JD: machine learning)": {
        "resume": "ML engineer with deep learning experience.",
        "jd": "We need machine learning expertise and deep learning skills.",
    },
    "Exact match (same text)": {
        "resume": "Python machine learning scikit-learn TensorFlow SQL pandas Docker",
        "jd": "Python machine learning scikit-learn TensorFlow SQL pandas Docker",
    },
}


def run_diagnostic():
    sep = "-" * 70
    print(f"\n{'CareerAI Matching Engine - Diagnostic Report':^70}")
    print(f"{'Weights: 60% skill overlap | 25% cosine | 15% keyword':^70}")
    print(sep)

    for case_name, texts in CASES.items():
        result = compute_match_score(texts["resume"], texts["jd"])
        d = result.get("debug", {})

        print(f"\n[CASE]  {case_name}")
        print(f"    Overall Score : {result['overall_score']}/100")
        print(f"    Matched Skills: {result.get('matched_skills', [])}")
        print(f"    Missing Skills: {result.get('missing_skills', [])[:8]}")
        print(f"\n    -- Component Breakdown --")
        print(f"    Skill overlap   : {d.get('skill_overlap_raw', 0):.4f}  -> contrib {d.get('skill_contribution', 0):.2f} pts")
        print(f"    Cosine sim      : {d.get('cosine_raw', 0):.4f}  -> contrib {d.get('cosine_contribution', 0):.2f} pts")
        print(f"    Keyword density : {d.get('keyword_density_raw', 0):.4f}  -> contrib {d.get('keyword_contribution', 0):.2f} pts")
        print(f"    Raw weighted sum: {d.get('raw_weighted_sum', 0):.2f}  ->  final: {result['overall_score']}")
        print(f"    JD skills ({d.get('jd_skills_count',0)}): resume skills ({d.get('resume_skills_count',0)}): matched ({d.get('matched_skills_count',0)})")
        print(sep)

    # ── Monotonicity check
    print("\n[CHECK]  Monotonicity Check")
    base_resume = "Python machine learning scikit-learn SQL"
    base_jd     = "Python machine learning scikit-learn SQL Docker AWS"
    r_base      = compute_match_score(base_resume, base_jd)

    extended_resume = base_resume + " Docker"
    r_extended  = compute_match_score(extended_resume, base_jd)

    print(f"    Base resume (4 skills matched)     : {r_base['overall_score']}/100")
    print(f"    Extended resume (+Docker = 5 matched): {r_extended['overall_score']}/100")
    assert r_extended["overall_score"] >= r_base["overall_score"], \
        "FAIL: Adding a matched skill decreased the score!"
    print("    [OK] Adding a skill does NOT decrease the score")

    # ── Determinism check
    r1 = compute_match_score(base_resume, base_jd)
    r2 = compute_match_score(base_resume, base_jd)
    assert r1["overall_score"] == r2["overall_score"], "FAIL: Score is not deterministic!"
    print("    [OK] Score is deterministic across two identical calls")

    print(f"\n{'Diagnostic complete':^70}\n")


if __name__ == "__main__":
    run_diagnostic()
