"""
tests/test_matching.py
-----------------------
Tests for resume-JD matching engine.

Covers:
  - Basic contract (dict shape, key presence, range)
  - Correctness (strong > weak, matched subset, missing non-empty)
  - Edge cases (empty inputs)
  - Monotonicity (adding a matched skill never decreases score)
  - Determinism (same inputs -> same output every time)
  - Alias normalisation (RESTful, sklearn, ML, deep-learning hyphen)
  - Cross-domain discrimination (tech JD vs non-tech resume -> low score)
  - Debug dict presence and weight assertion
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from src.matching.matcher import compute_match_score, SKILL_WEIGHT, COSINE_WEIGHT, KEYWORD_WEIGHT


STRONG_RESUME = """
Senior Machine Learning Engineer with 6 years of experience.
Skills: Python, TensorFlow, PyTorch, scikit-learn, NLP, Docker, Kubernetes, AWS, SQL,
pandas, NumPy, Git, CI/CD, REST API, deep learning, computer vision, MLOps.
"""

EXACT_JD = """
We are looking for a Machine Learning Engineer.
Required: Python, TensorFlow, scikit-learn, NLP, Docker, AWS, SQL, pandas.
Preferred: PyTorch, Kubernetes, computer vision, MLOps.
"""

WEAK_RESUME = """
Sales manager with 10 years of experience in retail management.
Skills: communication, leadership, Excel, PowerPoint, customer service, negotiation.
"""

TECHNICAL_JD = """
Full Stack Web Developer position.
Required: React, TypeScript, Node.js, PostgreSQL, Docker, Git, CSS, HTML.
Preferred: Kubernetes, GraphQL, AWS, CI/CD.
"""


class TestComputeMatchScore:
    # ── Basic contract ─────────────────────────────────────────────────────
    def test_returns_dict(self):
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        for key in ("overall_score", "matched_skills", "missing_skills",
                    "recommendations", "resume_skills", "jd_skills",
                    "skill_score", "cosine_score", "keyword_score", "methodology"):
            assert key in result, f"Missing key: {key}"

    def test_score_in_valid_range(self):
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        assert 0 <= result["overall_score"] <= 100

    # ── Correctness ────────────────────────────────────────────────────────
    def test_strong_match_higher_than_weak(self):
        strong = compute_match_score(STRONG_RESUME, EXACT_JD)
        weak = compute_match_score(WEAK_RESUME, EXACT_JD)
        assert strong["overall_score"] > weak["overall_score"]

    def test_matched_skills_subset_of_both(self):
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        matched = set(result["matched_skills"])
        resume_skills = set(result["resume_skills"])
        jd_skills = set(result["jd_skills"])
        # All matched skills must appear in both resume and JD skill sets
        assert matched.issubset(resume_skills), "Matched skill not in resume_skills"
        assert matched.issubset(jd_skills), "Matched skill not in jd_skills"
        assert len(matched) > 0, "Expected non-empty matched skills for strong resume"

    def test_missing_skills_not_in_resume(self):
        result = compute_match_score(WEAK_RESUME, TECHNICAL_JD)
        assert len(result["missing_skills"]) > 0

    # ── Edge cases ─────────────────────────────────────────────────────────
    def test_empty_resume_returns_error(self):
        result = compute_match_score("", EXACT_JD)
        assert "error" in result

    def test_empty_jd_returns_error(self):
        result = compute_match_score(STRONG_RESUME, "")
        assert "error" in result

    def test_whitespace_only_resume_returns_error(self):
        result = compute_match_score("   \n\t  ", EXACT_JD)
        assert "error" in result

    # ── Recommendations ────────────────────────────────────────────────────
    def test_recommendations_is_list(self):
        result = compute_match_score(WEAK_RESUME, TECHNICAL_JD)
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_methodology_documented(self):
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        assert "methodology" in result
        assert len(result["methodology"]) > 10

    # ── Determinism (scenario 1) ───────────────────────────────────────────
    def test_score_is_deterministic(self):
        """Same resume + same JD must always produce the exact same score."""
        r1 = compute_match_score(STRONG_RESUME, EXACT_JD)
        r2 = compute_match_score(STRONG_RESUME, EXACT_JD)
        assert r1["overall_score"] == r2["overall_score"]
        assert r1["matched_skills"] == r2["matched_skills"]
        assert r1["missing_skills"] == r2["missing_skills"]

    # ── Cross-domain discrimination ────────────────────────────────────────
    def test_cross_domain_low_score(self):
        result = compute_match_score(WEAK_RESUME, TECHNICAL_JD)
        assert result["overall_score"] < 50

    # ── Monotonicity (scenario 2 & 3) ─────────────────────────────────────
    def test_adding_required_skill_does_not_decrease_score(self):
        """Adding a skill that is in the JD must not decrease the overall score."""
        base_resume    = "Python machine learning scikit-learn SQL"
        base_jd        = "Python machine learning scikit-learn SQL Docker AWS"
        extended_resume = base_resume + " Docker"

        r_base     = compute_match_score(base_resume, base_jd)
        r_extended = compute_match_score(extended_resume, base_jd)

        assert r_extended["overall_score"] >= r_base["overall_score"], (
            f"Adding Docker decreased score: {r_base['overall_score']} -> {r_extended['overall_score']}"
        )

    def test_removing_matched_skill_does_not_increase_score(self):
        """Removing a skill that was matched must not increase the overall score."""
        full_resume    = "Python machine learning scikit-learn SQL Docker"
        reduced_resume = "Python machine learning scikit-learn SQL"
        jd             = "Python machine learning scikit-learn SQL Docker AWS"

        r_full    = compute_match_score(full_resume, jd)
        r_reduced = compute_match_score(reduced_resume, jd)

        assert r_reduced["overall_score"] <= r_full["overall_score"], (
            f"Removing Docker increased score: {r_full['overall_score']} -> {r_reduced['overall_score']}"
        )

    # ── Alias normalisation (scenarios 5-7) ───────────────────────────────
    def test_restful_alias_maps_to_rest_api(self):
        """'RESTful' in resume should match 'REST API' in JD via alias."""
        resume = "Python developer with RESTful services and API experience."
        jd     = "Looking for developer with REST API experience."
        result = compute_match_score(resume, jd)
        assert "REST API" in result["matched_skills"], (
            f"Expected 'REST API' in matched skills; got: {result['matched_skills']}"
        )

    def test_sklearn_alias_maps_to_scikit_learn(self):
        """'sklearn' in resume should match 'scikit-learn' in JD."""
        resume = "Data scientist proficient in sklearn and pandas."
        jd     = "Required: scikit-learn, Python, pandas."
        result = compute_match_score(resume, jd)
        assert "scikit-learn" in result["matched_skills"], (
            f"Expected 'scikit-learn' in matched skills; got: {result['matched_skills']}"
        )

    def test_ml_abbreviation_maps_to_machine_learning(self):
        """'ML' in resume should match 'machine learning' in JD."""
        resume = "ML engineer with production experience."
        jd     = "We need machine learning expertise."
        result = compute_match_score(resume, jd)
        assert "Machine Learning" in result["matched_skills"], (
            f"Expected 'Machine Learning' in matched skills; got: {result['matched_skills']}"
        )

    def test_deep_learning_hyphen_alias(self):
        """'deep-learning' (hyphenated) in text should match 'Deep Learning'."""
        resume = "Experience with deep-learning and neural networks."
        jd     = "Required: deep learning, neural networks."
        result = compute_match_score(resume, jd)
        assert "Deep Learning" in result["matched_skills"], (
            f"Expected 'Deep Learning' in matched; got: {result['matched_skills']}"
        )

    # ── Unrelated text -> low skill overlap (scenario 4) ──────────────────
    def test_unrelated_text_low_skill_overlap(self):
        """Unrelated non-technical text should yield near-zero skill overlap."""
        resume = "I enjoy hiking, cooking, and reading fiction novels."
        jd     = "Python machine learning Docker AWS Kubernetes SQL TensorFlow."
        result = compute_match_score(resume, jd)
        # With 0 shared skills, skill_score = 0.0 -> contribution = 0 pts
        assert result["skill_score"] == 0.0, (
            f"Expected skill_score=0, got {result['skill_score']}"
        )
        assert result["overall_score"] < 25

    # ── Debug dict (scenario 8) ────────────────────────────────────────────
    def test_debug_dict_present_and_complete(self):
        """debug dict must be present with all expected keys."""
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        assert "debug" in result
        d = result["debug"]
        for key in (
            "skill_overlap_raw", "cosine_raw", "keyword_density_raw",
            "skill_contribution", "cosine_contribution", "keyword_contribution",
            "raw_weighted_sum", "jd_skills_count", "resume_skills_count",
            "matched_skills_count", "missing_skills_count", "weights",
        ):
            assert key in d, f"Missing debug key: {key}"

    def test_weights_sum_to_one(self):
        """Scoring weights must always sum to exactly 1.0."""
        total = SKILL_WEIGHT + COSINE_WEIGHT + KEYWORD_WEIGHT
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_debug_contributions_sum_to_raw_score(self):
        """skill+cosine+keyword contributions must sum to raw_weighted_sum."""
        result = compute_match_score(STRONG_RESUME, EXACT_JD)
        d = result["debug"]
        total = d["skill_contribution"] + d["cosine_contribution"] + d["keyword_contribution"]
        assert abs(total - d["raw_weighted_sum"]) < 0.1, (
            f"Contributions {total:.4f} != raw_weighted_sum {d['raw_weighted_sum']:.4f}"
        )
