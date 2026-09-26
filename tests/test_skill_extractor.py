"""
tests/test_skill_extractor.py
------------------------------
Tests for skill extraction.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from src.skills.skill_extractor import extract_skills, extract_skills_set, extract_skills_with_context
from src.skills.skill_vocabulary import normalize_skill, get_all_canonical_skills


class TestExtractSkills:
    def test_extracts_known_skills(self):
        text = "Python developer with TensorFlow and Docker experience"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "TensorFlow" in skills
        assert "Docker" in skills

    def test_returns_canonical_form(self):
        text = "sklearn model trained on numpy data"
        skills = extract_skills(text)
        assert "scikit-learn" in skills  # sklearn → scikit-learn
        assert "NumPy" in skills         # numpy → NumPy

    def test_case_insensitive(self):
        text = "Experience with PYTHON, TENSORFLOW, and DOCKER"
        skills = extract_skills(text)
        assert "Python" in skills
        assert "TensorFlow" in skills

    def test_empty_text_returns_empty_list(self):
        assert extract_skills("") == []
        assert extract_skills(None) == []

    def test_no_duplicates(self):
        text = "Python Python Python Python developer"
        skills = extract_skills(text)
        assert skills.count("Python") == 1

    def test_sql_extracted(self):
        text = "proficient in SQL databases and query optimization"
        skills = extract_skills(text)
        assert "SQL" in skills

    def test_multi_word_skills(self):
        text = "experience with Machine Learning and Deep Learning"
        skills = extract_skills(text)
        assert "Machine Learning" in skills or "Deep Learning" in skills

    def test_returns_list(self):
        skills = extract_skills("Python developer")
        assert isinstance(skills, list)


class TestExtractSkillsSet:
    def test_returns_set(self):
        skills = extract_skills_set("Python Docker Kubernetes")
        assert isinstance(skills, set)

    def test_intersection_works(self):
        resume_skills = extract_skills_set("Python Docker SQL React")
        jd_skills = extract_skills_set("Python Docker AWS Kubernetes")
        matched = resume_skills & jd_skills
        assert "Python" in matched
        assert "Docker" in matched
        assert "React" not in matched


class TestExtractSkillsWithContext:
    def test_returns_list_of_dicts(self):
        result = extract_skills_with_context("Python and TensorFlow")
        assert isinstance(result, list)
        if result:
            assert "skill" in result[0]
            assert "context" in result[0]
            assert "position" in result[0]


class TestNormalizeSkill:
    def test_alias_maps_to_canonical(self):
        assert normalize_skill("sklearn") == "scikit-learn"
        assert normalize_skill("numpy") == "NumPy"
        assert normalize_skill("k8s") == "Kubernetes"

    def test_unknown_returns_none(self):
        assert normalize_skill("notarealskill123xyz") is None

    def test_canonical_maps_to_itself(self):
        assert normalize_skill("Python") == "Python"
        assert normalize_skill("Docker") == "Docker"


class TestGetAllCanonicalSkills:
    def test_returns_list(self):
        skills = get_all_canonical_skills()
        assert isinstance(skills, list)
        assert len(skills) > 50  # vocabulary should be substantial

    def test_no_duplicates(self):
        skills = get_all_canonical_skills()
        assert len(skills) == len(set(skills))
