"""
tests/test_preprocessing.py
----------------------------
Tests for text cleaning and preprocessing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from src.preprocessing.text_cleaner import (
    clean_text,
    normalize_unicode,
    preprocess_for_model,
    preprocess_for_matching,
    tokenize_and_lemmatize,
)


class TestNormalizeUnicode:
    def test_removes_accent(self):
        assert normalize_unicode("résumé") == "resume"

    def test_plain_text_unchanged(self):
        assert normalize_unicode("Python") == "Python"


class TestCleanText:
    def test_lowercases(self):
        result = clean_text("PYTHON Machine Learning")
        assert result == result.lower()

    def test_removes_url(self):
        result = clean_text("Visit https://github.com/user/repo for code")
        assert "https" not in result
        assert "github" not in result

    def test_removes_email(self):
        result = clean_text("Contact me at user@example.com")
        assert "@" not in result

    def test_removes_bullet_chars(self):
        result = clean_text("• Python\n● Machine Learning\n▪ Docker")
        assert "•" not in result
        assert "●" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_preserves_hyphenated_terms(self):
        result = clean_text("full-stack developer")
        assert "full" in result
        assert "stack" in result

    def test_removes_standalone_numbers(self):
        result = clean_text("5 years of experience")
        # numbers should be stripped
        assert " 5 " not in result


class TestTokenizeLemmatize:
    def test_returns_list(self):
        tokens = tokenize_and_lemmatize("machine learning developer")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_lemmatizes_plural(self):
        tokens = tokenize_and_lemmatize("developers build applications")
        # "developers" → "developer", "applications" → "application"
        assert "developer" in tokens

    def test_removes_stopwords_by_default(self):
        tokens = tokenize_and_lemmatize("the quick brown fox")
        assert "the" not in tokens

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize_and_lemmatize("the quick brown fox", remove_stops=False)
        assert "the" in tokens


class TestPreprocessForModel:
    def test_returns_non_empty_string(self):
        result = preprocess_for_model("Python developer with 5 years experience in ML")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_consistent_output(self):
        text = "Senior Machine Learning Engineer with TensorFlow experience"
        r1 = preprocess_for_model(text)
        r2 = preprocess_for_model(text)
        assert r1 == r2  # deterministic

    def test_empty_input(self):
        assert preprocess_for_model("") == ""

    def test_very_noisy_text(self):
        noisy = "•••  https://github.com  •••  user@test.com  •••  \n\n\n"
        result = preprocess_for_model(noisy)
        # Should not crash and should return a string (possibly empty)
        assert isinstance(result, str)
