"""
text_cleaner.py
---------------
Core text preprocessing utilities for resume and job-description text.
Used by both the training pipeline and real-time inference.
"""

import re
import string
import unicodedata
import nltk

# Download required NLTK data at first use (idempotent)
# NLTK 3.9+ uses punkt_tab instead of punkt
_NLTK_PACKAGES = [
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]
for _find_path, _pkg_name in _NLTK_PACKAGES:
    try:
        nltk.data.find(_find_path)
    except LookupError:
        nltk.download(_pkg_name, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOP_WORDS = set(stopwords.words("english"))
# Keep domain-relevant negations in text (e.g. "not required")
_STOP_WORDS -= {"no", "not", "nor"}

_LEMMATIZER = WordNetLemmatizer()

# Characters/patterns that carry no semantic value in resumes
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
_PHONE_PATTERN = re.compile(r"\+?\d[\d\s\-().]{7,}\d")
_BULLET_PATTERN = re.compile(r"[•●▪►◦‣⁃▶]")
_MULTIPLE_SPACES = re.compile(r"\s+")
_NUMBERS_ONLY = re.compile(r"\b\d+\b")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def remove_noise(text: str) -> str:
    """Strip URLs, emails, phone numbers, bullet characters."""
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _PHONE_PATTERN.sub(" ", text)
    text = _BULLET_PATTERN.sub(" ", text)
    return text


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Unicode normalization
    2. Lowercase
    3. Remove noise (URLs, emails, phones)
    4. Remove punctuation except hyphens (keep "machine-learning")
    5. Remove isolated numbers
    6. Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = text.lower()
    text = remove_noise(text)
    # Keep hyphens between words (technical terms like "full-stack")
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = _NUMBERS_ONLY.sub(" ", text)
    text = _MULTIPLE_SPACES.sub(" ", text)
    return text.strip()


def tokenize_and_lemmatize(text: str, remove_stops: bool = True) -> list[str]:
    """
    Tokenize cleaned text and apply lemmatization.
    Optionally remove stop words.
    Returns list of tokens.
    """
    tokens = word_tokenize(text)
    result = []
    for token in tokens:
        # Skip pure punctuation tokens and very short tokens
        if len(token) < 2:
            continue
        if token in string.punctuation:
            continue
        if remove_stops and token in _STOP_WORDS:
            continue
        lemma = _LEMMATIZER.lemmatize(token)
        result.append(lemma)
    return result


def preprocess_for_model(text: str) -> str:
    """
    Full pipeline → returns a clean string suitable for TF-IDF vectorization.
    Used consistently in both training and inference.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned, remove_stops=True)
    return " ".join(tokens)


def preprocess_for_matching(text: str) -> str:
    """
    Lighter pipeline for skill/keyword matching — keeps stop words,
    preserves hyphenated compounds.
    """
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned, remove_stops=False)
    return " ".join(tokens)
