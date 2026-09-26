"""
skill_extractor.py
------------------
Extracts canonical skills from free-form text using the configurable
SKILL_VOCABULARY.  Not hard-coded — driven entirely by the vocabulary module.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.skills.skill_vocabulary import SKILL_VOCABULARY, normalize_skill


def _build_pattern_list() -> list[tuple[re.Pattern, str]]:
    """
    Build compiled regex patterns for all skill terms + their aliases.
    Sorted longest-first to match more specific terms first.
    """
    entries = []
    for canonical, aliases in SKILL_VOCABULARY.items():
        all_terms = [canonical] + aliases
        for term in all_terms:
            # Escape and compile with word boundary
            escaped = re.escape(term)
            pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
            entries.append((pattern, canonical))

    # Longest term first (prevents partial matches eating longer ones)
    entries.sort(key=lambda x: -len(x[0].pattern))
    return entries


_PATTERNS = _build_pattern_list()


def extract_skills(text: str) -> list[str]:
    """
    Extract a deduplicated list of canonical skill names from text.
    Returns skills ordered by their first occurrence position.
    """
    if not text:
        return []

    found: dict[str, int] = {}  # canonical → position of first match

    for pattern, canonical in _PATTERNS:
        match = pattern.search(text)
        if match:
            pos = match.start()
            if canonical not in found:
                found[canonical] = pos

    # Sort by occurrence position
    ordered = sorted(found.items(), key=lambda x: x[1])
    return [skill for skill, _ in ordered]


def extract_skills_set(text: str) -> set[str]:
    """Return extracted skills as a set (for fast membership testing)."""
    return set(extract_skills(text))


def extract_skills_with_context(text: str) -> list[dict]:
    """
    Extract skills with surrounding context snippet.
    Useful for debugging and explainability.
    """
    if not text:
        return []

    results = []
    seen = set()

    for pattern, canonical in _PATTERNS:
        for match in pattern.finditer(text):
            if canonical in seen:
                continue
            seen.add(canonical)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace("\n", " ").strip()
            results.append({
                "skill": canonical,
                "position": match.start(),
                "matched_text": match.group(),
                "context": f"...{context}...",
            })

    results.sort(key=lambda x: x["position"])
    return results


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Senior ML Engineer with 5 years of experience in Python, TensorFlow, and PyTorch.
    Built NLP pipelines using Hugging Face transformers and deployed models on AWS SageMaker.
    Proficient in Docker, Kubernetes, and CI/CD pipelines. Used scikit-learn, pandas, NumPy.
    Databases: PostgreSQL, MongoDB, Redis. Frontend: React, TypeScript. Git, Linux, Agile.
    """
    skills = extract_skills(sample)
    print(f"Extracted {len(skills)} skills:")
    for s in skills:
        print(f"  - {s}")
