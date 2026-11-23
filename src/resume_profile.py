"""Conservative, evidence-based resume section parsing for JobScope.

These utilities extract possible education and experience evidence. They do not
verify credentials, employment history, dates, or hiring eligibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECTION_HEADERS = {
    "education": {"education", "academic background", "academic history", "educational background"},
    "experience": {"experience", "work experience", "professional experience", "employment history", "work history", "internships", "professional background"},
    "skills": {"skills", "technical skills", "core competencies", "skills and tools"},
    "projects": {"projects", "selected projects", "academic projects", "personal projects"},
    "other": {"awards", "certifications", "publications", "leadership", "activities", "summary", "profile", "objective", "research"},
}

EDUCATION_PATTERN = re.compile(
    r"\b(?:bachelor(?:'s)?|master(?:'s)?|ph\.?d\.?|doctorate|"
    r"b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|"
    r"university|college|institute of technology|degree|b\.eng|m\.eng)\b",
    flags=re.IGNORECASE,
)
EXPERIENCE_PATTERN = re.compile(
    r"\b(?:intern|analyst|engineer|scientist|research assistant|researcher|"
    r"developer|manager|consultant|associate|coordinator|lead|director|"
    r"assistant|specialist|accountant|designer)\b",
    flags=re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def _heading(text: str) -> str | None:
    normalized = re.sub(r"[\s:—–\-]+", " ", text.strip()).casefold()
    normalized = normalized.strip(" •_|*#")
    for category, words in SECTION_HEADERS.items():
        if normalized in words:
            return category
    return None


def parse_sections(text: str) -> dict[str, list[str]]:
    """Group lines under explicit section headings, retaining unmatched lines."""
    result = {"education": [], "experience": [], "skills": [], "projects": [], "other": []}
    active = "other"
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" \t")
        if not line:
            continue
        found = _heading(line)
        if found is not None:
            active = found
            continue
        result[active].append(line)
    return result


def _unique(lines: list[str], limit: int = 12) -> list[str]:
    seen = set()
    output = []
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(line)
        if len(output) >= limit:
            break
    return output


def extract_profile_evidence(text: str) -> dict[str, object]:
    """Extract candidate evidence without inventing schools, employers or dates.

    Education/experience summaries always quote or closely retain source lines.
    Missing evidence is reported as unverified, not inferred from other text.
    """
    sections = parse_sections(text)
    all_lines = [re.sub(r"\s+", " ", s).strip() for s in text.splitlines() if s.strip()]
    education = sections["education"] or [s for s in all_lines if EDUCATION_PATTERN.search(s)]
    experience = sections["experience"] or [s for s in all_lines if EXPERIENCE_PATTERN.search(s)]
    education = _unique(education, 8)
    experience = _unique(experience, 12)
    email = EMAIL_PATTERN.search(text)
    return {
        "education_lines": education,
        "experience_lines": experience,
        "project_lines": _unique(sections["projects"], 8),
        "email": email.group(0) if email else "",
        "education_summary": " · ".join(education[:2]) if education else "Not clearly identified — confirm manually",
        "experience_summary": " · ".join(experience[:2]) if experience else "Not clearly identified — confirm manually",
        "education_found": bool(education),
        "experience_found": bool(experience),
    }
