"""Signals I read from the job description text: years, degree, majors, internships.

These are simple rules (regular expressions), good enough for market level
numbers like "share of postings asking for 3+ years". They are not meant to
decide if one person qualifies for one job, that is what the eligibility flags
in feature_extraction.py are for.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(\d{1,2}|" + "|".join(WORD_NUMBERS) + r")"

# "3+ years of experience", "3-5 years experience", "two years of relevant
# work experience", "minimum of 5 yrs ... experience". The word "experience"
# must come within 40 characters, so I don't catch things like "100 years in business".
YEARS_PATTERN = re.compile(
    _NUM + r"\s*(?:\+|plus)?\s*(?:(?:-|to)\s*" + _NUM + r"\s*\+?\s*)?"
    r"(?:years?|yrs?)\b[^.;\n]{0,40}?experience",
    flags=re.IGNORECASE,
)

MAX_YEARS = 20

# Short forms (B.S., MS) only count with dots or with "degree"/"in" after them,
# otherwise "MS Office" would be read as a master's degree.
DEGREE_PATTERNS = [
    ("High school", re.compile(r"(?i:high school diploma|high school or equivalent)|\bGED\b")),
    ("Associate", re.compile(r"(?i:associate'?s? degree)")),
    ("Bachelor", re.compile(
        r"(?i:bachelor|undergraduate degree|four-year degree|4-year degree)"
        r"|\bB\.[SA]\.|\bB[SA] (?:degree|in)\b")),
    ("Master", re.compile(
        r"(?i:master'?s degree|master of|graduate degree)"
        r"|\bM\.S\.|\bMS (?:degree|in)\b|\bMBA\b")),
    ("PhD", re.compile(r"(?i:ph\.?\s?d\b|doctorate|doctoral degree)")),
]
DEGREE_ORDER = ["None stated"] + [name for name, _ in DEGREE_PATTERNS]


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else WORD_NUMBERS[token.lower()]


def min_years_required(text: str) -> float:
    """Smallest number of years of experience the text asks for, or NaN.

    For a range such as "3-5 years" the lower end (3) counts. Numbers above
    20 are ignored as noise.
    """
    if not isinstance(text, str):
        return np.nan
    found = [_to_int(m.group(1)) for m in YEARS_PATTERN.finditer(text)]
    found = [y for y in found if 0 < y <= MAX_YEARS]
    return float(min(found)) if found else np.nan


def lowest_degree_mentioned(text: str) -> str:
    """Lowest degree level the text mentions, or "None stated".

    Postings often say "Bachelor's required, Master's preferred", so the
    lowest level mentioned is the closer guess at the real requirement.
    """
    if not isinstance(text, str):
        return "None stated"
    for name, pattern in DEGREE_PATTERNS:
        if pattern.search(text):
            return name
    return "None stated"


def add_text_signals(jobs: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
    """Add min_years and degree columns read from description.

    Reading every description takes a while, so with cache_path the two
    columns are saved (keyed by job_id) and reused on the next run.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        if set(jobs["job_id"]).issubset(set(cached["job_id"])):
            return jobs.merge(cached, on="job_id", how="left")

    signals = pd.DataFrame({
        "job_id": jobs["job_id"].to_numpy(),
        "min_years": jobs["description"].apply(min_years_required).to_numpy(),
        "degree": jobs["description"].apply(lowest_degree_mentioned).to_numpy(),
    })
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(cache_path, index=False)
    return jobs.merge(signals, on="job_id", how="left")


# ---------------------------------------------------------------------------
# Majors (field of study)
# ---------------------------------------------------------------------------

# I start looking after a degree word and stop at the end of that sentence,
# so "Bachelor's degree required. Strong business sense" does not count
# Business as a major.
DEGREE_ANCHOR = re.compile(
    r"(?i:bachelor|master|associate'?s? degree|degree|diploma|ph\.?\s?d|doctorate|undergraduate)"
    r"|\bB\.[SA]\.|\bM\.S\.|\bBS\b|\bBA\b|\bMS\b|\bMBA\b"
)
WINDOW_END = re.compile(r"[.;:\n•]|\s{3,}")
WINDOW_MAX = 160

# Keyword rules for each major group, checked inside the window.
# Order matters only for display. A posting can list several majors.
MAJOR_PATTERNS = {
    "Computer Science": r"computer science|computer engineering|software engineering",
    "Information Systems / IT": r"information systems|information technology|\bmis\b|computer information",
    "Data Science / Analytics": r"data science|data analytics|analytics",
    "Statistics / Mathematics": r"statistic|mathematic|\bmath\b|applied math",
    "Engineering": r"(?<!computer )(?<!software )engineering",
    "Business": r"business|\bmba\b",
    "Finance": r"finance|financial",
    "Accounting": r"accounting",
    "Economics": r"economics",
    "Marketing / Communications": r"marketing|communications|journalism|public relations",
    "Nursing": r"nursing|\bbsn\b|\brn\b",
    "Health / Public Health": r"health administration|healthcare administration|public health|health science|pharmacy",
    "Life / Physical Sciences": r"biology|chemistry|biochem|life science|physics|environmental science",
    "Psychology / Social Sciences": r"psychology|sociology|social work|social science",
    "Education": r"in education|early childhood|special education|elementary education",
    "Human Resources": r"human resources",
    "Supply Chain / Logistics": r"supply chain|logistics",
    "Design / Arts": r"graphic design|design|fine arts",
}
_MAJOR_RES = {name: re.compile(p, flags=re.IGNORECASE) for name, p in MAJOR_PATTERNS.items()}
MAJOR_ORDER = list(MAJOR_PATTERNS)


def majors_mentioned(text: str) -> list[str]:
    """Major groups named right after a degree word, e.g. "BS in Computer Science".

    An empty list means no major was named. It does not mean any major is fine.
    """
    if not isinstance(text, str):
        return []
    found = set()
    for anchor in DEGREE_ANCHOR.finditer(text):
        if anchor.group(0) == "MBA":
            found.add("Business")
        rest = text[anchor.end(): anchor.end() + WINDOW_MAX]
        stop = WINDOW_END.search(rest)
        window = rest[: stop.start()] if stop else rest
        for name, pattern in _MAJOR_RES.items():
            if pattern.search(window):
                found.add(name)
    return [m for m in MAJOR_ORDER if m in found]


# ---------------------------------------------------------------------------
# Years groups, internships and research
# ---------------------------------------------------------------------------

YEARS_GROUPS = ["1 year", "2 to 3", "4 to 5", "6 to 9", "10+", "Not stated"]


def years_group(years: float) -> str:
    """Put a minimum-years value into a resume-style group."""
    if pd.isna(years):
        return "Not stated"
    if years <= 1:
        return "1 year"
    if years <= 3:
        return "2 to 3"
    if years <= 5:
        return "4 to 5"
    if years <= 9:
        return "6 to 9"
    return "10+"


INTERNSHIP_PATTERN = re.compile(
    r"internship experience|internships? (?:count|are considered|will be considered|is considered)"
    r"|including internships?|experience[^.;\n]{0,40}\binternships?\b",
    flags=re.IGNORECASE,
)
RESEARCH_PATTERN = re.compile(
    r"research experience|experience[^.;\n]{0,40}\bresearch\b",
    flags=re.IGNORECASE,
)


def mentions_internship(text: str) -> bool:
    """True when the text counts internship experience, e.g. "including internships"."""
    return isinstance(text, str) and bool(INTERNSHIP_PATTERN.search(text))


def mentions_research(text: str) -> bool:
    """True when the text asks for research experience."""
    return isinstance(text, str) and bool(RESEARCH_PATTERN.search(text))


def add_resume_signals(jobs: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
    """Add majors, internship_ok and research_asked read from description.

    Also adds years_group when min_years is already there (from
    add_text_signals). Cached by job_id like add_text_signals.
    """
    out = None
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        if set(jobs["job_id"]).issubset(set(cached["job_id"])):
            cached["majors"] = cached["majors"].apply(list)
            out = jobs.merge(cached, on="job_id", how="left")
    if out is None:
        signals = pd.DataFrame({
            "job_id": jobs["job_id"].to_numpy(),
            "majors": jobs["description"].apply(majors_mentioned).to_numpy(),
            "internship_ok": jobs["description"].apply(mentions_internship).to_numpy(),
            "research_asked": jobs["description"].apply(mentions_research).to_numpy(),
        })
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            signals.to_parquet(cache_path, index=False)
        out = jobs.merge(signals, on="job_id", how="left")
    if "min_years" in out.columns:
        out["years_group"] = out["min_years"].apply(years_group)
    return out
