"""Simple signals read from job description text: years of experience and degree.

These are rough, rule-based reads for market-level statistics (share of
postings asking for 3+ years, and so on). They are not meant to decide
whether one candidate qualifies; that is the job of the eligibility flags in
feature_extraction.py.
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
# must follow within 40 characters, so "100 years in business" is not caught.
YEARS_PATTERN = re.compile(
    _NUM + r"\s*(?:\+|plus)?\s*(?:(?:-|to)\s*" + _NUM + r"\s*\+?\s*)?"
    r"(?:years?|yrs?)\b[^.;\n]{0,40}?experience",
    flags=re.IGNORECASE,
)

MAX_YEARS = 20

# Short forms (B.S., MS) are matched case-sensitively and only with dots or
# with "degree"/"in" after them, so "MS Office" is not read as a master's.
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
    """Add ``min_years`` and ``degree`` columns read from ``description``.

    Reading every description takes a while, so with ``cache_path`` the two
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
