
"""Validated and cached data access for the JobScope website."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/app"

MARKET_PATH = DATA / "market_jobs.parquet"
METADATA_PATH = DATA / "market_metadata.json"

LIST_COLUMNS = [
    "job_function",
    "industry",
    "job_industry",
    "industry_group",
    "skill_labels",
    "majors",
]


def file_hash(path):
    """Calculate a SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_labels(value):
    """Convert parquet arrays to Python lists."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def full_market_available():
    """Check whether full market resources exist."""
    return (
        MARKET_PATH.is_file()
        and METADATA_PATH.is_file()
    )


def market_version():
    """Return the current dataset fingerprint."""
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata["market_sha256"]


@st.cache_data(
    show_spinner="Loading full historical market...",
    max_entries=1,
)
def load_full_market(version, file_stamp):
    """Load and validate the compact market dataset."""

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata["market_sha256"] != version:
        raise ValueError("Dataset version mismatch.")

    if file_hash(MARKET_PATH) != version:
        raise ValueError("Market checksum mismatch.")

    jobs = pd.read_parquet(MARKET_PATH)

    required = {
        "job_id",
        "title",
        "industry_group",
        "job_function",
        "skill_labels",
        "degree",
        "min_years",
        "majors",
    }

    if not required.issubset(jobs.columns):
        missing = required - set(jobs.columns)
        raise ValueError(f"Missing fields: {missing}")

    if len(jobs) != metadata["row_count"]:
        raise ValueError("Unexpected dataset row count.")

    if jobs["job_id"].isna().any():
        raise ValueError("Missing job IDs.")

    if not jobs["job_id"].is_unique:
        raise ValueError("Duplicate job IDs.")

    for column in LIST_COLUMNS:
        if column in jobs.columns:
            jobs[column] = jobs[column].map(
                normalize_labels
            )

    # Convert nullable categorical fields for UI compatibility.
    for column in jobs.select_dtypes(
        include=["category"]
    ).columns:
        jobs[column] = jobs[column].astype(object)

    return jobs


def load_market_data():
    """Load the full dataset using version-aware caching."""

    if not full_market_available():
        raise FileNotFoundError(
            "Full market resources are unavailable."
        )

    stamp = (
        MARKET_PATH.stat().st_size,
        MARKET_PATH.stat().st_mtime_ns,
    )

    return load_full_market(market_version(), stamp)
