
"""Build compact, full-dataset market features for JobScope."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_extraction import extract_skills
from src.industry_groups import group_industries
from src.text_signals import (
    min_years_required,
    lowest_degree_mentioned,
    majors_mentioned,
    years_group,
    mentions_internship,
    mentions_research,
)

SOURCE = ROOT / "data/processed/clean_jobs.parquet"
OUTPUT = ROOT / "data/app"

MARKET_COLUMNS = [
    "job_id",
    "title",
    "company_name",
    "location",
    "state",
    "location_type",
    "job_function",
    "industry",
    "job_industry",
    "salary_annual",
    "salary_currency",
    "salary_disclosed",
    "remote_allowed",
    "experience_level",
    "work_type",
    "listed_date",
]


def file_hash(path):
    """Return a SHA-256 fingerprint."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_labels(value):
    """Preserve multi-label values as lists."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def extract_features(descriptions):
    """Extract existing research signals offline."""
    records = []
    total = len(descriptions)
    start = time.perf_counter()

    for i, text in enumerate(descriptions, start=1):
        years = min_years_required(text)

        records.append({
            "skill_labels": extract_skills(text),
            "min_years": years,
            "degree": lowest_degree_mentioned(text),
            "majors": majors_mentioned(text),
            "years_group": years_group(years),
            "internship_ok": mentions_internship(text),
            "research_asked": mentions_research(text),
        })

        if i % 1000 == 0 or i == total:
            elapsed = time.perf_counter() - start
            print(
                f"Processed {i:,}/{total:,} "
                f"({i / total:.1%}) | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True,
            )

    return pd.DataFrame(records)


def build_market_data(limit=None):
    """Build and validate compact market features and dataset metadata offline."""
    start = time.perf_counter()

    print("Loading source dataset...")

    jobs = pd.read_parquet(
        SOURCE,
        columns=MARKET_COLUMNS + ["description"],
    )

    if limit is not None:
        jobs = jobs.head(limit)

    jobs = jobs.reset_index(drop=True)

    if jobs["job_id"].isna().any():
        raise ValueError("Missing job IDs detected.")

    if jobs["job_id"].duplicated().any():
        raise ValueError("Duplicate job IDs detected.")

    print(f"Source rows: {len(jobs):,}")

    market = jobs[MARKET_COLUMNS].copy()

    print("Applying industry taxonomy...")

    market["industry_group"] = (
        market["industry"]
        .map(normalize_labels)
        .map(group_industries)
    )

    for column in ["industry", "job_function"]:
        market[column] = market[column].map(
            normalize_labels
        )

    print("Extracting NLP features...")

    features = extract_features(jobs["description"])

    market = pd.concat(
        [market, features],
        axis=1,
    )

    print("Optimizing data types...")

    category_columns = [
        "state",
        "location_type",
        "experience_level",
        "work_type",
        "degree",
        "years_group",
        "salary_currency",
    ]

    for column in category_columns:
        market[column] = market[column].astype(
            "category"
        )

    market["min_years"] = market["min_years"].astype(
        "Float32"
    )

    market["listed_date"] = pd.to_datetime(
        market["listed_date"],
        errors="raise",
        utc=True,
    )

    del jobs, features

    assert market["job_id"].is_unique
    assert "description" not in market.columns

    if limit is None:
        assert len(market) == 119055

    OUTPUT.mkdir(parents=True, exist_ok=True)

    suffix = "_test" if limit is not None else ""

    market_path = OUTPUT / f"market_jobs{suffix}.parquet"

    print("Saving market dataset...")

    market.to_parquet(
        market_path,
        index=False,
        compression="zstd",
    )

    elapsed = time.perf_counter() - start

    metadata = {
        "dataset": "JobScope historical job postings",
        "mode": "test" if limit is not None else "full",
        "row_count": len(market),
        "unique_job_ids": int(
            market["job_id"].nunique()
        ),
        "date_start": str(market["listed_date"].min()),
        "date_end": str(market["listed_date"].max()),
        "columns": market.columns.tolist(),
        "source_sha256": file_hash(SOURCE),
        "skill_dictionary_sha256": file_hash(
            ROOT / "data/reference/skill_dictionary.json"
        ),
        "feature_code_sha256": {
            name: file_hash(ROOT / "src" / name)
            for name in [
                "feature_extraction.py",
                "text_signals.py",
                "industry_groups.py",
            ]
        },
        "market_sha256": file_hash(market_path),
        "file_size_mb": round(
            market_path.stat().st_size / 1024**2, 2
        ),
        "memory_mb": round(
            market.memory_usage(deep=True).sum()
            / 1024**2, 2
        ),
        "build_seconds": round(elapsed, 2),
        "skill_coverage_rate": float(
            market["skill_labels"].map(bool).mean()
        ),
        "experience_signal_rate": float(
            market["min_years"].notna().mean()
        ),
        "degree_signal_rate": float(
            market["degree"].ne("None stated").mean()
        ),
        "major_signal_rate": float(
            market["majors"].map(bool).mean()
        ),
    }

    metadata_path = OUTPUT / f"market_metadata{suffix}.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n=== Build Summary ===")
    print(json.dumps(metadata, indent=2))

    print("\nMarket dataset:", market_path)
    print("Metadata:", metadata_path)

    return market


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build a small test dataset first.",
    )

    args = parser.parse_args()

    build_market_data(limit=args.limit)
