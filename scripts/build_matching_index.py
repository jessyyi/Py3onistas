
"""Build a reusable offline TF-IDF index for JobScope."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching import TextMatcher, profile_to_text

SOURCE = ROOT / "data/processed/clean_jobs.parquet"
OUTPUT = ROOT / "data/app"


def file_hash(path):
    """Return a SHA-256 fingerprint."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_size_mb(path):
    """Return the file size in mebibytes, rounded to two decimal places."""
    return round(path.stat().st_size / 1024**2, 2)


def build_matching_index(limit=None):
    """Fit, persist, and validate the reusable offline TF-IDF matching index."""
    start = time.perf_counter()

    suffix = "_test" if limit is not None else ""

    market_path = OUTPUT / f"market_jobs{suffix}.parquet"
    market_metadata_path = (
        OUTPUT / f"market_metadata{suffix}.json"
    )

    with open(market_metadata_path, encoding="utf-8") as file:
        market_metadata = json.load(file)

    print("Validating source data...")

    source_hash = file_hash(SOURCE)

    if source_hash != market_metadata["source_sha256"]:
        raise ValueError("Source dataset version mismatch.")

    if file_hash(market_path) != market_metadata["market_sha256"]:
        raise ValueError("Market dataset checksum mismatch.")

    print("Loading job descriptions...")

    jobs = pd.read_parquet(
        SOURCE,
        columns=["job_id", "title", "description"],
    )

    if limit is not None:
        jobs = jobs.head(limit)

    jobs = jobs.reset_index(drop=True)

    market = pd.read_parquet(market_path)

    if not jobs["job_id"].is_unique:
        raise ValueError("Duplicate job IDs detected.")

    if not market["job_id"].is_unique:
        raise ValueError("Duplicate market job IDs detected.")

    if len(jobs) != len(market):
        raise ValueError("Dataset row counts do not match.")

    if not np.array_equal(
        jobs["job_id"].to_numpy(),
        market["job_id"].to_numpy(),
    ):
        raise ValueError("Job ID order mismatch.")

    print(f"Jobs: {len(jobs):,}")

    print("Fitting the original TextMatcher...")

    fit_start = time.perf_counter()

    matcher = TextMatcher().fit(jobs)

    vectorizer = matcher.vectorizer
    matrix = matcher.job_vectors.tocsr()

    fit_seconds = time.perf_counter() - fit_start

    print(f"TF-IDF shape: {matrix.shape}")
    print(f"Nonzero values: {matrix.nnz:,}")
    print(f"Fit time: {fit_seconds:.2f}s")

    # Keep structured metadata without full descriptions.
    matching_jobs = market.copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    jobs_path = OUTPUT / f"matching_jobs{suffix}.parquet"
    vectorizer_path = OUTPUT / f"tfidf_vectorizer{suffix}.joblib"
    matrix_path = OUTPUT / f"tfidf_matrix{suffix}.npz"
    ids_path = OUTPUT / f"matching_job_ids{suffix}.npy"
    metadata_path = OUTPUT / f"matching_metadata{suffix}.json"

    print("Saving matching resources...")

    matching_jobs.to_parquet(
        jobs_path,
        index=False,
        compression="zstd",
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3,
    )

    save_npz(
        matrix_path,
        matrix,
        compressed=True,
    )

    np.save(
        ids_path,
        jobs["job_id"].to_numpy(),
        allow_pickle=False,
    )

    sparse_memory_mb = (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    ) / 1024**2

    # Verify the saved model against the original matcher.
    candidates = [
        {
            "skills": ["Python", "SQL", "Machine Learning"],
            "education": "Master's degree in Data Science",
            "experience": "Data analysis and predictive modeling",
        },
        {
            "skills": ["Excel", "Accounting", "Financial Modeling"],
            "education": "Bachelor's degree in Finance",
            "experience": "Financial reporting and analysis",
        },
        {
            "skills": ["CPR", "Patient Care"],
            "education": "Bachelor's degree in Nursing",
            "experience": "Clinical healthcare experience",
        },
    ]

    print("Validating persisted model...")

    expected_scores = [
        matcher.score(candidate)
        for candidate in candidates
    ]

    # Release the original descriptions and fitted job table.
    del matcher, jobs, market, matching_jobs, matrix, vectorizer

    loaded_vectorizer = joblib.load(vectorizer_path)
    loaded_matrix = load_npz(matrix_path)
    loaded_ids = np.load(ids_path, allow_pickle=False)

    saved_jobs = pd.read_parquet(
        jobs_path,
        columns=["job_id"],
    )

    assert loaded_matrix.shape[0] == len(loaded_ids)
    assert np.array_equal(
        loaded_ids,
        saved_jobs["job_id"].to_numpy(),
    )

    score_times = []
    max_errors = []

    for candidate, expected in zip(
        candidates, expected_scores
    ):
        score_start = time.perf_counter()

        candidate_vector = loaded_vectorizer.transform(
            [profile_to_text(candidate)]
        )

        actual = cosine_similarity(
            candidate_vector,
            loaded_matrix,
        ).ravel()

        score_times.append(
            time.perf_counter() - score_start
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
        )

        max_errors.append(
            float(np.max(np.abs(actual - expected)))
        )

    assets = {
        path.name: {
            "size_mb": file_size_mb(path),
            "sha256": file_hash(path),
        }
        for path in [
            jobs_path,
            vectorizer_path,
            matrix_path,
            ids_path,
        ]
    }

    metadata = {
        "mode": "test" if limit is not None else "full",
        "row_count": len(loaded_ids),
        "vocabulary_size": loaded_matrix.shape[1],
        "matrix_shape": list(loaded_matrix.shape),
        "matrix_nnz": int(loaded_matrix.nnz),
        "matrix_dtype": str(loaded_matrix.dtype),
        "sparse_memory_mb": round(sparse_memory_mb, 2),
        "source_sha256": source_hash,
        "market_sha256": market_metadata["market_sha256"],
        "matching_code_sha256": file_hash(
            ROOT / "src/matching.py"
        ),
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "fit_seconds": round(fit_seconds, 2),
        "build_seconds": round(
            time.perf_counter() - start, 2
        ),
        "mean_text_scoring_seconds": round(
            float(np.mean(score_times)), 4
        ),
        "max_score_difference": max(max_errors),
        "equivalence_test_passed": True,
        "assets": assets,
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n=== Matching Index Summary ===")
    print(json.dumps(metadata, indent=2))

    print("\nAll equivalence checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build a small matching index first.",
    )

    args = parser.parse_args()

    build_matching_index(limit=args.limit)
