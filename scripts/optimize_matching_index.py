
"""Optimize the deployment index without refitting TF-IDF."""

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.matching import profile_to_text

DATA = ROOT / "data/app"


def file_hash(path):
    """Calculate a SHA-256 checksum for the supplied file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def memory_mb(matrix):
    """Calculate the memory footprint of the sparse matrix arrays in mebibytes."""
    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    ) / 1024**2


def top_k(scores, ids, k):
    """Return the top-ranked job IDs, breaking score ties by ascending job ID."""
    order = np.lexsort((ids, -scores))
    return ids[order[:k]]


def main():
    """Optimize the matching index and compare its scores and rankings with the baseline."""
    print("Loading original matching index...")

    with (DATA / "matching_metadata.json").open() as file:
        original_metadata = json.load(file)

    for name, info in original_metadata["assets"].items():
        if file_hash(DATA / name) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")

    vectorizer = joblib.load(
        DATA / "tfidf_vectorizer.joblib"
    )

    original = load_npz(
        DATA / "tfidf_matrix.npz"
    ).tocsr()

    ids = np.load(
        DATA / "matching_job_ids.npy",
        allow_pickle=False,
    )

    assert original.shape[0] == len(ids) == 119055
    assert vectorizer.norm == "l2"
    assert original.shape[1] == len(vectorizer.vocabulary_)

    print("Converting matrix to float32...")

    optimized = original.astype(np.float32)

    assert original.shape == optimized.shape
    assert original.nnz == optimized.nnz

    candidates = [
        {
            "skills": ["Python", "SQL", "Machine Learning"],
            "education": "Master's degree in Data Science",
            "experience": "Predictive modeling and analytics",
        },
        {
            "skills": ["Excel", "Accounting", "Financial Modeling"],
            "education": "Bachelor's degree in Finance",
            "experience": "Financial reporting",
        },
        {
            "skills": ["CPR", "Patient Care"],
            "education": "Bachelor's degree in Nursing",
            "experience": "Clinical healthcare experience",
        },
        {
            "skills": ["Java", "Software Engineering"],
            "education": "Computer Science",
            "experience": "Backend development",
        },
        {
            "skills": ["Marketing", "Data Analysis"],
            "education": "Bachelor's degree in Business",
            "experience": "Digital marketing",
        },
        {},
        {
            "skills": ["zzzzunknownskillzzzz"],
        },
    ]

    results = []

    for i, candidate in enumerate(candidates, 1):
        query = vectorizer.transform(
            [profile_to_text(candidate)]
        )

        start = time.perf_counter()

        baseline = cosine_similarity(
            query, original
        ).ravel()

        baseline_seconds = time.perf_counter() - start

        query32 = query.astype(np.float32)

        start = time.perf_counter()

        scores = (
            optimized @ query32.T
        ).toarray().ravel()

        optimized_seconds = time.perf_counter() - start

        error = float(
            np.max(np.abs(baseline - scores))
        )

        np.testing.assert_allclose(
            scores,
            baseline,
            rtol=1e-5,
            atol=1e-6,
        )

        baseline_top10 = top_k(baseline, ids, 10)
        optimized_top10 = top_k(scores, ids, 10)

        overlap = len(
            set(baseline_top10)
            & set(optimized_top10)
        ) / 10

        result = {
            "candidate": i,
            "max_absolute_error": error,
            "top10_overlap": overlap,
            "top10_exact_order": bool(
                np.array_equal(
                    baseline_top10,
                    optimized_top10,
                )
            ),
            "baseline_seconds": round(
                baseline_seconds, 4
            ),
            "optimized_seconds": round(
                optimized_seconds, 4
            ),
        }

        results.append(result)

        print(json.dumps(result, indent=2))

    output = DATA / "tfidf_matrix_float32.npz"

    print("Saving optimized matrix...")

    save_npz(
        output,
        optimized,
        compressed=True,
    )

    metadata = {
        "row_count": len(ids),
        "matrix_shape": list(optimized.shape),
        "matrix_nnz": int(optimized.nnz),
        "dtype": str(optimized.dtype),
        "original_memory_mb": round(
            memory_mb(original), 2
        ),
        "optimized_memory_mb": round(
            memory_mb(optimized), 2
        ),
        "original_file_mb": round(
            (DATA / "tfidf_matrix.npz").stat().st_size
            / 1024**2, 2
        ),
        "optimized_file_mb": round(
            output.stat().st_size / 1024**2, 2
        ),
        "mean_baseline_seconds": float(
            np.mean([
                r["baseline_seconds"] for r in results
            ])
        ),
        "mean_optimized_seconds": float(
            np.mean([
                r["optimized_seconds"] for r in results
            ])
        ),
        "max_absolute_error": max(
            r["max_absolute_error"] for r in results
        ),
        "min_top10_overlap": min(
            r["top10_overlap"] for r in results
        ),
        "matrix_sha256": file_hash(output),
        "original_matrix_sha256": file_hash(
            DATA / "tfidf_matrix.npz"
        ),
        "market_sha256": original_metadata["market_sha256"],
        "tests": results,
    }

    with (
        DATA / "optimized_matching_metadata.json"
    ).open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\n=== Optimization Summary ===")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
