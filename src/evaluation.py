
"""Human relevance evaluation for JobScope recommendations."""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def validate_labels(labels):
    """Validate relevance labels in the range 0, 1, 2."""
    values = np.asarray(labels)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Labels must be a nonempty 1D sequence.")

    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Relevance labels must be 0, 1, or 2.")

    return values.astype(int)


def precision_at_k(ranked_labels, k=5):
    """Calculate Precision@K using relevance score 2 as positive."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    labels = validate_labels(ranked_labels)

    if len(labels) < k:
        raise ValueError("Ranking contains fewer than k judged jobs.")

    return float(np.mean(labels[:k] == 2))


def ndcg_at_k(ranked_labels, all_pool_labels, k=5):
    """Calculate graded NDCG@K against the fully judged evaluation pool."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    ranked = validate_labels(ranked_labels)
    pool = validate_labels(all_pool_labels)

    if len(ranked) < k or len(pool) < k:
        raise ValueError("At least k judged jobs are required.")

    def dcg(labels):
        gains = np.power(2.0, labels) - 1.0
        discounts = np.log2(np.arange(len(labels)) + 2)
        return float(np.sum(gains / discounts))

    actual = dcg(ranked[:k])
    ideal = dcg(np.sort(pool)[::-1][:k])

    return actual / ideal if ideal > 0 else 0.0


def inter_rater_agreement(jessy_labels, aileen_labels):
    """Calculate agreement on identical candidate-job pairs."""
    keys = ["candidate_id", "job_id"]
    required = set(keys + ["relevance_score"])

    for name, frame in [
        ("Jessy", jessy_labels),
        ("Aileen", aileen_labels),
    ]:
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} annotation columns are missing.")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate pairs.")
        validate_labels(frame["relevance_score"])

    merged = jessy_labels[keys + ["relevance_score"]].merge(
        aileen_labels[keys + ["relevance_score"]],
        on=keys,
        how="outer",
        suffixes=("_jessy", "_aileen"),
        indicator=True,
        validate="one_to_one",
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError("Annotators must score identical pairs.")

    a = merged["relevance_score_jessy"].astype(int)
    b = merged["relevance_score_aileen"].astype(int)

    return {
        "n_pairs": len(merged),
        "exact_agreement": float((a == b).mean()),
        "weighted_kappa": float(
            cohen_kappa_score(
                a, b, labels=[0, 1, 2], weights="quadratic"
            )
        ),
    }


def evaluate_ranked_pool(ranked_jobs, labels, k=5):
    """Evaluate one ranking over a completely judged candidate pool."""
    if ranked_jobs["job_id"].duplicated().any():
        raise ValueError("Ranking contains duplicate job IDs.")

    if labels["job_id"].duplicated().any():
        raise ValueError("Labels contain duplicate job IDs.")

    if ranked_jobs.empty or labels.empty:
        raise ValueError("Ranking and labels must not be empty.")

    ranked_ids = set(ranked_jobs["job_id"])
    label_ids = set(labels["job_id"])

    if ranked_ids != label_ids:
        raise ValueError("Ranking and judged pool must match exactly.")

    merged = ranked_jobs[["job_id"]].merge(
        labels[["job_id", "relevance_score"]],
        on="job_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    ranked_labels = validate_labels(merged["relevance_score"])
    pool_labels = validate_labels(labels["relevance_score"])

    return {
        "precision_at_k": precision_at_k(ranked_labels, k),
        "ndcg_at_k": ndcg_at_k(ranked_labels, pool_labels, k),
    }
