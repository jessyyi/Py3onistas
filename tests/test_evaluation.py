
"""Unit tests for human relevance evaluation."""

import pandas as pd
import pytest

from src.evaluation import (
    precision_at_k,
    ndcg_at_k,
    inter_rater_agreement,
    evaluate_ranked_pool,
)


def test_precision_at_k():
    assert precision_at_k([2, 1, 2, 0, 2], k=5) == 0.6


def test_ndcg_perfect_ranking():
    labels = [2, 2, 1, 0, 0]
    assert ndcg_at_k(labels, labels, k=5) == pytest.approx(1.0)


def test_ndcg_worse_ranking():
    pool = [2, 2, 1, 0, 0]
    reversed_ranking = [0, 0, 1, 2, 2]

    assert ndcg_at_k(reversed_ranking, pool, k=5) < 1.0


def test_inter_rater_agreement():
    jessy = pd.DataFrame({
        "candidate_id": ["c1", "c1", "c1"],
        "job_id": [1, 2, 3],
        "relevance_score": [2, 1, 0],
    })
    aileen = pd.DataFrame({
        "candidate_id": ["c1", "c1", "c1"],
        "job_id": [3, 1, 2],
        "relevance_score": [0, 2, 1],
    })

    result = inter_rater_agreement(jessy, aileen)

    assert result["n_pairs"] == 3
    assert result["exact_agreement"] == 1.0
    assert result["weighted_kappa"] == pytest.approx(1.0)


def test_evaluate_ranked_pool():
    ranking = pd.DataFrame({"job_id": [1, 2, 3, 4, 5]})
    labels = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5],
        "relevance_score": [2, 2, 1, 0, 0],
    })

    result = evaluate_ranked_pool(ranking, labels, k=5)

    assert result["precision_at_k"] == 0.4
    assert result["ndcg_at_k"] == pytest.approx(1.0)


def test_missing_labels_rejected():
    ranking = pd.DataFrame({"job_id": [1, 2, 3]})
    labels = pd.DataFrame({
        "job_id": [1, 2],
        "relevance_score": [2, 1],
    })

    with pytest.raises(ValueError, match="must match"):
        evaluate_ranked_pool(ranking, labels, k=2)
