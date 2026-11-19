"""Checks for text matching and Top-K recommendations."""

import pandas as pd
import pytest

from src.matching import TextMatcher, skill_coverage
from src.recommendation import recommend_top_k, score_jobs


@pytest.fixture
def example_jobs():
    return pd.DataFrame({
        "job_id": ["1", "2", "3"],
        "title": ["Python Developer", "Accountant", "Nurse"],
        "description": [
            "Python programming software development",
            "Accounting bookkeeping financial reporting",
            "Nursing patient care hospital",
        ],
    })


def test_matching_ranks_related_job_first(example_jobs):
    matcher = TextMatcher().fit(example_jobs)
    candidate = {"skills": ["Python", "programming"]}

    result = recommend_top_k(
        candidate, example_jobs, k=2, matcher=matcher
    )

    assert result["job_id"].tolist() == ["1", "2"]
    assert result.iloc[0]["text_score"] > result.iloc[1]["text_score"]
    assert "text_score" not in example_jobs.columns


def test_unknown_words_produce_zero_scores(example_jobs):
    matcher = TextMatcher().fit(example_jobs)
    scores = matcher.score({"skills": ["zzzxxyy"]})

    assert (scores == 0).all()


def test_changed_job_order_is_rejected(example_jobs):
    matcher = TextMatcher().fit(example_jobs)

    with pytest.raises(ValueError, match="jobs must match"):
        recommend_top_k(
            {"skills": ["Python"]},
            example_jobs.iloc[::-1],
            matcher=matcher,
        )


def test_duplicate_job_ids_are_rejected(example_jobs):
    example_jobs.loc[1, "job_id"] = "1"

    with pytest.raises(ValueError, match="unique"):
        TextMatcher().fit(example_jobs)


def test_skill_coverage_normalizes_and_deduplicates():
    result = skill_coverage([" Python ", "SQL"], ["python", "Python", "AWS"])
    assert result["skill_score"] == 0.5
    assert result["matched_skills"] == ["python"]
    assert result["unverified_skills"] == ["aws"]
    assert skill_coverage(["Python"], [])["skill_score"] is None


def test_hybrid_fallback_and_skill_only_exclusion(example_jobs):
    example_jobs["skills"] = [["python", "aws"], [], ["nursing"]]
    matcher = TextMatcher().fit(example_jobs)
    candidate = {"skills": ["python"]}
    scores = score_jobs(candidate, example_jobs, matcher=matcher,
                        skills_column="skills", text_weight=0.6)
    assert scores.loc[0, "hybrid_score"] == pytest.approx(
        0.6 * scores.loc[0, "text_score"] + 0.4 * 0.5)
    assert pd.isna(scores.loc[1, "skill_score"])
    assert scores.loc[1, "hybrid_score"] == scores.loc[1, "text_score"]
    assert not scores.loc[1, "hybrid_uses_skills"]
    ranked = recommend_top_k(candidate, example_jobs, matcher=matcher,
                             method="skill", skills_column="skills")
    assert ranked["job_id"].tolist() == ["1", "3"]


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_invalid_hybrid_weight(example_jobs, weight):
    matcher = TextMatcher().fit(example_jobs)
    with pytest.raises(ValueError, match="text_weight"):
        score_jobs({}, example_jobs, matcher=matcher, text_weight=weight)
