
"""Integration tests for NLP extraction and hybrid job matching."""

import pandas as pd
import pytest

from src.feature_extraction import extract_skills
from src.matching import TextMatcher
from src.recommendation import recommend_top_k


@pytest.fixture
def candidate():
    return {
        "candidate_id": "test_candidate",
        "skills": ["Python", "SQL", "Machine Learning"],
        "education": "Master's in Data Science",
        "experience": "Data analysis and statistical modeling",
    }


@pytest.fixture
def sample_jobs():
    return pd.DataFrame({
        "job_id": ["1", "2", "3", "4"],
        "title": [
            "Data Scientist",
            "Database Analyst",
            "Accountant",
            "Empty Description",
        ],
        "description": [
            "Python SQL machine learning statistical modeling",
            "SQL database management and reporting",
            "Accounting bookkeeping and financial reporting",
            "",
        ],
    })


def test_nlp_output_format(sample_jobs):
    """Check that extracted skills are normalized lists."""
    skills = sample_jobs["description"].apply(extract_skills)

    assert skills.apply(lambda x: isinstance(x, list)).all()
    assert "Python" in skills.iloc[0]
    assert "SQL" in skills.iloc[0]
    assert skills.iloc[3] == []


def test_end_to_end_hybrid_matching(sample_jobs, candidate):
    """Run NLP extraction through hybrid Top-K ranking."""
    jobs = sample_jobs.copy()
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)

    matcher = TextMatcher().fit(jobs)

    results = recommend_top_k(
        candidate,
        jobs,
        k=3,
        matcher=matcher,
        method="hybrid",
        skills_column="skill_labels",
        text_weight=0.5,
    )

    assert len(results) == 3
    assert results["job_id"].is_unique
    assert results["hybrid_score"].notna().all()
    assert results["hybrid_score"].is_monotonic_decreasing
    assert results.iloc[0]["job_id"] == "1"


def test_empty_description_integration(sample_jobs, candidate):
    """Ensure empty descriptions do not break the pipeline."""
    jobs = sample_jobs.copy()
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)

    matcher = TextMatcher().fit(jobs)
    results = recommend_top_k(
        candidate,
        jobs,
        k=4,
        matcher=matcher,
        method="hybrid",
        skills_column="skill_labels",
    )

    empty_job = results.loc[results["job_id"] == "4"]

    assert len(empty_job) == 1
    assert pd.isna(empty_job.iloc[0]["skill_score"])


def test_real_cleaned_sample(candidate):
    """Check integration on the shared 500-job sample."""
    jobs = pd.read_csv("data/sample/sample_jobs.csv")

    assert len(jobs) == 500
    assert jobs["job_id"].is_unique

    jobs["description"] = jobs["description"].fillna("")
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)

    matcher = TextMatcher().fit(jobs)

    results = recommend_top_k(
        candidate,
        jobs,
        k=10,
        matcher=matcher,
        method="hybrid",
        skills_column="skill_labels",
    )

    assert len(results) == 10
    assert results["job_id"].is_unique
    assert results["hybrid_score"].is_monotonic_decreasing
