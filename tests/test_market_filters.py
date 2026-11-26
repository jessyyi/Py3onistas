"""Unit tests for hierarchy and filter counts."""

import pandas as pd

from src.market_filters import (
    filter_labels, filter_location, group_counts, label_counts, state_region, state_subregion
)


def fixture_jobs():
    return pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "job_function": [["Data Analyst", "Data Science"], ["Finance"], ["Software Engineer"], ["Data Analyst"]],
        "industry": [["Software Development"], ["Banking"], ["Software Development"], []],
        "state": ["NY", "TX", "CA", "NJ"],
    })


def test_counts_are_sorted_and_not_double_counted():
    jobs = fixture_jobs()
    assert label_counts(jobs, "job_function")["Data Analyst"] == 2
    assert group_counts(jobs, "job_function", "role")["Data & AI"] == 2


def test_hierarchy_filters():
    jobs = fixture_jobs()
    assert filter_labels(jobs, "job_function", group="Data & AI", kind="role").job_id.tolist() == [1, 4]
    assert filter_labels(jobs, "industry", group="Financial Services", kind="industry").job_id.tolist() == [2]
    assert filter_location(jobs, region="East").job_id.tolist() == [1, 4]
    assert filter_location(jobs, region="West", subregion="Pacific", state="CA").job_id.tolist() == [3]


def test_subregions():
    assert state_region("NY") == "East"
    assert state_subregion("TX") == "South Central"
    assert state_region("CA") == "West"
