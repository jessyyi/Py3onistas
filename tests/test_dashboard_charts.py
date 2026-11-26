"""Tests for dataset-backed chart inputs and chart selection."""

import pandas as pd

from src.dashboard_charts import available_charts, chart_data, valid_salary
from src.market_filters import filter_location


def make_jobs():
    return pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "state": ["NY", "NY", "CA", "TX"],
        "job_function": [["Data Analyst"], ["Data Analyst"], ["Software Engineer"], ["Accountant"]],
        "industry": [["Technology"], ["Technology"], ["Software"], ["Financial Services"]],
        "skill_labels": [["Python", "SQL"], ["Python"], ["Python"], []],
        "salary_annual": [100000, 120000, 180000, None],
        "salary_currency": ["USD", "USD", "USD", "USD"],
        "work_type": ["Full-time", "Full-time", "Part-time", "Full-time"],
        "experience_level": ["Entry", "Mid", "Mid", "Senior"],
        "listed_date": ["2024-01-05", "2024-02-01", "2024-02-14", "2024-02-28"],
    })


def test_chart_catalog_hides_unsupported_fields():
    charts = available_charts(make_jobs())
    assert "education" not in charts
    assert "salary_by_function" in charts


def test_salary_only_comparable_usd_fulltime():
    jobs = make_jobs()
    assert valid_salary(jobs).tolist() == [100000, 120000]
    assert chart_data(jobs, "salary_disclosure")["postings"].tolist() == [2, 2]


def test_charts_follow_location_selection():
    jobs = filter_location(make_jobs(), region="East")
    assert len(jobs) == 2
    data = chart_data(jobs, "skills")
    counts = dict(zip(data["category"], data["postings"]))
    assert counts == {"Python": 2, "SQL": 1}
    assert chart_data(jobs, "salary")["annual_salary_usd"].tolist() == [100000, 120000]
    assert chart_data(jobs, "timeline")["postings"].sum() == 2


def test_salary_function_chart_uses_minimum_sample():
    result = chart_data(make_jobs(), "salary_by_function")
    assert result.empty  # fewer than three comparable salary records per function
