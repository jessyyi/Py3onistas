"""Unit tests for src/data_processing.py, run on a tiny made-up dataset."""

import math

import pandas as pd
import pytest

from src.data_processing import (
    CLEAN_COLUMNS,
    audit_clean_jobs,
    build_clean_jobs,
    check_schema,
    classify_location,
    count_same_text,
    find_content_duplicates,
    labels_to_lists,
    save_clean_jobs,
)


@pytest.fixture
def tables():
    postings = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6],
        "company_id": [10, 10, 20, None, 20, 10],
        "company_name": ["Acme", "Acme", None, "Solo", "Beta", "Acme"],
        "title": ["Data Analyst", "Data Analyst", "Nurse", "Clerk", "Engineer", "Analyst"],
        "description": ["same text", "same text", "care", "", "build", "other"],
        "location": ["New York, NY", "New York, NY", "Greater Boston",
                     "Toronto, ON", "California, United States", "United States"],
        "min_salary": [50.0, 50.0, None, None, 100000.0, None],
        "med_salary": [None, None, None, None, None, 30.0],
        "max_salary": [70.0, 70.0, None, None, 120000.0, None],
        "pay_period": ["HOURLY", "HOURLY", None, None, "YEARLY", "HOURLY"],
        "currency": ["USD", "USD", None, None, "USD", "EUR"],
        "normalized_salary": [124800.0, 124800.0, None, None, 110000.0, 62400.0],
        "remote_allowed": [1.0, 1.0, None, None, None, None],
        "formatted_experience_level": ["Entry level"] * 6,
        "formatted_work_type": ["Full-time"] * 6,
        "listed_time": [1.7e12] * 6,
    })
    return {
        "postings": postings,
        "companies": pd.DataFrame({"company_id": [10, 20], "name": ["Acme", "Beta Corp"]}),
        "company_industries": pd.DataFrame({
            "company_id": [10, 10, 20], "industry": ["Software", "Finance", "Hospitals"],
        }),
        "job_skills": pd.DataFrame({
            "job_id": [1, 1, 3, 5, 999], "skill_abr": ["ANLS", "IT", "HCPR", "ENG", "IT"],
        }),
        "job_industries": pd.DataFrame({"job_id": [1, 3], "industry_id": [4, 14]}),
        "skills_map": pd.DataFrame({
            "skill_abr": ["ANLS", "IT", "HCPR", "ENG"],
            "skill_name": ["Analyst", "Information Technology", "Health Care Provider", "Engineering"],
        }),
        "industries_map": pd.DataFrame({"industry_id": [4, 14], "industry_name": ["Software", "Hospitals"]}),
    }


def test_schema_check_passes_and_reports_missing(tables):
    assert check_schema(tables) == {}
    broken = dict(tables, job_skills=tables["job_skills"].drop(columns="skill_abr"))
    assert check_schema(broken) == {"job_skills": ["skill_abr"]}


def test_classify_location_types_and_states():
    loc = pd.Series([
        "New York, NY", "Los Angeles, California, United States", "Greater Boston",
        "United States", "Toronto, ON", "NAMER", None,
    ])
    result = classify_location(loc)
    assert result["location_type"].tolist() == [
        "city_state", "state_name", "metro_area", "country_only", "other", "other", "missing",
    ]
    assert result["state"].tolist()[:2] == ["NY", "CA"]
    assert result["state"].isna().tolist()[2:] == [True] * 5


def test_labels_to_lists_keeps_one_row_per_key():
    link = pd.DataFrame({"job_id": [1, 1, 1, 2], "label": ["b", "a", "a", "c"]})
    lists = labels_to_lists(link, "job_id", "label")
    assert lists.to_dict() == {1: ["a", "b"], 2: ["c"]}


def test_content_duplicates_ignore_empty_descriptions():
    df = pd.DataFrame({
        "company_name": ["A", "a ", "A", "A"],
        "title": ["X", "x", "X", "X"],
        "description": ["text", "text", "", ""],
        "location": ["Austin, TX"] * 4,
    })
    assert find_content_duplicates(df).tolist() == [False, True, False, False]


def test_same_job_in_other_city_is_kept_but_counted():
    df = pd.DataFrame({
        "company_name": ["A", "A", "A"],
        "title": ["X", "X", "X"],
        "description": ["text", "text", "text"],
        "location": ["Austin, TX", "Dallas, TX", "Austin, TX"],
    })
    assert find_content_duplicates(df).tolist() == [False, False, True]
    assert find_content_duplicates(df, include_location=False).tolist() == [False, True, True]
    assert count_same_text(df).tolist() == [3, 3, 3]


def test_build_clean_jobs(tables):
    clean, log = build_clean_jobs(tables)

    assert list(clean.columns) == CLEAN_COLUMNS
    assert log["rows_dropped_not_us"] == 1          # Toronto
    assert log["content_duplicates_same_location"] == 1   # job 2 repeats job 1 in New York
    assert clean["job_id"].tolist() == [1, 3, 5, 6]
    assert clean["job_id"].is_unique

    row = clean.set_index("job_id")
    assert row.loc[1, "job_function"] == ["Analyst", "Information Technology"]
    assert row.loc[1, "industry"] == ["Finance", "Software"]
    assert row.loc[6, "job_function"] == []
    assert row.loc[3, "company_name"] == "Beta Corp"   # filled from companies.csv

    # Missing salary stays NaN; non-USD salary gets no yearly value.
    assert math.isnan(row.loc[3, "salary_annual"])
    assert not row.loc[3, "salary_disclosed"]
    assert math.isnan(row.loc[6, "salary_annual"])
    assert row.loc[6, "salary_disclosed"]
    assert row.loc[1, "salary_annual"] == 124800.0


def test_audit_counts_each_function_once_per_posting(tables):
    clean, _ = build_clean_jobs(tables)
    audit = audit_clean_jobs(clean)
    coverage = audit["function_coverage"]
    assert coverage.loc["Analyst", "postings"] == 1
    assert coverage.loc["(none)", "postings"] == 1
    assert audit["salary"].loc["salary_disclosed", "value"] == 3


def test_save_writes_three_files(tables, tmp_path):
    clean, _ = build_clean_jobs(tables)
    paths = save_clean_jobs(clean, tmp_path, sample_size=2)
    assert all(p.exists() for p in paths.values())
    reloaded = pd.read_parquet(paths["parquet"])
    assert list(reloaded.loc[0, "job_function"]) == ["Analyst", "Information Technology"]
    assert len(pd.read_csv(paths["sample"])) == 2
