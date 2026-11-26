"""Tests for the resume-order analysis on a tiny made-up table."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import resume_analysis as ra
from src.text_signals import (add_resume_signals, add_text_signals, majors_mentioned,
                              mentions_internship, mentions_research, years_group)


@pytest.fixture
def jobs():
    df = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6],
        "title": ["Senior Data Analyst - Remote", "Data Analyst II", "Registered Nurse (Nights)",
                  "Data Analyst", "Accountant", "Data Scientist"],
        "description": [
            "Bachelor's degree in Statistics or Computer Science. 3+ years of experience. python sql",
            "Bachelor's degree in Economics. 1 year of experience, including internships. sql excel",
            "Associate's degree in Nursing. 2 years of experience.",
            "No degree needed. python sql excel",
            "Bachelor's degree in Accounting. 5 years of experience. excel",
            "Master's degree in Statistics. Research experience. 2 years of experience. python sql",
        ],
        "job_function": [["Analyst", "IT"], ["Analyst"], ["Health Care Provider"], ["Analyst"],
                         ["Accounting/Auditing"], ["IT"]],
        "industry": [["Software"], ["Banking"], ["Hospitals"], ["Software"], ["Banking"], ["Software"]],
        "experience_level": ["Mid-Senior level", "Entry level", "Entry level", "Entry level",
                             "Mid-Senior level", "Associate"],
        "work_type": ["Full-time"] * 6,
        "salary_annual": [100_000.0, 70_000.0, 80_000.0, np.nan, 90_000.0, 130_000.0],
        "listed_date": [date(2024, 4, 1)] * 6,
    })
    df = add_text_signals(df)
    return add_resume_signals(df)


@pytest.fixture
def job_skills(jobs):
    words = ("python", "sql", "excel")
    return pd.DataFrame({"job_id": jobs["job_id"],
                         "skills": jobs["description"].apply(lambda t: [w for w in words if w in t])})


@pytest.mark.parametrize("text, majors", [
    ("Bachelor's degree in Computer Science or Statistics.", ["Computer Science", "Statistics / Mathematics"]),
    ("Bachelor's degree required. Strong business sense.", []),
    ("BS in Mechanical Engineering", ["Engineering"]),
    ("MBA preferred", ["Business"]),
    ("Proficient in MS Office", []),
])
def test_majors_mentioned(text, majors):
    assert majors_mentioned(text) == majors


def test_internship_research_and_years_group():
    assert mentions_internship("2 years of experience, including internships")
    assert not mentions_internship("This is a summer internship")
    assert mentions_research("Prior research experience is a plus")
    assert [years_group(v) for v in [1, 3, 5, 8, 12, np.nan]] == \
        ["1 year", "2 to 3", "4 to 5", "6 to 9", "10+", "Not stated"]


def test_normalize_title():
    assert ra.normalize_title("Senior Data Analyst - Remote (Req #123)") == "data analyst"
    assert ra.normalize_title("Data Analyst II") == "data analyst"
    assert ra.normalize_title("Registered Nurse (Nights)") == "registered nurse"


def test_degree_requirements_uses_two_denominators(jobs):
    table = ra.degree_requirements(jobs, by="job_function", min_n=1, top=None)
    analyst = table.loc["Analyst"]
    assert analyst["postings"] == 3
    assert math.isclose(analyst["states_degree"], 2 / 3)   # job 4 names no degree
    assert analyst["Bachelor"] == 1.0                       # among the 2 that name one


def test_major_share_counts_only_postings_naming_a_major(jobs):
    share, n = ra.major_share(jobs, by="industry", min_n=1)
    assert n["Software"] == 2                               # job 4 names no major
    assert share.loc["Software", "Statistics / Mathematics"] == 1.0


def test_years_and_grid(jobs):
    years = ra.years_by_group(jobs, by="experience_level", min_n=1)
    assert math.isclose(years.loc["Entry level", "states_years"], 2 / 3)
    grid, n = ra.education_experience_grid(jobs)
    assert n == 6 and math.isclose(grid.to_numpy().sum(), 1.0)


def test_titles_for_profile(jobs, job_skills):
    titles, info = ra.titles_for_profile(jobs, job_skills, ["Python", "SQL"], years=2,
                                         degree="Bachelor", min_postings=1)
    # job 1 asks 3 years, job 5 has only excel, job 6 needs a master's -> left out
    assert set(titles.index) == {"data analyst"}
    assert titles.loc["data analyst", "postings"] == 2     # jobs 2 and 4
    assert "Excel" in titles.loc["data analyst", "often_missing"] or "excel" in titles.loc["data analyst", "often_missing"]


def test_top_titles_by_years(jobs):
    table = ra.top_titles_by_years(jobs)
    assert set(table["years_group"]) <= set(ra.YEARS_GROUPS)
    assert "data analyst" in set(table["title_norm"])


def test_paired_dots_and_state_salary_chart(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src import market_plots as mp
    table = pd.DataFrame({"a": [0.1, 0.05], "b": [0.02, 0.08]}, index=["IT", "Sales"])
    mp.plot_paired_dots(table, ["a", "b"], ["A", "B"], "t", "s", "n",
                        counts=pd.Series({"IT": 10, "Sales": 20}), path=tmp_path / "p.png")
    mp.plot_state_tiles(pd.Series({"NY": 90.0, "CA": 100.0}), "t", "s", "n", value_format="${:.0f}k",
                        path=tmp_path / "m.png")
    assert (tmp_path / "p.png").exists() and (tmp_path / "m.png").exists()
