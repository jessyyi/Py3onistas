"""Tests for industry_groups.py and rq1_charts.py. I use small made-up tables and check the numbers by hand
(the comments next to the asserts show how I counted)."""

from datetime import date

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import rq1_charts as rc
from src.industry_groups import (GROUPS, STAFFING, add_industry_group, group_industries,
                                 group_table, unmapped_industries)
from src.text_signals import add_resume_signals, add_text_signals


# ---------- industry groups ----------

def test_each_name_has_one_group():
    names = [n for group in GROUPS.values() for n in group]
    assert len(names) == len(set(names))


def test_it_and_software_count_once():
    assert group_industries(["IT Services and IT Consulting", "Software Development"]) == ["CS/IT"]
    assert group_industries(np.array(["Banking", "Insurance"])) == ["Finance"]


def test_unknown_name_is_kept_and_missing_is_empty():
    assert group_industries(["Some New Industry"]) == ["Some New Industry"]
    assert group_industries(None) == []
    assert group_industries(np.nan) == []


def test_add_industry_group_keeps_raw_and_can_run_twice():
    jobs = pd.DataFrame({"industry": [["Software Development"], ["Retail Groceries", "Retail"], []]})
    once = add_industry_group(jobs)
    twice = add_industry_group(once)
    assert once["industry"].tolist() == [["CS/IT"], ["Retail"], []]
    assert twice["industry"].tolist() == once["industry"].tolist()
    assert once["industry_raw"].iloc[0] == ["Software Development"]
    assert unmapped_industries(once) == []
    assert "industry" not in jobs.columns or jobs["industry"].iloc[0] == ["Software Development"]


def test_group_table_has_every_name():
    t = group_table()
    assert len(t) == sum(len(v) for v in GROUPS.values())
    assert set(t["industry_group"]) == set(GROUPS)


# ---------- chart tables ----------

@pytest.fixture
def jobs():
    df = pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6, 7],
        "title": ["Data Analyst", "Analyst", "Nurse", "Engineer", "Accountant", "Data Scientist", "Recruiter"],
        "description": [
            "Bachelor's degree in Computer Science. 1 year of experience. python sql",
            "Master's degree in Statistics. 3 years of experience. sql excel",
            "Associate's degree in Nursing. 2 years of experience.",
            "Bachelor's degree in Engineering. 6 years of experience. python",
            "Bachelor's degree in Accounting. 4 years of experience. excel",
            "PhD in Computer Science. 5 years of experience. python sql",
            "High school diploma. 1 year of experience. excel",
        ],
        "job_function": [["Information Technology", "Analyst"], ["Analyst"], ["Health Care Provider"],
                         ["Information Technology", "Engineering"], ["Finance", "Accounting/Auditing"],
                         ["Information Technology"], ["Human Resources"]],
        "industry": [["Software Development"], ["IT Services and IT Consulting"], ["Hospitals and Health Care"],
                     ["Software Development", "IT Services and IT Consulting"], ["Banking"], ["Software Development"],
                     [STAFFING]],
        "experience_level": ["Entry level", "Associate", "Entry level", "Mid-Senior level", "Mid-Senior level",
                             "Director", "Entry level"],
        "state": ["CA", "CA", "NY", "CA", "NY", "NY", "CA"],
        "work_type": ["Full-time"] * 7,
        "salary_annual": [100_000.0, 120_000.0, 80_000.0, 150_000.0, 90_000.0, 200_000.0, 50_000.0],
        "listed_date": [date(2024, 4, 1)] * 7,
    })
    df = add_text_signals(df)
    df = add_resume_signals(df)
    return add_industry_group(df)


@pytest.fixture
def job_skills(jobs):
    words = ("python", "sql", "excel")
    return pd.DataFrame({"job_id": jobs["job_id"],
                         "skills": jobs["description"].apply(lambda t: [w for w in words if w in t])})


def test_minimum_degree_shares_add_up_and_skip_staffing(jobs):
    t = rc.minimum_degree(jobs, min_named=1)
    assert STAFFING not in t.index
    parts = t[["Below bachelor", "Bachelor", "Master", "PhD"]].sum(axis=1)
    assert np.allclose(parts, 1)
    # CS/IT: bachelor (1), master (2), bachelor (4), PhD (6) -> 2 of 4 are master or higher
    assert t.loc["CS/IT", "master_or_higher"] == pytest.approx(0.5)
    assert t.loc["CS/IT", "postings_naming_degree"] == 4


def test_years_by_level_bins(jobs):
    t = rc.years_by_level(jobs)
    assert np.allclose(t[rc.YEAR_BINS].sum(axis=1), 1)
    # entry level postings ask 1, 2 and 1 years
    assert t.loc["Entry level", "1 year"] == pytest.approx(2 / 3)
    assert t.loc["Entry level", "median_years"] == 1
    assert t.loc["Director", "6+ years"] == 0  # 5 years goes in 4 to 5
    assert t.loc["Director", "4 to 5 years"] == 1


def test_years_grid_hides_small_cells(jobs):
    med, n = rc.years_grid(jobs, min_n=1, leave_out=())
    assert med.loc["CS/IT", "Entry level"] == 1
    assert n.loc["CS/IT", "Mid-Senior level"] == 1
    med2, _ = rc.years_grid(jobs, min_n=2, leave_out=())
    assert np.isnan(med2.loc["CS/IT", "Mid-Senior level"])


def test_top_skills_and_majors(jobs, job_skills):
    skills = rc.top_skills(jobs, job_skills, n_industries=3, min_share=0)
    cs = skills[skills["industry"] == "CS/IT"]
    assert cs.iloc[0]["skill"] == "python"  # 3 of 4 CS/IT postings
    assert cs.iloc[0]["share"] == pytest.approx(0.75)
    assert STAFFING not in set(skills["industry"])
    majors = rc.top_majors(jobs, n_industries=3, min_named=1)
    assert majors[majors["industry"] == "CS/IT"].iloc[0]["major"] == "Computer Science"


def test_function_pairs_overlap(jobs):
    pairs = rc.function_pairs(jobs, min_postings=1)
    row = pairs[(pairs["function_a"] == "Analyst") & (pairs["function_b"] == "Information Technology")].iloc[0]
    # Analyst has 2 postings, IT has 3, they share 1 -> 1 / 2
    assert row["shared_postings"] == 1
    assert row["overlap"] == pytest.approx(0.5)


def test_strong_pairs_leave_out_and_threshold():
    pairs = pd.DataFrame({"function_a": ["Management", "Sales", "IT"],
                          "function_b": ["Manufacturing", "Business Development", "Design"],
                          "shared_postings": [10, 9, 2], "overlap": [0.9, 0.92, 0.2]})
    strong = rc.strong_pairs(pairs, min_overlap=0.5)
    assert strong["function_a"].tolist() == ["Sales"]


def test_salary_grid(jobs):
    med, n, states, sizes = rc.salary_grid(jobs, n_industries=3, n_states=2, min_n=1)
    assert med.loc["CS/IT", "CA"] == 120_000  # 100k, 120k, 150k
    assert n.loc["CS/IT", "CA"] == 3
    assert states["CA"] == 4  # the staffing posting still counts for the state size
    med2, *_ = rc.salary_grid(jobs, n_industries=3, n_states=2, min_n=2)
    assert np.isnan(med2.loc["Finance", "NY"])


def test_plots_run(jobs, job_skills, tmp_path):
    rc.plot_minimum_degree(rc.minimum_degree(jobs, min_named=1), tmp_path / "a.png")
    rc.plot_years_by_level(rc.years_by_level(jobs), tmp_path / "b.png")
    rc.plot_skills(rc.top_skills(jobs, job_skills, n_industries=2, min_share=0), tmp_path / "c.png")
    med, n = rc.years_grid(jobs, min_n=1, leave_out=())
    rc.plot_years_grid(med, tmp_path / "d.png")
    med, n, states, sizes = rc.salary_grid(jobs, n_industries=3, n_states=2, min_n=1)
    rc.plot_salary_grid(med, states, sizes, tmp_path / "e.png")
    assert len(list(tmp_path.glob("*.png"))) == 5
