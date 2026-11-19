"""Unit tests for market_analysis.py, text_signals.py and market_plots.py,
run on a tiny made-up table."""

import math
from datetime import date

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src import market_analysis as ma
from src import market_plots as mp
from src.text_signals import add_text_signals, lowest_degree_mentioned, min_years_required


@pytest.fixture
def jobs():
    return pd.DataFrame({
        "job_id": [1, 2, 3, 4, 5, 6],
        "company_id": [10, 10, 20, 30, 30, 30],
        "company_name": ["A", "A", "B", "C", "C", "C"],
        "description": ["python sql. 3+ years of experience. Bachelor's degree",
                        "sql, 1 year of experience",
                        "excel",
                        "",
                        "python excel. Master's degree. 5 years of experience",
                        "python sql"],
        "job_function": [["IT", "Analyst"], ["IT"], ["Sales"], [], ["Analyst"], ["IT"]],
        "industry": [["Software"], ["Software"], ["Retail"], ["Staffing and Recruiting"], [], ["Software"]],
        "state": ["NY", "CA", "CA", None, "TX", "NY"],
        "listed_date": [date(2024, 4, 1), date(2022, 6, 1), date(2024, 4, 1),
                        date(2024, 4, 1), date(2024, 4, 1), date(2023, 1, 5)],
        "work_type": ["Full-time"] * 2 + ["Part-time"] + ["Full-time"] * 3,
        "experience_level": ["Entry level", "Entry level", "Associate", None, "Mid-Senior level", "Entry level"],
        "remote_allowed": [True, False, False, False, True, False],
        "salary_min": [90_000.0, 70_000.0, 20_000.0, np.nan, 3_000.0, 100_000.0],
        "salary_max": [110_000.0, 90_000.0, 40_000.0, np.nan, 7_000.0, 200_000.0],
        "salary_annual": [100_000.0, 80_000.0, 30_000.0, np.nan, 5_000.0, 150_000.0],
        "salary_disclosed": [True, True, True, False, True, True],
    })


@pytest.fixture
def job_skills(jobs):
    def extractor(text):
        return [w for w in ("python", "sql", "excel") if w in text]
    return ma.extract_skills_for_jobs(jobs, extractor)


# --- skills -----------------------------------------------------------------

def test_extract_skills_uses_cache(jobs, tmp_path):
    calls = []

    def counting(text):
        calls.append(text)
        return [w for w in ("python", "sql") if w in text]

    path = tmp_path / "skills.parquet"
    first = ma.extract_skills_for_jobs(jobs, counting, cache_path=path)
    second = ma.extract_skills_for_jobs(jobs, counting, cache_path=path)
    assert len(calls) == 6
    assert second["skills"].tolist() == first["skills"].tolist()


def test_skill_share_counts_each_function(jobs, job_skills):
    share, postings = ma.skill_share_by_function(jobs, job_skills, min_postings=1, top_functions=None)
    assert postings.to_dict() == {"IT": 3, "Analyst": 2, "Sales": 1}
    assert share.loc["IT", "sql"] == 1.0
    assert share.loc["Analyst", "sql"] == 0.5


def test_signature_skills_lift(jobs, job_skills):
    table = ma.signature_skills(jobs, job_skills, min_postings=1, min_share=0.0)
    sales = table[table["job_function"] == "Sales"].set_index("skills")
    # excel is in 1 of 1 Sales postings and 2 of 6 postings overall -> lift 3
    assert math.isclose(sales.loc["excel", "lift"], 3.0)


def test_skill_pairs(job_skills):
    pairs = ma.skill_pairs(job_skills, min_count=1)
    row = pairs[(pairs["skill_a"] == "python") & (pairs["skill_b"] == "sql")].iloc[0]
    assert row["together"] == 2
    assert math.isclose(row["jaccard"], 2 / 4)


def test_skill_salary_premium_compares_within_function(jobs, job_skills):
    table = ma.skill_salary_premium(jobs, job_skills, skills=["python"], min_n=1)
    # IT full-time valid salaries: python 100k and 150k (median 125k), no python 80k
    assert math.isclose(table.loc["python", "premium"], 125_000 / 80_000 - 1)
    assert table.loc["python", "functions_compared"] == 1


# --- market structure -----------------------------------------------------------

def test_function_cooccurrence(jobs):
    share = ma.function_cooccurrence(jobs)
    assert share.loc["Analyst", "IT"] == 0.5        # 1 of 2 Analyst postings is also IT
    assert math.isclose(share.loc["IT", "Analyst"], 1 / 3)
    assert np.isnan(share.loc["IT", "IT"])


def test_crosstab_share(jobs):
    share, rows = ma.crosstab_share(jobs, "industry", "job_function")
    assert rows["Software"] == 3
    assert share.loc["Software", "IT"] == 1.0


def test_company_concentration(jobs):
    table, info = ma.company_concentration(jobs, top=2)
    assert table.index[0] == "C"
    assert math.isclose(info["staffing_share"], 1 / 6)


def test_count_rate_and_missing(jobs):
    assert ma.count_by_group(jobs, "industry").loc["Software", "postings"] == 3
    rates = ma.rate_by_group(jobs, "state", min_n=1)
    assert rates.loc["CA", "postings"] == 2
    missing = ma.missing_by_group(jobs.assign(group="all"), "group", "job_function")
    assert missing.loc["all", "missing"] == 1


# --- salary -------------------------------------------------------------------

def test_salary_ranges_filters(jobs):
    summary, info = ma.salary_ranges(jobs, by="job_function", min_n=1)
    assert info == {"with_salary": 5, "after_work_type_filter": 4, "outside_bounds": 1}
    assert summary.loc["IT", "median"] == 100_000


def test_range_width_and_remote_gap(jobs):
    width = ma.range_width(jobs, by="job_function", min_n=1)
    assert width.loc["IT", "n"] == 3
    gap = ma.remote_pay_gap(jobs, by="job_function", min_n=1)
    # IT: remote 100k vs on-site median of 80k and 150k = 115k
    assert math.isclose(gap.loc["IT", "gap"], 100_000 / 115_000 - 1)


def test_pay_transparency_flag_uses_listing_date(jobs):
    flagged = ma.add_pay_transparency_flag(jobs)["transparency_law"].tolist()
    # NY 2024 yes; CA 2022 no (law from 2023); CA 2024 yes; no state -> NaN; TX no; NY Jan 2023 no
    assert flagged[0] is True and flagged[1] is False and flagged[2] is True
    assert pd.isna(flagged[3]) and flagged[4] is False and flagged[5] is False


# --- text signals ---------------------------------------------------------------

@pytest.mark.parametrize("text, years", [
    ("3+ years of experience", 3), ("3-5 years experience in sales", 3),
    ("two years of relevant work experience", 2), ("100 years in business", None),
])
def test_min_years_required(text, years):
    result = min_years_required(text)
    assert (np.isnan(result) if years is None else result == years)


@pytest.mark.parametrize("text, degree", [
    ("Proficient in MS Office", "None stated"),
    ("Bachelor's degree required, Master's degree preferred", "Bachelor"),
    ("M.S. or PhD", "Master"), ("Ph.D. in statistics", "PhD"),
])
def test_lowest_degree_mentioned(text, degree):
    assert lowest_degree_mentioned(text) == degree


def test_experience_paradox_and_mix(jobs):
    jobs = add_text_signals(jobs)
    table = ma.experience_paradox(jobs, jobs["min_years"])
    assert table.loc["Entry level", "states_years"] == 2
    assert table.loc["Entry level", "share_3plus"] == 0.5
    mix = ma.mix_by_group(jobs, "job_function", jobs["degree"], order=["None stated", "Bachelor", "Master"], min_n=1)
    assert math.isclose(mix.loc["Analyst"].sum(), 1.0)


# --- competition and details ----------------------------------------------------

def test_posting_details_and_competition(jobs):
    tables = {
        "postings": pd.DataFrame({"job_id": jobs["job_id"], "views": [10, 20, 0, 5, 40, 30],
                                  "applies": [1, 2, 0, np.nan, 4, 3]}),
        "companies": pd.DataFrame({"company_id": [10, 20, 30], "company_size": [7, 2, 4]}),
    }
    detailed = ma.add_posting_details(jobs, tables)
    assert detailed.loc[0, "company_size"] == 7
    table = ma.competition_by_group(detailed, "job_function", min_n=1)
    assert math.isclose(table.loc["IT", "applies_per_100_views"], 6 / 60 * 100)


# --- charts ---------------------------------------------------------------------

def test_all_charts_save(jobs, job_skills, tmp_path):
    jobs = add_text_signals(jobs)
    rates = ma.rate_by_group(jobs, "state", min_n=1)
    mp.plot_lollipop(rates["rate"], "t", "s", "n", top=1, bottom=1, average=0.5,
                     counts=rates["postings"], path=tmp_path / "1.png")
    mp.plot_lollipop(rates["rate"], "t", "s", "n", sort=False, colors=[mp.BLUE, mp.LIGHT, mp.RED][:len(rates)],
                     label_all=True, path=tmp_path / "2.png")
    premium = ma.skill_salary_premium(jobs, job_skills, skills=["python", "sql"], min_n=1)
    mp.plot_diverging_lollipop(premium["premium"], "t", "s", "n", counts=premium["postings_with_skill"],
                               path=tmp_path / "3.png")
    summary, _ = ma.salary_ranges(jobs, min_n=1)
    mp.plot_ranges(summary, "t", "s", "n", path=tmp_path / "4.png")
    gap = ma.remote_pay_gap(jobs, min_n=1)
    mp.plot_dumbbell(gap, "onsite_median", "remote_median", "On-site", "Remote", "t", "s", "n", tmp_path / "5.png")
    share, postings = ma.skill_share_by_function(jobs, job_skills, min_postings=1)
    mp.plot_bubble_matrix(share, "t", "s", "n", highlight=share > 0.5, row_counts=postings, path=tmp_path / "6.png")
    mix = ma.mix_by_group(jobs, "job_function", jobs["degree"], order=["None stated", "Bachelor", "Master"], min_n=1)
    mp.plot_stacked_mix(mix, "t", "s", "n", tmp_path / "7.png")
    mp.plot_waffle_compare({"a": 0.4, "b": 0.2}, {"a": 10, "b": 20}, "t", "s", "n", path=tmp_path / "8.png")
    mp.plot_state_tiles(rates["rate"], "t", "s", "n", highlight={"CA"}, highlight_label="law", path=tmp_path / "9.png")
    mp.plot_over_time(ma.postings_over_time(jobs, "M"), "t", "s", "n", path=tmp_path / "10.png")
    assert len(list(tmp_path.glob("*.png"))) == 10


def test_empty_table_draws_nothing():
    assert mp.plot_lollipop(pd.Series(dtype=float), "t", "s", "n") is None
