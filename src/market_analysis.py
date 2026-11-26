"""My market analysis functions for RQ1.

Each function takes the clean table from data_processing.load_clean_jobs
(a few also need extra tables) and returns a small summary table. The charts
are in market_plots.py, and notebooks/01_data_eda.ipynb calls both in order.

How I count list columns (job_function, industry, skills): a posting with
several labels counts once under each label. Shares are within each label,
so across labels they can add up to more than 100%.
"""

from __future__ import annotations

import time
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

FULL_TIME = "Full-time"
SALARY_BOUNDS = (10_000, 1_000_000)
STAFFING_INDUSTRY = "Staffing and Recruiting"

# start date of each state's law that makes job postings show a salary range
# Source: https://www.clockspot.com/articles/pay-transparency-laws-by-state (checked Nov 17, 2026).
# I only count the law for postings listed on or after that date
PAY_TRANSPARENCY_START = {
    "CO": date(2021, 1, 1),
    "CA": date(2023, 1, 1),
    "WA": date(2023, 1, 1),
    "NY": date(2023, 9, 17),
    "HI": date(2024, 1, 1),
    "MD": date(2024, 10, 1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_list(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _explode_if_list(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per label when column holds lists. Empty lists are dropped."""
    first = df[column].dropna().head(1)
    # parquet gives lists back as numpy arrays, so I check for both
    if len(first) and _is_list(first.iloc[0]):
        df = df.explode(column)
    # explode repeats the old index, so I reset it
    return df.dropna(subset=[column]).reset_index(drop=True)


def _valid_salary(jobs: pd.DataFrame, full_time_only: bool = True,
                  bounds: tuple[float, float] = SALARY_BOUNDS) -> pd.DataFrame:
    """Postings with a usable yearly USD salary (optionally full-time only)."""
    df = jobs[jobs["salary_annual"].notna()]
    if full_time_only:
        df = df[df["work_type"] == FULL_TIME]
    return df[df["salary_annual"].between(*bounds)]


# ---------------------------------------------------------------------------
# Extra columns from the raw tables
# ---------------------------------------------------------------------------

def add_posting_details(jobs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add views, applies and company size from the raw Kaggle tables.

    tables comes from data_processing.load_raw_tables. The clean table
    itself is not changed; this returns a copy with three more columns.
    company_size is LinkedIn's 0 to 7 size scale (0 = smallest).
    """
    postings = tables["postings"][["job_id", "views", "applies"]]
    out = jobs.merge(postings, on="job_id", how="left")
    if "companies" in tables and "company_size" in tables["companies"].columns:
        size = tables["companies"].drop_duplicates("company_id").set_index("company_id")["company_size"]
        out["company_size"] = out["company_id"].map(size)
    return out


# ---------------------------------------------------------------------------
# A. Market structure
# ---------------------------------------------------------------------------

def count_by_group(jobs: pd.DataFrame, by: str, top: int | None = None) -> pd.DataFrame:
    """Postings per group and their share of all postings."""
    df = _explode_if_list(jobs[["job_id", by]], by)
    counts = df.groupby(by).size().sort_values(ascending=False).rename("postings").to_frame()
    counts["share_of_all"] = counts["postings"] / len(jobs)
    return counts.head(top) if top else counts


def function_cooccurrence(jobs: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """For each pair of job functions: share of postings tagged A that are also tagged B.

    Rows are A, columns are B, the diagonal is left empty. Only the top
    largest functions are kept.
    """
    labels = jobs["job_function"][jobs["job_function"].apply(len) > 0]
    keep = labels.explode().value_counts().head(top).index
    matrix = pd.DataFrame(0, index=keep, columns=keep, dtype=float)
    for tags in labels:
        tags = [t for t in tags if t in keep]
        for a, b in combinations(tags, 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1
    totals = labels.explode().value_counts().reindex(keep)
    values = matrix.div(totals, axis=0).to_numpy(copy=True)
    np.fill_diagonal(values, np.nan)
    return pd.DataFrame(values, index=keep, columns=keep)


def crosstab_share(jobs: pd.DataFrame, row: str, col: str,
                   top_rows: int = 12, top_cols: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Share of each row group's postings that fall in each column group.

    Example: row="industry", col="job_function" answers "of the postings in
    Hospitals and Health Care, what share are Sales jobs?". Returns
    (share, postings_per_row).
    """
    df = _explode_if_list(_explode_if_list(jobs[["job_id", row, col]], row), col)
    rows = df.groupby(row)["job_id"].nunique().sort_values(ascending=False).head(top_rows)
    cols = df[col].value_counts().head(top_cols).index
    df = df[df[row].isin(rows.index) & df[col].isin(cols)]
    counts = df.groupby([row, col])["job_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(index=rows.index, columns=cols, fill_value=0)
    return counts.div(rows, axis=0), rows


def company_concentration(jobs: pd.DataFrame, top: int = 20) -> tuple[pd.DataFrame, dict]:
    """Largest employers by postings, and how much of the market they hold.

    info reports the number of companies, the share of postings from the
    top 1% of companies, and the share from staffing and recruiting firms.
    """
    counts = jobs["company_name"].value_counts()
    table = counts.head(top).rename("postings").to_frame()
    table["share_of_all"] = table["postings"] / len(jobs)
    table["cumulative_share"] = table["share_of_all"].cumsum()

    top_1pct = max(1, int(len(counts) * 0.01))
    staffing = jobs["industry"].apply(lambda v: _is_list(v) and STAFFING_INDUSTRY in list(v))
    info = {
        "companies": len(counts),
        "top_1pct_companies": top_1pct,
        "top_1pct_share": counts.head(top_1pct).sum() / len(jobs),
        "staffing_share": staffing.mean(),
    }
    return table, info


# ---------------------------------------------------------------------------
# B. Skills (uses Jessy's extract_skills)
# ---------------------------------------------------------------------------

def extract_skills_for_jobs(
    jobs: pd.DataFrame,
    extractor: Callable[[str], Iterable[str]],
    cache_path: str | Path | None = None,
    report_every: int = 10_000,
) -> pd.DataFrame:
    """Run a skill extractor on every description, once.

    Returns job_id and skills (a sorted list per job). With
    cache_path the result is saved and reused on the next run, as long as
    it covers every job_id in jobs.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        if set(jobs["job_id"]).issubset(set(cached["job_id"])):
            cached["skills"] = cached["skills"].apply(list)
            return cached[cached["job_id"].isin(jobs["job_id"])].reset_index(drop=True)

    skills = []
    start = time.perf_counter()
    for i, text in enumerate(jobs["description"], start=1):
        skills.append(sorted(set(extractor(text))))
        if report_every and i % report_every == 0:
            print(f"{i:,} of {len(jobs):,} done, {time.perf_counter() - start:.0f} s")

    result = pd.DataFrame({"job_id": jobs["job_id"].to_numpy(), "skills": skills})
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def skill_share_by_function(
    jobs: pd.DataFrame,
    job_skills: pd.DataFrame,
    min_postings: int = 1000,
    top_skills: int = 20,
    top_functions: int | None = 12,
) -> tuple[pd.DataFrame, pd.Series]:
    """Share of postings in each job function that mention each skill.

    Returns (share, postings): one row per function, one column per skill
    (0 to 1), and the postings behind each row. Postings where no skill was
    found still count in the denominator.
    """
    merged = jobs[["job_id", "job_function"]].merge(job_skills, on="job_id", how="inner")
    by_function = _explode_if_list(merged, "job_function")

    postings = by_function.groupby("job_function").size()
    postings = postings[postings >= min_postings].sort_values(ascending=False)
    if top_functions is not None:
        postings = postings.head(top_functions)
    by_function = by_function[by_function["job_function"].isin(postings.index)]

    chosen = job_skills["skills"].explode().dropna().value_counts().head(top_skills).index
    pairs = by_function.explode("skills").dropna(subset=["skills"])
    pairs = pairs[pairs["skills"].isin(chosen)]
    counts = pairs.groupby(["job_function", "skills"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=postings.index, columns=chosen, fill_value=0)
    return counts.div(postings, axis=0), postings


def signature_skills(
    jobs: pd.DataFrame,
    job_skills: pd.DataFrame,
    min_postings: int = 1000,
    min_share: float = 0.05,
    per_function: int = 5,
) -> pd.DataFrame:
    """Skills that are much more common in one job function than overall.

    lift = share of the function's postings that mention the skill, divided
    by the share of all postings that mention it. A lift of 3 means three
    times as common as in the market overall. Skills mentioned by fewer than
    min_share of the function's postings are skipped, so rare words do not
    win by chance.
    """
    merged = jobs[["job_id", "job_function"]].merge(job_skills, on="job_id", how="inner")
    total = len(merged)
    overall = merged["skills"].explode().dropna().value_counts() / total

    by_function = _explode_if_list(merged, "job_function")
    postings = by_function.groupby("job_function").size()
    postings = postings[postings >= min_postings]
    by_function = by_function[by_function["job_function"].isin(postings.index)]

    pairs = by_function.explode("skills").dropna(subset=["skills"])
    counts = pairs.groupby(["job_function", "skills"]).size().rename("mentions").reset_index()
    counts["share_in_function"] = counts["mentions"] / counts["job_function"].map(postings)
    counts["share_overall"] = counts["skills"].map(overall)
    counts["lift"] = counts["share_in_function"] / counts["share_overall"]

    counts = counts[counts["share_in_function"] >= min_share]
    counts = counts.sort_values(["job_function", "lift"], ascending=[True, False])
    return counts.groupby("job_function").head(per_function).reset_index(drop=True)


def skill_pairs(job_skills: pd.DataFrame, min_count: int = 200, top: int = 20) -> pd.DataFrame:
    """Skill pairs that appear in the same posting more often than chance.

    lift = how often the pair appears together, divided by how often it
    would appear if the two skills were unrelated. jaccard = postings with
    both / postings with either.
    """
    n = len(job_skills)
    single = job_skills["skills"].explode().dropna().value_counts()
    pair_counts: dict[tuple[str, str], int] = {}
    for skills in job_skills["skills"]:
        for a, b in combinations(sorted(skills), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = [
        {"skill_a": a, "skill_b": b, "together": c,
         "lift": c * n / (single[a] * single[b]),
         "jaccard": c / (single[a] + single[b] - c)}
        for (a, b), c in pair_counts.items() if c >= min_count
    ]
    if not rows:
        return pd.DataFrame(columns=["skill_a", "skill_b", "together", "lift", "jaccard"])
    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(top).reset_index(drop=True)


def skill_salary_premium(
    jobs: pd.DataFrame,
    job_skills: pd.DataFrame,
    skills: Iterable[str] | None = None,
    min_n: int = 50,
    top_skills: int = 25,
) -> pd.DataFrame:
    """How much more postings that mention a skill pay, within the same job function.

    For each skill and each job function, compare the median salary of
    postings that mention the skill with those that do not. The premium is
    the average of these within-function gaps, weighted by how many postings
    mention the skill. Comparing inside each function keeps the result from
    just reflecting which functions use the skill (for example, Python
    appears mostly in IT, which pays more anyway).

    Only full-time postings with a valid yearly USD salary are used. Within
    a function, both groups need at least min_n postings.
    """
    paid = _valid_salary(jobs)[["job_id", "job_function", "salary_annual"]]
    paid = paid.merge(job_skills, on="job_id", how="inner")
    paid = _explode_if_list(paid, "job_function")

    if skills is None:
        skills = job_skills["skills"].explode().dropna().value_counts().head(top_skills).index

    rows = []
    for skill in skills:
        paid["has_skill"] = paid["skills"].apply(lambda v, s=skill: s in v)
        gaps, weights, n_with = [], [], 0
        for _, group in paid.groupby("job_function"):
            with_skill = group.loc[group["has_skill"], "salary_annual"]
            without = group.loc[~group["has_skill"], "salary_annual"]
            if len(with_skill) >= min_n and len(without) >= min_n:
                gaps.append(with_skill.median() / without.median() - 1)
                weights.append(len(with_skill))
                n_with += len(with_skill)
        if gaps:
            rows.append({
                "skill": skill,
                "premium": float(np.average(gaps, weights=weights)),
                "functions_compared": len(gaps),
                "postings_with_skill": n_with,
            })
    table = pd.DataFrame(rows, columns=["skill", "premium", "functions_compared", "postings_with_skill"])
    return table.sort_values("premium").set_index("skill")


# ---------------------------------------------------------------------------
# C. Experience and degree (entry level)
# ---------------------------------------------------------------------------

def experience_paradox(jobs: pd.DataFrame, min_years: pd.Series,
                       levels: Iterable[str] | None = None) -> pd.DataFrame:
    """Years of experience asked for, by LinkedIn experience level.

    min_years is one number per posting (NaN when none is stated), from
    text_signals.min_years_required. For each level: how many postings
    state a number, and among those, the share asking for 2+, 3+ and 5+ years.
    """
    df = pd.DataFrame({"level": jobs["experience_level"], "years": min_years.to_numpy()})
    df = df.dropna(subset=["level"])
    stated = df.dropna(subset=["years"])
    table = pd.DataFrame({
        "postings": df.groupby("level").size(),
        "states_years": stated.groupby("level").size(),
        "median_years": stated.groupby("level")["years"].median(),
        "share_2plus": stated.groupby("level")["years"].apply(lambda s: (s >= 2).mean()),
        "share_3plus": stated.groupby("level")["years"].apply(lambda s: (s >= 3).mean()),
        "share_5plus": stated.groupby("level")["years"].apply(lambda s: (s >= 5).mean()),
    }).fillna({"states_years": 0})
    table["states_years_share"] = table["states_years"] / table["postings"]
    if levels is not None:
        table = table.reindex([lvl for lvl in levels if lvl in table.index])
    return table


def mix_by_group(jobs: pd.DataFrame, by: str, values: pd.Series,
                 order: Iterable[str], min_n: int = 1000, top: int | None = 12) -> pd.DataFrame:
    """Share of each category of values within each group (rows sum to 1).

    Used for degree level by job function. order fixes the column order.
    """
    df = pd.DataFrame({by: jobs[by].to_numpy(), "value": values.to_numpy()})
    df = _explode_if_list(df, by)
    sizes = df.groupby(by).size()
    sizes = sizes[sizes >= min_n].sort_values(ascending=False)
    if top:
        sizes = sizes.head(top)
    df = df[df[by].isin(sizes.index)]
    table = pd.crosstab(df[by], df["value"], normalize="index")
    return table.reindex(index=sizes.index, columns=list(order), fill_value=0)


# ---------------------------------------------------------------------------
# D. Salary
# ---------------------------------------------------------------------------

def salary_ranges(
    jobs: pd.DataFrame,
    by: str = "job_function",
    full_time_only: bool = True,
    bounds: tuple[float, float] = SALARY_BOUNDS,
    min_n: int = 200,
    order: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Median and middle 50% of yearly USD salary per group.

    Full-time only by default, so annualized hourly pay of part-time jobs
    does not pull numbers down. Values outside bounds count as entry
    errors; info reports how many were dropped at each step.
    """
    df = jobs[jobs["salary_annual"].notna()]
    info = {"with_salary": len(df)}
    if full_time_only:
        df = df[df["work_type"] == FULL_TIME]
    info["after_work_type_filter"] = len(df)
    inside = df["salary_annual"].between(*bounds)
    info["outside_bounds"] = int((~inside).sum())
    df = _explode_if_list(df[inside], by)

    grouped = df.groupby(by)["salary_annual"]
    summary = pd.DataFrame({
        "n": grouped.size(),
        "p25": grouped.quantile(0.25),
        "median": grouped.median(),
        "p75": grouped.quantile(0.75),
    })
    summary = summary[summary["n"] >= min_n]
    if order is not None:
        summary = summary.reindex([g for g in order if g in summary.index])
    else:
        summary = summary.sort_values("median")
    return summary, info


def range_width(jobs: pd.DataFrame, by: str = "job_function", min_n: int = 200) -> pd.DataFrame:
    """How wide posted salary ranges are: max / min, for postings that give both.

    A ratio of 1.5 means the top of the range is 50% above the bottom. Very
    wide ranges tell a job seeker little about the actual pay.
    """
    df = jobs[jobs["salary_min"].notna() & jobs["salary_max"].notna() & (jobs["salary_min"] > 0)]
    df = df.assign(ratio=df["salary_max"] / df["salary_min"])
    df = _explode_if_list(df[["ratio", by]], by)
    grouped = df.groupby(by)["ratio"]
    table = pd.DataFrame({"n": grouped.size(), "median_ratio": grouped.median(),
                          "share_over_1_5": grouped.apply(lambda s: (s > 1.5).mean())})
    return table[table["n"] >= min_n].sort_values("median_ratio")


def remote_pay_gap(jobs: pd.DataFrame, by: str = "job_function", min_n: int = 100) -> pd.DataFrame:
    """Median full-time salary of remote vs on-site postings, per group."""
    df = _explode_if_list(_valid_salary(jobs)[["salary_annual", "remote_allowed", by]], by)
    grouped = df.groupby([by, "remote_allowed"])["salary_annual"]
    wide = pd.DataFrame({"median": grouped.median(), "n": grouped.size()}).unstack("remote_allowed")
    table = pd.DataFrame({
        "onsite_median": wide[("median", False)],
        "remote_median": wide[("median", True)],
        "onsite_n": wide[("n", False)],
        "remote_n": wide[("n", True)],
    }).dropna()
    table = table[(table["onsite_n"] >= min_n) & (table["remote_n"] >= min_n)]
    table["gap"] = table["remote_median"] / table["onsite_median"] - 1
    return table.sort_values("gap")


def add_pay_transparency_flag(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add transparency_law: True when the posting's state required salary
    ranges on the day the posting was listed. Postings without a state or a
    date get NaN."""
    start = jobs["state"].map(PAY_TRANSPARENCY_START)
    listed = pd.to_datetime(jobs["listed_date"], errors="coerce").dt.date
    flag = pd.Series(np.nan, index=jobs.index, dtype=object)
    known = jobs["state"].notna() & listed.notna()
    flag[known] = [bool(s is not None and s == s and d >= s)
                   for s, d in zip(start[known], listed[known])]
    return jobs.assign(transparency_law=flag)


# ---------------------------------------------------------------------------
# E. Rates, competition, data quality, time
# ---------------------------------------------------------------------------

def rate_by_group(jobs: pd.DataFrame, by: str, flag: str = "salary_disclosed",
                  min_n: int = 500) -> pd.DataFrame:
    """Share of postings in each group where flag is true (groups with
    fewer than min_n postings are left out)."""
    df = _explode_if_list(jobs[[by, flag]].dropna(subset=[flag]), by)
    df = df.assign(**{flag: df[flag].astype(bool)})
    table = df.groupby(by)[flag].agg(postings="size", rate="mean")
    return table[table["postings"] >= min_n].sort_values("rate")


def competition_by_group(jobs: pd.DataFrame, by: str = "job_function", min_n: int = 300) -> pd.DataFrame:
    """Applications and views per posting, for postings where LinkedIn counted them.

    applies_per_100_views = total applies / total views x 100, using only
    postings that have both numbers.
    """
    df = jobs[jobs["views"].notna() & jobs["applies"].notna() & (jobs["views"] > 0)]
    df = _explode_if_list(df[["views", "applies", by]], by)
    grouped = df.groupby(by)
    table = pd.DataFrame({
        "n": grouped.size(),
        "median_applies": grouped["applies"].median(),
        "median_views": grouped["views"].median(),
        "applies_per_100_views": grouped["applies"].sum() / grouped["views"].sum() * 100,
    })
    return table[table["n"] >= min_n].sort_values("applies_per_100_views")


def missing_by_group(jobs: pd.DataFrame, by: str, column: str) -> pd.DataFrame:
    """How many postings per group have no value in column."""
    df = _explode_if_list(jobs[[by, column]], by)
    empty = df[column].apply(
        lambda v: v is None or (_is_list(v) and len(v) == 0) or (isinstance(v, float) and np.isnan(v))
    )
    table = pd.DataFrame({"postings": df.groupby(by).size(), "missing": empty.groupby(df[by]).sum()})
    table["missing_share"] = table["missing"] / table["postings"]
    return table.sort_values("postings", ascending=False)


def missing_matrix(jobs: pd.DataFrame, columns: Iterable[str], by: str = "job_function",
                   top: int = 12) -> pd.DataFrame:
    """Share missing for several columns (columns) in each group (rows)."""
    parts = {c: missing_by_group(jobs, by, c)["missing_share"] for c in columns}
    table = pd.DataFrame(parts)
    sizes = count_by_group(jobs, by).head(top).index
    return table.reindex(sizes)


def postings_over_time(jobs: pd.DataFrame, freq: str = "W") -> pd.Series:
    """Number of postings per period (week by default) by listed date."""
    dates = pd.to_datetime(jobs["listed_date"], errors="coerce").dropna()
    return dates.dt.to_period(freq).value_counts().sort_index()
