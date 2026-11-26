"""Job market read in resume order: education, experience, skills (RQ1).

A job seeker reads a posting the way they write a resume: what degree and
major, how many years and what kind of experience, which skills. Each
function here answers one of those questions for the whole market.

Inputs are the clean table with the columns added by
text_signals.add_text_signals (min_years, degree) and
text_signals.add_resume_signals (majors, internship_ok,
research_asked, years_group), plus the skills table from
market_analysis.extract_skills_for_jobs.

A requirement that is not written in the posting is "Not stated". It is
never read as "not required".
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.market_analysis import _explode_if_list, _valid_salary
from src.text_signals import DEGREE_ORDER, MAJOR_ORDER, YEARS_GROUPS

STATED_DEGREES = [d for d in DEGREE_ORDER if d != "None stated"]
STATED_YEARS = [y for y in YEARS_GROUPS if y != "Not stated"]


def _groups(df: pd.DataFrame, by: str, min_n: int, top: int | None) -> pd.Series:
    """Postings per group, largest first, groups under min_n left out."""
    sizes = df.groupby(by)["job_id"].nunique()
    sizes = sizes[sizes >= min_n].sort_values(ascending=False)
    return sizes.head(top) if top else sizes


# ---------------------------------------------------------------------------
# 1. Education
# ---------------------------------------------------------------------------

def degree_requirements(jobs: pd.DataFrame, by: str | None = None,
                        min_n: int = 1000, top: int | None = 12) -> pd.DataFrame:
    """Degree level asked for, per group.

    Two different shares, because many postings name no degree:
    states_degree = share of all postings that name any degree, and the
    level columns (High school ... PhD) = shares among postings that name one
    (they add up to 1). With by=None the result is one row for all postings.
    """
    df = jobs[["job_id", "degree"] + ([by] if by else [])].copy()
    if by is None:
        df["group"] = "All postings"
        by = "group"
    df = _explode_if_list(df, by)
    sizes = _groups(df, by, min_n, top)
    df = df[df[by].isin(sizes.index)]

    stated = df[df["degree"] != "None stated"]
    levels = pd.crosstab(stated[by], stated["degree"], normalize="index")
    levels = levels.reindex(index=sizes.index, columns=STATED_DEGREES, fill_value=0)
    table = levels.copy()
    table.insert(0, "states_degree", stated.groupby(by).size().reindex(sizes.index).fillna(0) / sizes)
    table.insert(0, "postings", sizes)
    return table


def major_share(jobs: pd.DataFrame, by: str, min_n: int = 1000, top_groups: int = 12,
                top_majors: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """Which majors postings name, per group (industry or job function).

    Share = postings in the group that name the major, divided by postings in
    the group that name any major. Returns (share, postings_naming_a_major).
    """
    df = _explode_if_list(jobs[["job_id", by, "majors"]], by)
    named = df[df["majors"].apply(len) > 0]
    sizes = _groups(named, by, min_n, top_groups)
    named = named[named[by].isin(sizes.index)]

    pairs = named.explode("majors")
    keep = pairs["majors"].value_counts().head(top_majors).index
    keep = [m for m in MAJOR_ORDER if m in set(keep)]
    counts = pairs.groupby([by, "majors"])["job_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(index=sizes.index, columns=keep, fill_value=0)
    return counts.div(sizes, axis=0), sizes


def majors_overall(jobs: pd.DataFrame) -> pd.DataFrame:
    """How often each major group is named, among postings that name any major."""
    named = jobs[jobs["majors"].apply(len) > 0]
    counts = named["majors"].explode().value_counts()
    return pd.DataFrame({"postings": counts, "share": counts / len(named)})


# ---------------------------------------------------------------------------
# 2. Experience
# ---------------------------------------------------------------------------

def years_by_group(jobs: pd.DataFrame, by: str, min_n: int = 1000,
                   top: int | None = 12, order: Iterable[str] | None = None) -> pd.DataFrame:
    """Minimum years asked, grouped like a resume (1 year, 2 to 3, ...), per group.

    states_years = share of all postings that name a number of years.
    The year columns are shares among those postings (they add up to 1).
    """
    df = _explode_if_list(jobs[["job_id", by, "years_group"]], by)
    sizes = _groups(df, by, min_n, top)
    if order is not None:
        sizes = sizes.reindex([g for g in order if g in sizes.index])
    df = df[df[by].isin(sizes.index)]
    stated = df[df["years_group"] != "Not stated"]
    mix = pd.crosstab(stated[by], stated["years_group"], normalize="index")
    mix = mix.reindex(index=sizes.index, columns=STATED_YEARS, fill_value=0)
    mix.insert(0, "states_years", stated.groupby(by).size().reindex(sizes.index).fillna(0) / sizes)
    mix.insert(0, "postings", sizes)
    return mix


def education_experience_grid(jobs: pd.DataFrame, function: str | None = None) -> tuple[pd.DataFrame, int]:
    """Degree level (rows) by years asked (columns), as a share of postings.

    Includes "None stated" and "Not stated", so the grid shows how often a
    posting names both, one, or neither. With function it covers one
    job function only. Returns (share, postings).
    """
    df = jobs
    if function is not None:
        df = jobs[jobs["job_function"].apply(lambda v: function in list(v))]
    grid = pd.crosstab(df["degree"], df["years_group"], normalize="all")
    grid = grid.reindex(index=STATED_DEGREES + ["None stated"], columns=YEARS_GROUPS, fill_value=0)
    return grid, len(df)


def skill_share_by_group(jobs: pd.DataFrame, job_skills: pd.DataFrame, by: str,
                         min_n: int = 300, top_skills: int = 15,
                         order: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Share of postings in each group that mention each skill.

    Like market_analysis.skill_share_by_function but for any column,
    for example experience_level or years_group.
    """
    merged = jobs[["job_id", by]].merge(job_skills, on="job_id", how="inner")
    merged = _explode_if_list(merged, by)
    sizes = merged.groupby(by).size()
    sizes = sizes[sizes >= min_n]
    if order is not None:
        sizes = sizes.reindex([g for g in order if g in sizes.index])
    else:
        sizes = sizes.sort_values(ascending=False)
    merged = merged[merged[by].isin(sizes.index)]
    chosen = job_skills["skills"].explode().dropna().value_counts().head(top_skills).index
    pairs = merged.explode("skills").dropna(subset=["skills"])
    pairs = pairs[pairs["skills"].isin(chosen)]
    counts = pairs.groupby([by, "skills"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=sizes.index, columns=chosen, fill_value=0)
    return counts.div(sizes, axis=0), sizes


# ---------------------------------------------------------------------------
# 3. Job titles
# ---------------------------------------------------------------------------

_SENIORITY = re.compile(
    r"\b(senior|sr|junior|jr|lead|principal|staff|entry[ -]level|remote|hybrid|onsite|on-site"
    r"|part[ -]time|full[ -]time|temporary|temp|contract|i{1,3}|iv|level \d+)\b"
)


def normalize_title(title: str) -> str:
    """A short, comparable job title: lower case, no seniority words, no codes.

    "Senior Data Analyst - Remote (Req #123)" becomes "data analyst".
    Seniority is already in experience_level, so it is removed here.
    """
    if not isinstance(title, str):
        return ""
    t = title.lower()
    t = re.split(r"\s[-–|]\s|\(|,|\s/\s", t)[0]
    t = _SENIORITY.sub(" ", t)
    t = re.sub(r"[^a-z& ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def top_titles_by_years(jobs: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """The most common job titles in each years-of-experience group.

    Answers "with this many years, which jobs ask for about that much?".
    Median salary uses full-time postings with a valid yearly USD salary.
    """
    df = jobs.assign(title_norm=jobs["title"].apply(normalize_title))
    df = df[df["title_norm"] != ""]
    pay = _valid_salary(df).set_index("job_id")["salary_annual"]
    df = df.assign(pay=df["job_id"].map(pay))
    rows = []
    for group in YEARS_GROUPS:
        part = df[df["years_group"] == group]
        counts = part.groupby("title_norm").agg(postings=("job_id", "size"), median_salary=("pay", "median"))
        counts = counts.sort_values("postings", ascending=False).head(top).reset_index()
        counts.insert(0, "years_group", group)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


def titles_for_profile(jobs: pd.DataFrame, job_skills: pd.DataFrame, skills: Iterable[str],
                       years: float, degree: str, min_coverage: float = 0.5,
                       min_postings: int = 5, top: int = 15) -> tuple[pd.DataFrame, dict]:
    """Job titles a person could apply for, given their skills, years and degree.

    A posting is open to the person when (1) it asks for no more years than
    they have, or names no number, (2) it asks for no higher degree than
    theirs, or names none, and (3) the person has at least min_coverage
    of the skills found in the posting. Postings with no extracted skill are
    left out, because coverage cannot be measured.

    Returns (titles, info). titles has, per title: postings open,
    the share of those that state requirements, median skill coverage,
    median salary, and the skills the person most often lacks.
    """
    have = {s.lower() for s in skills}
    level = DEGREE_ORDER.index(degree)

    df = jobs.merge(job_skills, on="job_id", how="inner")
    df = df[df["skills"].apply(len) > 0]
    years_ok = df["min_years"].isna() | (df["min_years"] <= years)
    degree_ok = df["degree"].map(DEGREE_ORDER.index) <= level
    degree_ok |= df["degree"] == "None stated"
    df = df[years_ok & degree_ok].copy()

    df["coverage"] = df["skills"].apply(lambda s: np.mean([x.lower() in have for x in s]))
    df = df[df["coverage"] >= min_coverage].copy()
    df["missing"] = df["skills"].apply(lambda s: [x for x in s if x.lower() not in have])
    df["title_norm"] = df["title"].apply(normalize_title)
    df["states_req"] = df["min_years"].notna() | (df["degree"] != "None stated")
    pay = _valid_salary(df).set_index("job_id")["salary_annual"]
    df["pay"] = df["job_id"].map(pay)

    def lacking(series: pd.Series) -> str:
        counts = series.explode().dropna().value_counts().head(3)
        return ", ".join(counts.index)

    titles = df[df["title_norm"] != ""].groupby("title_norm").agg(
        postings=("job_id", "size"),
        states_requirements=("states_req", "mean"),
        median_coverage=("coverage", "median"),
        median_salary=("pay", "median"),
        often_missing=("missing", lacking),
    )
    titles = titles[titles["postings"] >= min_postings].sort_values("postings", ascending=False).head(top)
    info = {"postings_open": len(df), "titles": int(df["title_norm"].nunique())}
    return titles, info


# ---------------------------------------------------------------------------
# 4. Tables for the website story
# ---------------------------------------------------------------------------

def major_function_flows(jobs: pd.DataFrame, top_majors: int = 10, top_functions: int = 10,
                         min_postings: int = 30) -> pd.DataFrame:
    """How many postings name each major and carry each job function.

    One row per (major, job_function) pair, for the flow chart from majors
    to careers. Only postings that name a major are used.
    """
    df = jobs.loc[jobs["majors"].apply(len) > 0, ["job_id", "majors", "job_function"]]
    df = _explode_if_list(_explode_if_list(df, "majors"), "job_function")
    majors = df["majors"].value_counts().head(top_majors).index
    functions = df["job_function"].value_counts().head(top_functions).index
    df = df[df["majors"].isin(majors) & df["job_function"].isin(functions)]
    flows = df.groupby(["majors", "job_function"])["job_id"].nunique().rename("postings").reset_index()
    flows = flows[flows["postings"] >= min_postings]
    return flows.rename(columns={"majors": "major"}).sort_values("postings", ascending=False).reset_index(drop=True)


def years_ladder(jobs: pd.DataFrame, levels: Iterable[str], max_years: int = 12) -> pd.DataFrame:
    """For each LinkedIn level, the share of postings asking for each number of years.

    Only postings that name a number. Values above max_years are put in
    the max_years bucket. Columns: level, years, postings, share.
    """
    df = jobs.dropna(subset=["min_years", "experience_level"])
    df = df[df["experience_level"].isin(list(levels))]
    years = df["min_years"].clip(upper=max_years).astype(int)
    counts = df.groupby([df["experience_level"], years]).size().rename("postings").reset_index()
    counts.columns = ["level", "years", "postings"]
    counts["share"] = counts["postings"] / counts.groupby("level")["postings"].transform("sum")
    return counts


def skill_profiles(jobs: pd.DataFrame, job_skills: pd.DataFrame, by: str = "job_function",
                   min_n: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """Share of each group's postings that mention every skill (all skills, not a top list)."""
    merged = _explode_if_list(jobs[["job_id", by]].merge(job_skills, on="job_id", how="inner"), by)
    sizes = merged.groupby(by).size()
    sizes = sizes[sizes >= min_n].sort_values(ascending=False)
    merged = merged[merged[by].isin(sizes.index)]
    pairs = merged.explode("skills").dropna(subset=["skills"])
    counts = pairs.groupby([by, "skills"]).size().unstack(fill_value=0).reindex(sizes.index, fill_value=0)
    return counts.div(sizes, axis=0), sizes


def field_similarity(profiles: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity of skill profiles between groups (1 = same skill mix).

    Each group is a vector of skill shares. Two fields that ask for the same
    skills in the same proportions get a value near 1.
    """
    m = profiles.to_numpy(dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    unit = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    sim = unit @ unit.T
    np.fill_diagonal(sim, np.nan)
    return pd.DataFrame(sim, index=profiles.index, columns=profiles.index)


def transitions(profiles: pd.DataFrame, similarity: pd.DataFrame, closest: int = 3,
                skills: int = 3, min_share: float = 0.03) -> pd.DataFrame:
    """For each field, the closest other fields by skills, what they share, and what to add.

    shared_skills = skills common in both fields (highest of the smaller share).
    skills_to_add = skills that at least min_share of the target's postings
    mention, and that the starting field mentions at most half as often,
    ordered by the gap in share.
    """
    rows = []
    for source in similarity.index:
        for target in similarity.loc[source].dropna().sort_values(ascending=False).head(closest).index:
            a, b = profiles.loc[source], profiles.loc[target]
            both = np.minimum(a, b)
            shared = both[both >= min_share].sort_values(ascending=False).head(skills).index
            # a skill to add is common in the target but at most half as common in the source
            new = (b >= min_share) & (a < 0.5 * b)
            add = (b - a)[new].sort_values(ascending=False).head(skills).index
            rows.append({"from_field": source, "to_field": target,
                         "similarity": similarity.loc[source, target],
                         "shared_skills": ", ".join(shared), "skills_to_add": ", ".join(add)})
    return pd.DataFrame(rows)


def field_profiles(jobs: pd.DataFrame, profiles: pd.DataFrame, similarity: pd.DataFrame,
                   salary: pd.DataFrame, top_skills: int = 5) -> pd.DataFrame:
    """One row per job function with what a resume needs to show for it.

    profiles and similarity come from skill_profiles and
    field_similarity; salary from market_analysis.salary_ranges
    by job function.
    """
    fields = list(profiles.index)
    degree = degree_requirements(jobs, by="job_function", min_n=1, top=None).reindex(fields)
    years = years_by_group(jobs, by="job_function", min_n=1, top=None).reindex(fields)
    flat = _explode_if_list(jobs[["job_id", "job_function", "min_years", "remote_allowed", "internship_ok"]],
                            "job_function")
    flat = flat[flat["job_function"].isin(fields)].groupby("job_function")
    overall = profiles.mean(axis=0)
    rows = []
    for f in fields:
        lift = profiles.loc[f] / overall.replace(0, np.nan)
        key = profiles.loc[f][(profiles.loc[f] >= 0.05) & (lift >= 1.2)].sort_values(ascending=False)
        common = profiles.loc[f].sort_values(ascending=False)
        level_cols = degree.loc[f, STATED_DEGREES]
        rows.append({
            "job_function": f,
            "postings": int(degree.loc[f, "postings"]),
            "names_degree": degree.loc[f, "states_degree"],
            "most_named_degree": level_cols.idxmax() if level_cols.sum() > 0 else "None stated",
            "bachelor_or_higher": level_cols[["Bachelor", "Master", "PhD"]].sum(),
            "master_or_higher": level_cols[["Master", "PhD"]].sum(),
            "names_years": years.loc[f, "states_years"],
            "median_years": flat["min_years"].median().get(f, np.nan),
            "share_3plus_years": flat["min_years"].apply(lambda s: (s.dropna() >= 3).mean()).get(f, np.nan),
            "internship_ok": flat["internship_ok"].mean().get(f, np.nan),
            "remote_share": flat["remote_allowed"].mean().get(f, np.nan),
            "top_skills": ", ".join(common.head(top_skills).index),
            "key_skills": ", ".join(key.head(top_skills).index),
            "median_salary": salary["median"].get(f, np.nan) if "median" in salary else np.nan,
            "salary_p25": salary["p25"].get(f, np.nan) if "p25" in salary else np.nan,
            "salary_p75": salary["p75"].get(f, np.nan) if "p75" in salary else np.nan,
            "closest_fields": ", ".join(similarity.loc[f].dropna().sort_values(ascending=False).head(3).index),
        })
    return pd.DataFrame(rows).set_index("job_function")
