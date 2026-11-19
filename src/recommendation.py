"""Top-K recommendations using a prepared matching model."""

from numbers import Integral
from math import isfinite

import pandas as pd

from src.matching import skill_coverage


def score_jobs(candidate, jobs, *, matcher, skills_column=None, text_weight=0.5):
    """Calculate text, skill, and hybrid scores without changing input jobs.

    Args:
        candidate: Declared profile with an optional list of skills.
        jobs: The same table, including row order, used to fit matcher.
        matcher: A fitted TextMatcher, reusable across candidate queries.
        skills_column: Column containing lists of normalized job skill labels.
            None means that skill evidence is unavailable for every job.
        text_weight: Text contribution to hybrid scoring, between zero and one.

    Returns:
        A copy of jobs with text_score, skill_score, matched_skills,
        unverified_skills, hybrid_score, and hybrid_uses_skills columns.
        Empty job skill lists yield NaN skill scores. Hybrid uses a weighted
        mean when skills are available, and text score alone otherwise.

    Raises:
        ValueError: If the weight is invalid, matcher is not fitted, jobs differs
            from the fitted table, or the requested skills column is missing.

    Notes:
        Skill aliases must be normalized upstream for both jobs and candidates.
        Weights are experimental; scores are not employment probabilities.
    """
    if not isfinite(text_weight) or not 0 <= text_weight <= 1:
        raise ValueError("text_weight must be between 0 and 1.")
    if not hasattr(matcher, "jobs"):
        raise ValueError("Fit the matcher before requesting recommendations.")

    if not jobs.equals(matcher.jobs):
        raise ValueError("jobs must match the table used to fit the matcher.")

    results = jobs.copy()
    results["text_score"] = matcher.score(candidate)
    if skills_column is not None and skills_column not in jobs.columns:
        raise ValueError(f"Missing skills column: {skills_column}")

    skill_lists = jobs[skills_column] if skills_column is not None else [[] for _ in range(len(jobs))]
    coverage = pd.DataFrame([
        skill_coverage(candidate.get("skills", []), skills)
        for skills in skill_lists
    ], index=jobs.index)
    for column in coverage.columns:
        results[column] = coverage[column]

    results["skill_score"] = pd.to_numeric(results["skill_score"], errors="coerce")
    known = results["skill_score"].notna()
    results["hybrid_uses_skills"] = known
    results["hybrid_score"] = results["text_score"]
    results.loc[known, "hybrid_score"] = (
        text_weight * results.loc[known, "text_score"]
        + (1 - text_weight) * results.loc[known, "skill_score"]
    )
    return results


def recommend_top_k(candidate, jobs, k=10, *, matcher, method="text",
                    skills_column=None, text_weight=0.5):
    """Return the highest-scoring jobs for one candidate and ranking method.

    Args:
        candidate: Declared profile, including skills for skill-based methods.
        jobs: The same job table used to fit matcher.
        k: Positive maximum number of jobs to return.
        matcher: A fitted TextMatcher; this function does not refit it.
        method: One of 'text', 'skill', or 'hybrid'.
        skills_column: Column of normalized job skill lists, or None.
        text_weight: Text contribution to hybrid scoring, between zero and one.

    Returns:
        Up to k jobs with scores and skill explanations, sorted by the selected
        score descending, then job_id ascending. The index is reset. Unknown
        selected scores are excluded, so skill-only may return fewer than k.

    Raises:
        ValueError: If k or method is invalid, or score_jobs rejects its inputs.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")
    if method not in {"text", "skill", "hybrid"}:
        raise ValueError("method must be text, skill or hybrid.")
    results = score_jobs(candidate, jobs, matcher=matcher,
                         skills_column=skills_column, text_weight=text_weight)
    score_column = f"{method}_score"

    return results.dropna(subset=[score_column]).sort_values(
        [score_column, "job_id"],
        ascending=[False, True],
    ).head(k).reset_index(drop=True)


def rank_job_categories(
    scored_jobs: pd.DataFrame,
    job_categories: pd.DataFrame,
    *,
    score_column="text_score",
    m: int = 5,
) -> pd.DataFrame:
    """Rank job categories independently of the user's selected category.

    Args:
        scored_jobs: One row per non-missing, unique job_id, with score_column.
        job_categories: job_id/category links, with one category per row.
            Multiple categories per job are allowed; duplicate links are removed.
        score_column: Job score to aggregate, such as text_score or hybrid_score.
        m: Positive maximum number of highest-scoring jobs used per category.

    Returns:
        A table with rank, category, category_score, jobs_used, and jobs_available.
        Category scores are means of up to m non-missing job scores. Available
        counts include linked jobs with missing scores; categories with no known
        scores are omitted. Links outside the supplied job pool are ignored.
        Ties are resolved by category name, with ranks starting at one.

    Raises:
        ValueError: If m is invalid, required columns are missing, or scored job
            identifiers are missing or duplicated.

    Notes:
        Averaging avoids the direct count advantage of summing all scores, but
        does not eliminate category-size effects. Small categories can be ranked
        on fewer than m jobs. Scores are not validated career suitability measures.
    """
    if isinstance(m, bool) or not isinstance(m, Integral) or m < 1:
        raise ValueError("m must be a positive integer.")

    if not {"job_id", score_column}.issubset(scored_jobs.columns):
        raise ValueError("Missing job ID or score column.")

    if not {"job_id", "category"}.issubset(job_categories.columns):
        raise ValueError("Missing job ID or category column.")

    if (
        scored_jobs["job_id"].isna().any()
        or scored_jobs["job_id"].duplicated().any()
    ):
        raise ValueError("Scored job IDs must be non-missing and unique.")

    links = (
        job_categories[["job_id", "category"]]
        .dropna()
        .drop_duplicates()
    )

    # Restrict category links to the supplied candidate's job pool.
    linked = links.merge(
        scored_jobs[["job_id", score_column]],
        on="job_id",
        how="inner",
        validate="many_to_one",
    )

    sizes = linked.groupby("category").size().rename("jobs_available")
    eligible = linked.dropna(subset=[score_column])

    best = (
        eligible.sort_values(
            ["category", score_column, "job_id"],
            ascending=[True, False, True],
        )
        .groupby("category", sort=False)
        .head(m)
    )

    ranking = (
        best.groupby("category")
        .agg(
            category_score=(score_column, "mean"),
            jobs_used=("job_id", "size"),
        )
        .join(sizes)
        .reset_index()
        .sort_values(
            ["category_score", "category"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking
