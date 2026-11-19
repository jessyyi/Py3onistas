"""Top-K recommendations using a prepared matching model."""

from numbers import Integral
from math import isfinite

import pandas as pd

from src.matching import skill_coverage


def score_jobs(candidate, jobs, *, matcher, skills_column=None, text_weight=0.5):
    """Score jobs; absent skill evidence falls back to text for hybrid.

    Skill labels must be normalized upstream. Weights are experimental,
    and scores do not represent eligibility or employment probabilities.
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
    """Rank by text, skill or hybrid; exclude unknown skill-only scores."""
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
