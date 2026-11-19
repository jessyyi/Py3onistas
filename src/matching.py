"""TF-IDF matching for historical job descriptions."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class CandidateProfile(TypedDict, total=False):
    """Optional declared profile fields; candidate_id is metadata only."""
    candidate_id: str
    skills: list[str]
    education: str
    experience: str


class SkillCoverageResult(TypedDict):
    """Skill evidence summary with an unknown score for empty job labels."""
    skill_score: float | None
    matched_skills: list[str]
    unverified_skills: list[str]


def profile_to_text(profile: CandidateProfile) -> str:
    """Combine declared candidate information into matching text.

    Args:
        profile: Optional skills, education, and experience fields.

    Returns:
        A single string excluding candidate_id. Missing fields contribute
        empty strings. Education and experience are text, not eligibility checks.
    """
    return " ".join([
        " ".join(profile.get("skills", [])),
        profile.get("education", ""),
        profile.get("experience", ""),
    ])


class TextMatcher:
    """Prepare job vectors once and reuse them for candidate scoring."""

    def fit(self, jobs: pd.DataFrame) -> "TextMatcher":
        """Fit TF-IDF on job descriptions and retain their original row order.

        Args:
            jobs: Nonempty table with job_id, title, and description columns.
                Job identifiers must be unique and non-missing. Missing
                descriptions are treated as empty strings.

        Returns:
            This matcher, with a copy of the input table and sparse job vectors.
            Reuse it for subsequent candidate queries.

        Raises:
            ValueError: If required columns or valid job identifiers are missing,
                the table is empty, or descriptions yield an empty vocabulary.
        """
        required = {"job_id", "title", "description"}
        missing = required - set(jobs.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if jobs.empty:
            raise ValueError("The job table is empty.")

        if jobs["job_id"].isna().any() or jobs["job_id"].duplicated().any():
            raise ValueError("job_id must be non-missing and unique.")

        self.jobs = jobs.copy()
        descriptions = self.jobs["description"].fillna("")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_vectors = self.vectorizer.fit_transform(descriptions)
        return self

    def score(self, candidate: CandidateProfile) -> NDArray[np.float64]:
        """Compare a candidate with the fitted job descriptions.

        Args:
            candidate: Declared profile fields combined by profile_to_text.

        Returns:
            One cosine similarity score per fitted job, in original row order.
            Words outside the fitted vocabulary are ignored. Scores measure
            text similarity, not eligibility or employment probability.

        Raises:
            RuntimeError: If fit has not been called successfully.
        """
        if not hasattr(self, "job_vectors"):
            raise RuntimeError("Call fit(jobs) before score(candidate).")

        candidate_vector = self.vectorizer.transform(
            [profile_to_text(candidate)]
        )
        return cosine_similarity(
            candidate_vector, self.job_vectors
        ).ravel()


def skill_coverage(
    candidate_skills: list[str],
    job_skills: list[str],
) -> SkillCoverageResult:
    """Measure coverage of detected job skills by declared candidate skills.

    Args:
        candidate_skills: Candidate skill names with aliases normalized upstream.
        job_skills: Detected job skill names using the same naming conventions.

    Returns:
        A dictionary with skill_score, matched_skills, and unverified_skills.
        The score is matched distinct labels divided by distinct job labels,
        or None when no job labels are available. Explanations are sorted lists
        of stripped, case-folded labels; duplicates and blank labels are ignored.

    Notes:
        This function neither extracts skills nor maps aliases. Unverified
        labels are not proven absent from the candidate. Complete coverage
        applies only to detected labels, not all requirements of the job.
    """
    candidate_set = {
        skill.strip().casefold()
        for skill in candidate_skills
        if skill.strip()
    }
    required_set = {
        skill.strip().casefold()
        for skill in job_skills
        if skill.strip()
    }

    matched = sorted(candidate_set & required_set)
    unverified = sorted(required_set - candidate_set)

    return {
        "skill_score": (
            len(matched) / len(required_set) if required_set else None
        ),
        "matched_skills": matched,
        "unverified_skills": unverified,
    }
