"""TF-IDF matching for historical job descriptions."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def profile_to_text(profile):
    """Combine declared skills, education, and experience."""
    return " ".join([
        " ".join(profile.get("skills", [])),
        profile.get("education", ""),
        profile.get("experience", ""),
    ])


class TextMatcher:
    """Prepare job vectors once and reuse them for candidate scoring."""

    def fit(self, jobs):
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

    def score(self, candidate):
        """Return one text similarity score per job, in original row order."""
        if not hasattr(self, "job_vectors"):
            raise RuntimeError("Call fit(jobs) before score(candidate).")

        candidate_vector = self.vectorizer.transform(
            [profile_to_text(candidate)]
        )
        return cosine_similarity(
            candidate_vector, self.job_vectors
        ).ravel()


def skill_coverage(candidate_skills, job_skills):
    """Compare normalized skill labels; no extraction or alias mapping."""
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
