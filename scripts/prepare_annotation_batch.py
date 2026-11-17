
"""Prepare reproducible, blinded candidate-job annotation batches."""

import random
from pathlib import Path

import pandas as pd

from src.feature_extraction import extract_skills
from src.matching import TextMatcher
from src.recommendation import recommend_top_k


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/annotation_batches"
SEED = 1007
PAIRS_PER_CANDIDATE = 15

PROFILES = [
    {
        "candidate_id": "candidate_ds",
        "skills": ["Python", "SQL", "Machine Learning", "Data Analysis"],
        "education": "Master's in Data Science",
        "experience": "Data analytics, NLP, and statistical modeling",
    },
    {
        "candidate_id": "candidate_finance",
        "skills": ["Excel", "Financial Modeling", "SQL", "Risk Management"],
        "education": "Bachelor's in Finance",
        "experience": "Financial analysis, accounting, and risk assessment",
    },
    {
        "candidate_id": "candidate_health",
        "skills": ["Patient Care", "Healthcare", "Communication"],
        "education": "Bachelor's in Health Sciences",
        "experience": "Clinical support and healthcare administration",
    },
]


def main():
    rng = random.Random(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    jobs = pd.read_csv(ROOT / "data/sample/sample_jobs.csv")
    jobs["description"] = jobs["description"].fillna("")
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)

    assert jobs["job_id"].is_unique
    assert len(jobs) >= PAIRS_PER_CANDIDATE

    matcher = TextMatcher().fit(jobs)

    annotations = []
    manifest = []

    for profile in PROFILES:
        candidate_id = profile["candidate_id"]
        selected = {}

        for method in ["text", "skill", "hybrid"]:
            ranked = recommend_top_k(
                profile,
                jobs,
                k=5,
                matcher=matcher,
                method=method,
                skills_column="skill_labels",
                text_weight=0.5,
            )

            for job_id in ranked["job_id"]:
                job_id = int(job_id)
                selected.setdefault(job_id, set()).add(method)

        # Add random jobs if overlapping Top-5 lists yield fewer than 15.
        remaining = [
            int(job_id)
            for job_id in jobs["job_id"]
            if int(job_id) not in selected
        ]
        needed = PAIRS_PER_CANDIDATE - len(selected)

        for job_id in rng.sample(remaining, needed):
            selected[job_id] = {"random"}

        # Randomize presentation order to avoid revealing model ranks.
        selected_ids = list(selected)
        rng.shuffle(selected_ids)

        job_lookup = jobs.set_index("job_id")

        for job_id in selected_ids:
            job = job_lookup.loc[job_id]

            annotations.append({
                "candidate_id": candidate_id,
                "candidate_skills": ", ".join(profile["skills"]),
                "candidate_education": profile["education"],
                "candidate_experience": profile["experience"],
                "job_id": job_id,
                "job_title": job["title"],
                "job_description": job["description"],
                "relevance_score": "",
                "notes": "",
            })

            manifest.append({
                "candidate_id": candidate_id,
                "job_id": job_id,
                "selection_sources": ",".join(sorted(selected[job_id])),
                "sampling_seed": SEED,
            })

    base = pd.DataFrame(annotations)

    # Save independent copies for blinded annotation.
    for annotator in ["jessy", "aileen"]:
        output = base.copy()
        output.insert(0, "annotator", annotator)

        path = OUTPUT / f"annotations_{annotator}.csv"
        output.to_csv(path, index=False)
        print(f"Created: {path}")

    # Keep model selection information separate from annotator files.
    pd.DataFrame(manifest).to_csv(
        OUTPUT / "sampling_manifest.csv",
        index=False,
    )

    print(f"Candidate profiles: {len(PROFILES)}")
    print(f"Unique candidate-job pairs: {len(base)}")
    print(f"Sampling seed: {SEED}")
    print("Annotation batches contain no model scores or ranks.")


if __name__ == "__main__":
    main()
