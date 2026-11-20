"""Evaluate JobScope methods on the same fully annotated 15-job pools.

Run from the repository root after three annotators have completed their scores.
Jessy's AI-assisted draft must be personally reviewed before results are reported
as an independent three-human-annotator evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.feature_extraction import extract_skills
from src.matching import TextMatcher
from src.recommendation import score_jobs
from src.evaluation import evaluate_ranked_pool

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data/processed/annotation_batches"
OUT = ROOT / "data/processed/evaluation_results"
ANNOTATORS = ("jessy", "aileen", "damir")
METHODS = {"text": "text_score", "skill": "skill_score", "hybrid": "hybrid_score"}


def load_annotations():
    keys = ["candidate_id", "job_id"]
    combined = None
    for name in ANNOTATORS:
        path = BATCH / f"annotations_{name}.csv"
        frame = pd.read_csv(path)
        required = set(keys + ["relevance_score"])
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns in {path}")
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate candidate-job pairs in {path}")
        frame["relevance_score"] = pd.to_numeric(frame["relevance_score"], errors="coerce")
        if frame["relevance_score"].isna().any() or not frame["relevance_score"].isin([0, 1, 2]).all():
            raise ValueError(f"Unrated or invalid labels in {path}")
        part = frame[keys + ["relevance_score"]].rename(columns={"relevance_score": f"score_{name}"})
        if combined is None:
            combined = part
        else:
            combined = combined.merge(part, on=keys, how="outer", validate="one_to_one", indicator=True)
            if not combined["_merge"].eq("both").all():
                raise ValueError("All annotators must score identical candidate-job pairs")
            combined = combined.drop(columns="_merge")
    rating_cols = [f"score_{name}" for name in ANNOTATORS]
    combined["relevance_score"] = combined[rating_cols].median(axis=1).astype(int)
    combined["all_agree"] = combined[rating_cols].nunique(axis=1).eq(1)
    return combined


def main():
    labels = load_annotations()
    if labels.groupby("candidate_id").size().ne(15).any():
        raise ValueError("Expected exactly 15 judged jobs for every candidate")

    OUT.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT / "provisional_consensus.csv", index=False)
    print(f"Three-way exact agreement: {labels.all_agree.mean():.1%}")
    for i, first in enumerate(ANNOTATORS):
        for second in ANNOTATORS[i + 1:]:
            a, b = labels[f"score_{first}"], labels[f"score_{second}"]
            print(f"{first} vs {second}: agreement={(a == b).mean():.1%}; "
                  f"quadratic weighted kappa={cohen_kappa_score(a, b, labels=[0,1,2], weights='quadratic'):.3f}")

    jobs_path = ROOT / "data/sample/sample_jobs.csv"
    jobs = pd.read_csv(jobs_path)
    jobs["description"] = jobs["description"].fillna("")
    jobs["skill_labels"] = jobs["description"].apply(extract_skills)
    if jobs["job_id"].duplicated().any():
        raise ValueError("Job IDs in the sample dataset must be unique")
    matcher = TextMatcher().fit(jobs)

    profiles = {}
    for name in ANNOTATORS:
        frame = pd.read_csv(BATCH / f"annotations_{name}.csv")
        for candidate_id, group in frame.groupby("candidate_id"):
            first = group.iloc[0]
            profile = {
                "candidate_id": candidate_id,
                "skills": [s.strip() for s in str(first["candidate_skills"]).split(",") if s.strip()],
                "education": str(first["candidate_education"]),
                "experience": str(first["candidate_experience"]),
            }
            if candidate_id in profiles and profile != profiles[candidate_id]:
                raise ValueError(f"Inconsistent profile: {candidate_id}")
            profiles[candidate_id] = profile

    reports, all_rankings = [], []
    for candidate_id, judged in labels.groupby("candidate_id"):
        pool_ids = set(judged["job_id"])
        if not pool_ids.issubset(set(jobs["job_id"])):
            raise ValueError(f"Annotated jobs missing from sample data: {candidate_id}")
        # Fit and score on the original full sample dataset, then restrict
        # ranking to the same 15-job judged pool for fair comparison.
        scored = score_jobs(profiles[candidate_id], jobs, matcher=matcher,
                            skills_column="skill_labels", text_weight=0.5)
        pool = scored[scored["job_id"].isin(pool_ids)].copy()
        if len(pool) != 15:
            raise ValueError(f"Expected 15 unique jobs in evaluated pool for {candidate_id}")
        for method, column in METHODS.items():
            ranked = pool.dropna(subset=[column]).sort_values(
                [column, "job_id"], ascending=[False, True]
            ).copy()
            ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
            ranked.insert(0, "method", method)
            ranked.insert(0, "candidate_id", candidate_id)
            ranked = ranked.merge(judged[["job_id", "relevance_score"]], on="job_id",
                                  how="left", validate="one_to_one")
            all_rankings.append(ranked[["candidate_id", "method", "rank", "job_id", "title", column, "relevance_score"]].rename(columns={column: "model_score"}))
            # A method can exclude rows with no detected skills; then a complete
            # 15-job ranking is not available, so pool-based metrics are NA.
            row = {"candidate_id": candidate_id, "method": method,
                   "judged_pool_size": len(pool), "ranked_size": len(ranked)}
            for k in (5, 10):
                if len(ranked) == len(pool):
                    metrics = evaluate_ranked_pool(
                        ranked[["job_id"]], judged[["job_id", "relevance_score"]], k=k
                    )
                    row[f"precision_at_{k}"] = metrics["precision_at_k"]
                    row[f"ndcg_at_{k}"] = metrics["ndcg_at_k"]
                else:
                    row[f"precision_at_{k}"] = np.nan
                    row[f"ndcg_at_{k}"] = np.nan
            row["note"] = "Full judged pool" if len(ranked) == len(pool) else "Incomplete ranking; metrics omitted"
            reports.append(row)
    report = pd.DataFrame(reports)
    report.to_csv(OUT / "model_comparison_by_profile.csv", index=False)
    pd.concat(all_rankings, ignore_index=True).to_csv(OUT / "ranked_judged_jobs.csv", index=False)
    print("\nModel comparison (provisional until human adjudication):")
    print(report.to_string(index=False))
    print(f"\nOutput folder: {OUT}")


if __name__ == "__main__":
    main()
